import numpy as np
import torch
import torch.optim as optim
import math
import random
import os
import csv

from rnn_policy import RNNPolicy
from agent import Agent


# --- ENVIRONMENT ---
class SearchEnv:
    def __init__(self, config, device):
        self.conf = config
        self.device = device
        self.agents = []
        self.targets = []
        self.found_targets = []
        # [x,y,cos(phi),sin(phi),d_nearest_target,d_nearest_agent, cos(rel_angle_to_target), sin(rel_angle_to_target)]
        self.input_dim = 8
        self.output_dim = 3
        self.coverage_grid = np.zeros((self.conf.WIDTH, self.conf.HEIGHT)) # added by us 
        
        self.policy_net = RNNPolicy(self.input_dim, self.output_dim, self.conf.hidden_size, device).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.conf.lr)
        
        self.epsilon = self.conf.epsilon_start
        self.batch_hidden = None
        self.saved_agent_params = [] 

    def getstarts(self, nr_ag):
        positions = random.choices(['t','b','l','r'],k=nr_ag)
        starts=[]
        for p in positions:
            if p == 't':
                starts.append([random.randrange(0,self.conf.WIDTH,1),self.conf.HEIGHT])
            elif p == 'b':
                starts.append([random.randrange(0,self.conf.WIDTH,1),0])
            elif p == 'l':
                starts.append([0,random.randrange(0,self.conf.HEIGHT,1)])
            elif p == 'r':
                starts.append([self.conf.WIDTH,random.randrange(0,self.conf.HEIGHT,1)])
        return starts
        
    def reset(self):
        self.agents = []
        if self.conf.start_mode == 'centralized':
            starts = self.getstarts(1) # Starting positions 
            for i in range(self.conf.Nr):
                pos = starts[0]
                ag = Agent(i, pos[0], pos[1], self.conf)
                self.agents.append(ag)
        else:
            starts = self.getstarts(self.conf.Nr) # Starting positions 
            for i in range(self.conf.Nr):
                pos = starts[i % len(starts)]
                ag = Agent(i, pos[0], pos[1], self.conf)
                self.agents.append(ag)
            
        self.batch_hidden = self.policy_net.init_hidden(self.conf.Nr)
            
        self.targets = []
        for _ in range(self.conf.Nt):
            self.targets.append((random.uniform(5, 95), random.uniform(20, 95)))
        self.found_targets = [False] * self.conf.Nt
        self.coverage_grid = np.zeros((self.conf.WIDTH, self.conf.HEIGHT), dtype=np.uint8)
        return self.get_observations()

    def get_observations(self):
        obs_list = []
        for ag in self.agents:
            d_t = 100.0
            nearest_t = None

            # Distance to nearest UNFOUND target
            for i, t in enumerate(self.targets):
                if not self.found_targets[i]:
                    d = math.sqrt((ag.x - t[0])**2 + (ag.y - t[1])**2)
                    if d < d_t:
                        d_t = d
                        nearest_t = t
            d_a = 100.0
            # Distance to nearest agent
            for other in self.agents:
                if other.id != ag.id:
                    d = math.sqrt((ag.x - other.x)**2 + (ag.y - other.y)**2)
                    d_a = min(d_a, d)
            # Normalize observations
            # Relative bearing to nearest target (in agent's frame)
            if nearest_t is None:
                rel_cos, rel_sin = 0.0, 0.0
            else:
                dx = nearest_t[0] - ag.x
                dy = nearest_t[1] - ag.y
                ang_to_t = math.atan2(dy, dx)
                rel = ang_to_t - ag.phi
                rel_cos, rel_sin = math.cos(rel), math.sin(rel)

            obs_list.append([
                ag.x/self.conf.WIDTH,
                ag.y/self.conf.HEIGHT,
                math.cos(ag.phi),
                math.sin(ag.phi),
                d_t/100.0,
                d_a/100.0,
                rel_cos,
                rel_sin
            ])
        return torch.tensor(obs_list, dtype=torch.float32, device=self.device)
    
    def calculate_area_coverage(self, agent):
        """
        Calculates reward based on how many NEW grid cells are revealed
        within the agent's detection radius (Rv).
        """
        # 1. Define Bounding Box
        x_min = max(0, int(agent.x - self.conf.Rv))
        x_max = min(self.conf.WIDTH, int(agent.x + self.conf.Rv + 1))
        y_min = max(0, int(agent.y - self.conf.Rv))
        y_max = min(self.conf.HEIGHT, int(agent.y + self.conf.Rv + 1))

        # 2. Extract local grid slice
        # Shape comes out as (width_span, height_span) e.g., (29, 15)
        local_grid = self.coverage_grid[x_min:x_max, y_min:y_max]
        
        # 3. Create Distance Mask
        x_idx, y_idx = np.ogrid[x_min:x_max, y_min:y_max]
        
        # Calculate squared distance
        # x_idx broadcasts to (N, 1) and y_idx to (1, M), resulting in (N, M)
        dist_sq = (x_idx - agent.x)**2 + (y_idx - agent.y)**2
        
        # 4. Create Boolean Mask
        new_cells_mask = (dist_sq <= self.conf.Rv**2) & (local_grid == 0)
        
        # 5. Update and Reward
        num_new_cells = np.sum(new_cells_mask)
        
        # Apply mask to the global grid
        self.coverage_grid[x_min:x_max, y_min:y_max][new_cells_mask] = 1
        
        if num_new_cells > 0:
            return num_new_cells * self.conf.r_cover_scale
        else:
            return self.conf.r_cover_penalty

    def select_actions_batch(self, obs_batch):
        # Improved Action Selection Strategy [cite: 368]
        h_values, new_hidden = self.policy_net(obs_batch, self.batch_hidden)
        self.batch_hidden = new_hidden.detach()
        h_probs = torch.softmax(h_values, dim=1) 
        
        N = self.output_dim
        p_values = (1 - self.epsilon) * h_probs + (self.epsilon / N)
        
        if self.epsilon == 0:
            actions_indices = torch.argmax(p_values, dim=1)
        else:
            # Beta distribution logic for preference updates [cite: 374]
            alphas = torch.tensor([ag.alpha for ag in self.agents], device=self.device).unsqueeze(1).expand(-1, N)
            betas  = torch.tensor([ag.beta for ag in self.agents], device=self.device).unsqueeze(1).expand(-1, N)
            m = torch.distributions.Beta(alphas, betas)
            noise_matrix = m.sample()
            
            # Equation (21)
            scores = self.conf.omega2 * p_values.detach() + (1 - self.conf.omega2) * noise_matrix
            actions_indices = torch.argmax(scores, dim=1)
            
            # Update alpha/beta based on comparison with random number
            actions_cpu = actions_indices.tolist()
            noise_cpu = noise_matrix.cpu().numpy()
            for i, action_idx in enumerate(actions_cpu):
                chosen_lambda = noise_cpu[i, action_idx]
                if random.random() < chosen_lambda:
                    self.agents[i].alpha += 1.0
                else:
                    self.agents[i].beta += 1.0

        chosen_log_probs = torch.log(h_probs.gather(1, actions_indices.unsqueeze(1))).squeeze(1)
        return actions_indices.tolist(), chosen_log_probs

    def step(self, actions, current_step_num, max_steps):
        rewards = []
        active_targets = [t for i, t in enumerate(self.targets) if not self.found_targets[i]]
        
        # --- 1. Calculate Individual Agent Rewards ---
        for i, ag in enumerate(self.agents):
            neighbors = [o for j, o in enumerate(self.agents) if i != j]
            
            # Get physical penalties (Boundary, Collision)
            r_phys = ag.move(actions[i], neighbors)
            
            # Get Odor reward
            r_odor = ag.get_odor_reward(active_targets)
            
            # Movement cost
            r_move = self.conf.rm
            
            # --- NEW: Area Coverage Reward ---
            # Call the separate function to handle radius exploration
            r_cover = 0.0
            if getattr(self.conf, "use_coverage_reward", False):
                r_cover = self.calculate_area_coverage(ag)
            
            # Sum individual rewards
            rewards.append(r_move + r_phys + r_odor + r_cover)
            
        # --- 2. Calculate Global Search Rewards (Target Finding) ---
        step_found_count = 0
        for t_idx, t in enumerate(self.targets):
            if self.found_targets[t_idx]: continue
            for ag in self.agents:
                # Check target detection (Distance <= Rv)
                d = math.sqrt((ag.x - t[0])**2 + (ag.y - t[1])**2)
                if d <= self.conf.Rv:
                    # Sensor detection probability check
                    if random.random() < self.conf.pd:
                        self.found_targets[t_idx] = True
                        step_found_count += 1
                        break
        
        found_reward = (step_found_count * self.conf.r1)
        all_found = all(self.found_targets)
        
        # --- 3. Calculate Terminal Rewards (Presence Penalty) ---
        r_presence = 0
        if all_found:
            found_reward += self.conf.r2
            r_presence = - (current_step_num / max_steps) * self.conf.rho
        elif current_step_num >= max_steps:
             r_presence = -1.0 * self.conf.rho

        # Distribute shared rewards
        final_rewards = [r + found_reward + r_presence for r in rewards]
        
        return final_rewards, all_found

    # --- CHECKPOINTING UTILS ---
    def save_checkpoint(self, episode, history_rewards, history_found):
        if not os.path.exists(self.conf.ckpt_dir):
            os.makedirs(self.conf.ckpt_dir)
            
        checkpoint = {
            'episode': episode,
            'model_state': self.policy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'rewards': history_rewards,
            'found': history_found,
            'agent_params': [{'alpha': ag.alpha, 'beta': ag.beta} for ag in self.agents] 
        }
        path = os.path.join(self.conf.ckpt_dir, f"ckpt_ep_{episode}.pth")
        torch.save(checkpoint, path)
        torch.save(checkpoint, os.path.join(self.conf.ckpt_dir, "ckpt_latest.pth"))
        
    def load_checkpoint(self):
        latest_path = os.path.join(self.conf.ckpt_dir, "ckpt_ep_999.pth")
        if not os.path.exists(latest_path):
            print("No checkpoint found. Starting from scratch.")
            return 0, [], []
            
        print(f"Loading checkpoint from {latest_path}...")
        checkpoint = torch.load(latest_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        start_episode = 1000#checkpoint['episode'] + 1
        history_rewards = checkpoint['rewards']
        history_found = checkpoint['found']
        self.saved_agent_params = checkpoint.get('agent_params', [])
        
        print(f"Resuming from Episode {start_episode}")
        return start_episode, history_rewards, history_found

    def log_to_csv(self, episode, avg_reward, avg_found):
        file_exists = os.path.isfile(self.conf.log_file)
        with open(self.conf.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Episode', 'Epsilon', 'Avg_Reward', 'Avg_Found'])
            writer.writerow([episode, f"{self.epsilon:.4f}", f"{avg_reward:.2f}", f"{avg_found:.2f}"])

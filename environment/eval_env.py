import numpy as np
import torch
import torch.optim as optim
import math
import random
import os
import csv

from rnn_policy import RNNPolicy
from agent import Agent


# --- EVALUATION ENVIRONMENT ---
class EvalEnv:
    def __init__(self, config, device):
        self.conf = config
        self.agents = []
        self.targets = []
        self.found_targets = []
        self.input_dim = 6
        self.output_dim = 3
        self.device = device
        
        self.policy_net = RNNPolicy(self.input_dim, self.output_dim, self.conf.hidden_size, self.device).to(self.device)
        self.load_model()
        self.policy_net.eval()
        self.batch_hidden = None

    def load_model(self):
        path = os.path.join(self.conf.ckpt_dir, "ckpt_latest.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}. Run training first.")
        
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location=self.device,weights_only=False)
        self.policy_net.load_state_dict(checkpoint['model_state'])

    def getstarts(self, nr_ag, paperdef=None):
        if paperdef == "distributed":
            return ([(0,0),(50,0),(100,0)]*nr_ag)[:nr_ag]
        if paperdef == "centralized":
            return [(50,0)]*nr_ag
        
        positions = random.choices(['t','b','l','r'], k=nr_ag)
        starts=[]
        for p in positions:
            if p == 't': starts.append([random.randrange(0,100,1), 100])
            elif p == 'b': starts.append([random.randrange(0,100,1), 0])
            elif p == 'l': starts.append([0, random.randrange(0,100,1)])
            elif p == 'r': starts.append([100, random.randrange(0,100,1)])
        return starts

    def reset(self):
        self.agents = []
        starts = self.getstarts(self.conf.Nr, paperdef="centralized") 
        for i in range(self.conf.Nr):
            pos = starts[i % len(starts)]
            ag = Agent(i, pos[0], pos[1], self.conf)
            self.agents.append(ag)
            
        self.batch_hidden = self.policy_net.init_hidden(self.conf.Nr)
            
        self.targets = []
        for _ in range(self.conf.Nt):
            self.targets.append((random.uniform(5, 95), random.uniform(5, 95)))
        self.found_targets = [False] * self.conf.Nt
        
        return self.get_observations()

    def get_observations(self):
        obs_list = []
        for ag in self.agents:
            d_t = 100.0
            for i, t in enumerate(self.targets):
                if not self.found_targets[i]:
                    d = math.sqrt((ag.x - t[0])**2 + (ag.y - t[1])**2)
                    d_t = min(d_t, d)
            d_a = 100.0
            for other in self.agents:
                if other.id != ag.id:
                    d = math.sqrt((ag.x - other.x)**2 + (ag.y - other.y)**2)
                    d_a = min(d_a, d)
            obs_list.append([ag.x/self.conf.WIDTH, ag.y/self.conf.HEIGHT, math.cos(ag.phi), math.sin(ag.phi), d_t/100.0, d_a/100.0])
        return torch.tensor(obs_list, dtype=torch.float32, device=self.device)

    def select_actions(self, obs_batch):
        with torch.no_grad():
            h_values, new_hidden = self.policy_net(obs_batch, self.batch_hidden)
            self.batch_hidden = new_hidden.detach()
            h_probs = torch.softmax(h_values, dim=1)
            actions_indices = torch.argmax(h_probs, dim=1)
        return actions_indices.tolist()

    # --- UPDATED STEP FUNCTION TO CALCULATE REWARDS ---
    def step(self, actions, current_step, max_steps):
        rewards = []
        active_targets = [t for i, t in enumerate(self.targets) if not self.found_targets[i]]
        
        # 1. Individual Agent Physical/Odor Rewards
        for i, ag in enumerate(self.agents):
            neighbors = [o for j, o in enumerate(self.agents) if i!=j]
            r_phys = ag.move(actions[i], neighbors)
            r_odor = ag.get_odor_reward(active_targets)
            rewards.append(self.conf.rm + r_phys + r_odor)
            
        # 2. Global Target Finding Rewards
        step_found_count = 0
        for t_idx, t in enumerate(self.targets):
            if self.found_targets[t_idx]: continue
            for ag in self.agents:
                d = math.sqrt((ag.x - t[0])**2 + (ag.y - t[1])**2)
                if d <= self.conf.Rv:
                    if random.random() < self.conf.pd:
                        self.found_targets[t_idx] = True
                        step_found_count += 1
                        break
        
        found_reward = (step_found_count * self.conf.r1)
        all_found = all(self.found_targets)
        
        # 3. Terminal/Presence Penalties
        r_presence = 0
        if all_found:
            found_reward += self.conf.r2
            # Finished early: Penalty proportional to time taken
            r_presence = - (current_step / max_steps) * self.conf.rho
        elif current_step >= max_steps:
             # Timeout: Max penalty
             r_presence = -1.0 * self.conf.rho

        # Distribute shared rewards
        final_agent_rewards = [r + found_reward + r_presence for r in rewards]
        total_step_reward = sum(final_agent_rewards) # Sum of all agents for this step
        
        return total_step_reward, all_found, step_found_count
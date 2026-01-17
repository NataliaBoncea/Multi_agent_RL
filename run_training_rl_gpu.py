import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import csv

from config.train_config import TrainConfig
from rnn_policy import RNNPolicy
from agent import Agent
from environment.search_env import SearchEnv


# --- GPU SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    

# --- RUN SIMULATION ---
def run_simulation():
    config = TrainConfig()
    env = SearchEnv(config, device)
    start_episode, history_rewards, history_found = env.load_checkpoint()
    
    history = [] 
    step_counts = []
    found_counts = []
    agent_trajectories = [[] for _ in range(config.Nr)]
    
    print(f"Starting Training on {device}...")
    
    for ep in range(start_episode, config.num_episodes): 
        obs = env.reset()
        if hasattr(env, 'saved_agent_params') and env.saved_agent_params:
            for i, params in enumerate(env.saved_agent_params):
                if i < len(env.agents):
                    env.agents[i].alpha = params['alpha']
                    env.agents[i].beta = params['beta']
            del env.saved_agent_params
            
        ep_log_probs = []
        ep_rewards = []
        
        record = (ep == config.num_episodes - 1)
        current_max_steps = config.test_steps if record else config.max_steps
        
        for step in range(current_max_steps):
            
            actions, log_probs = env.select_actions_batch(obs)
            # Pass step number for Presence Penalty calculation
            rewards, done = env.step(actions, step + 1, current_max_steps)
            
            ep_log_probs.append(log_probs)
            
            # Summing rewards implies a fully cooperative goal (Team Reward)
            ep_rewards.append(sum(rewards))
            obs = env.get_observations()

            if record:
                for i, ag in enumerate(env.agents):
                    agent_trajectories[i].append((ag.x, ag.y))
                
                history.append({
                    'agents': [(a.x, a.y) for a in env.agents],
                    'targets': [f for f in env.found_targets] 
                })
                step_counts.append(step + 1)
                found_counts.append(sum(env.found_targets))
            
            if done: 
                break
        
        # Add padding frames for the animation if it finished early
        if record and len(history) > 0:
            last_frame = history[-1]
            last_found = found_counts[-1]
            last_step = step_counts[-1]
            for s in range(20):
                history.append(last_frame)
                step_counts.append(last_step)
                found_counts.append(last_found)
                for i in range(config.Nr):
                    agent_trajectories[i].append(agent_trajectories[i][-1])

        total_reward = sum(ep_rewards)
        history_rewards.append(total_reward)
        history_found.append(sum(env.found_targets))
        
        # Anneal epsilon
        env.epsilon = max(config.epsilon_end, env.epsilon - config.epsilon_decay)
        
        # --- REINFORCE UPDATE ---
        if len(ep_rewards) > 0:
            R = 0
            returns = []
            for r in reversed(ep_rewards):
                R = r + config.gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, device=device)
            
            # Normalize with safety check to prevent NaN if std is 0
            if returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = returns - returns.mean()
            
            log_probs_tensor = torch.stack(ep_log_probs)
            returns_expanded = returns.unsqueeze(1).expand_as(log_probs_tensor)
            
            loss = -(log_probs_tensor * returns_expanded).sum()
            
            env.optimizer.zero_grad()
            loss.backward()
            env.optimizer.step()
        
        if ep % config.save_freq == 0: 
            avg_rew = np.mean(history_rewards[-50:]) if len(history_rewards) >= 50 else np.mean(history_rewards)
            avg_found = np.mean(history_found[-50:]) if len(history_found) >= 50 else np.mean(history_found)
            print(f"Episode {ep} | Epsilon: {env.epsilon:.3f} | AVG Reward: {avg_rew:.2f} | AVG Found: {avg_found:.1f}")
            env.save_checkpoint(ep, history_rewards, history_found)
            env.log_to_csv(ep, avg_rew, avg_found)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_rewards)
    plt.title("Total Reward per Episode")
    plt.savefig('training_metrics_final.png')
    
    print("Generating Animation...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.set_title("Multi-Agent Cooperative Search (Improved)")
    
    tx = [t[0] for t in env.targets]
    ty = [t[1] for t in env.targets]
    scat_targets = ax.scatter(tx, ty, s=50, marker='^', c='black', label='Targets')
    
    agent_plots = []
    trace_plots = []
    colors = ['g', 'b', 'm']
    for i in range(config.Nr):
        tp, = ax.plot([], [], '--', c=colors[i], alpha=0.3, linewidth=1)
        trace_plots.append(tp)
        p, = ax.plot([], [], 'o', c=colors[i], label=f'Agent {i}')
        agent_plots.append(p)
        
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    found_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(loc='lower right')
    
    def update(frame):
        if frame >= len(history): return agent_plots + trace_plots + [scat_targets, step_text, found_text]
        
        data = history[frame]
        for i, pos in enumerate(data['agents']):
            agent_plots[i].set_data([pos[0]], [pos[1]])
            xs = [p[0] for p in agent_trajectories[i][:frame+1]]
            ys = [p[1] for p in agent_trajectories[i][:frame+1]]
            trace_plots[i].set_data(xs, ys)
        
        current_colors = ['tab:orange' if f else 'black' for f in data['targets']]
        scat_targets.set_color(current_colors)
        step_text.set_text(f"Step: {step_counts[frame]}")
        found_text.set_text(f"Found: {found_counts[frame]}/{config.Nt}")
        
        return agent_plots + trace_plots + [scat_targets, step_text, found_text]

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=30, blit=False)
    ani.save('simulation.gif', writer='pillow', fps=20)
    print("Training Complete. GIF Saved.")

if __name__ == "__main__":
    run_simulation()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import math
import random
import os

from config.eval_config import EvalConfig
from rnn_policy import RNNPolicy
from agent import Agent
from environment.eval_env import EvalEnv


# --- GPU SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation using device: {device}")


# --- MAIN EVALUATION LOOP ---
def run_evaluation():
    config = EvalConfig()
    env = EvalEnv(config, device)
    
    print(f"\n--- Starting Evaluation for {config.num_test_episodes} Episodes ---")
    
    # METRICS STORAGE
    metric_success = []      # 1 if all found, 0 otherwise
    metric_targets = []      # Count of targets found
    metric_rewards = []      # Total accumulated reward per episode
    metric_time = []         # Steps taken per episode

    # VISUALIZATION STORAGE (Last episode only)
    vis_history = []
    vis_agent_trajectories = []
    
    for ep in range(config.num_test_episodes):
        obs = env.reset()
        step = 0
        done = False
        episode_reward = 0
        
        # Reset visual storage for the current episode (it overwrites previous ones)
        vis_history = []
        vis_agent_trajectories = [[] for _ in range(config.Nr)]
        
        # Record Initial State
        vis_history.append({
            'agents': [(a.x, a.y) for a in env.agents],
            'targets': [f for f in env.found_targets], 
            'step': step,
            'found_total': sum(env.found_targets)
        })
        for i, ag in enumerate(env.agents):
            vis_agent_trajectories[i].append((ag.x, ag.y))

        while step < config.max_steps and not done:
            step += 1
            
            actions = env.select_actions(obs)
            
            # Updated step call with reward calculation
            step_reward, all_found, found_count = env.step(actions, step, config.max_steps)
            
            episode_reward += step_reward
            obs = env.get_observations()
            
            # Record Step Data
            vis_history.append({
                'agents': [(a.x, a.y) for a in env.agents],
                'targets': [f for f in env.found_targets],
                'step': step,
                'found_total': sum(env.found_targets)
            })
            for i, ag in enumerate(env.agents):
                vis_agent_trajectories[i].append((ag.x, ag.y))

            if all_found:
                done = True
        
        # --- COLLECT METRICS FOR THIS EPISODE ---
        found_total = sum(env.found_targets)
        metric_success.append(1 if found_total == config.Nt else 0)
        metric_targets.append(found_total)
        metric_rewards.append(episode_reward)
        metric_time.append(step)
        
        print(f"Episode {ep+1}: Found {found_total}/{config.Nt} | Reward: {episode_reward:.1f} | Steps: {step}")

    # --- FINAL REPORT ---
    avg_success = np.mean(metric_success) * 100
    avg_targets = np.mean(metric_targets)
    avg_reward = np.mean(metric_rewards)
    avg_time = np.mean(metric_time)

    print("\n" + "="*40)
    print("       PERFORMANCE EVALUATION REPORT       ")
    print("="*40)
    print(f"1. Mission Success Rate:   {avg_success:.2f}%")
    print(f"2. Avg Number of Targets:  {avg_targets:.2f} / {config.Nt}")
    print(f"3. Average Reward:         {avg_reward:.2f}")
    print(f"4. Average Search Time:    {avg_time:.2f} steps")
    print("="*40)

    # --- GENERATE GIF (Last Episode) ---
    print("\nGenerating GIF for the final episode...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.set_title("Evaluation Run (Last Episode)")

    tx = [t[0] for t in env.targets]
    ty = [t[1] for t in env.targets]
    scat_targets = ax.scatter(tx, ty, s=50, marker='^', c='black', label='Targets')

    agent_plots = []
    trace_plots = []
    colors = ['g', 'b', 'm', 'c', 'y']
    for i in range(config.Nr):
        c = colors[i % len(colors)]
        tp, = ax.plot([], [], '--', c=c, alpha=0.3, linewidth=1)
        trace_plots.append(tp)
        p, = ax.plot([], [], 'o', c=c, label=f'Agent {i}')
        agent_plots.append(p)

    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    def update(frame):
        if frame >= len(vis_history): frame = len(vis_history) - 1
        data = vis_history[frame]
        
        for i, pos in enumerate(data['agents']):
            agent_plots[i].set_data([pos[0]], [pos[1]])
            xs = [p[0] for p in vis_agent_trajectories[i][:frame+1]]
            ys = [p[1] for p in vis_agent_trajectories[i][:frame+1]]
            trace_plots[i].set_data(xs, ys)
            
        current_colors = ['tab:orange' if f else 'black' for f in data['targets']]
        scat_targets.set_color(current_colors)
        
        step_text.set_text(f"Step: {data['step']} | Found: {data['found_total']}/{config.Nt}")
        return agent_plots + trace_plots + [scat_targets, step_text]

    ani = animation.FuncAnimation(fig, update, frames=len(vis_history), interval=50, blit=False)
    ani.save('evaluation_last_episode.gif', writer='pillow', fps=20)
    print("Saved 'evaluation_last_episode.gif'")

if __name__ == "__main__":
    run_evaluation()
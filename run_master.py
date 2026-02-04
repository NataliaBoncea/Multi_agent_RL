import numpy as np
import torch
import os
import csv
import math
import multiprocessing
from tqdm import tqdm

# Internal Project Imports
from config.auto_config import AutoConfig
from environment.search_env import SearchEnv
from environment.eval_env import EvalEnv

# Global lock for synchronized CSV writing across parallel processes
write_lock = multiprocessing.Lock()

def train_and_evaluate_worker(task_params):
    """
    Isolated worker for a single configuration.
    1. Trains the model from scratch (or resumes).
    2. Evaluates the finished model.
    """
    width, agents, mode, targets, report_path = task_params
    
    # Initialize unique CUDA context for this specific process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    master_cfg = AutoConfig()
    # Updated to pass 'targets' to the scenario generator
    cfg = master_cfg.get_scenario(width, agents, mode, targets)

    # Ensure the checkpoint directory exists immediately
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # --- PHASE 1: TRAINING ---
    # SearchEnv handles rewards (including Area Coverage) and Beta-noise exploration
    env_train = SearchEnv(cfg, device)
    start_episode, history_rewards, history_found = env_train.load_checkpoint()
    
    for ep in range(start_episode, cfg.num_episodes): 
        obs = env_train.reset()
        ep_log_probs, ep_rewards = [], []
        
        for step in range(cfg.max_steps):
            # RNN Policy hidden state is managed within the environment batch
            actions, log_probs = env_train.select_actions_batch(obs)
            
            # Step includes Odor reward and Presence Penalty
            rewards, done = env_train.step(actions, step + 1, cfg.max_steps)
            
            ep_log_probs.append(log_probs)
            # Fully cooperative team reward summation
            ep_rewards.append(sum(rewards))
            obs = env_train.get_observations()
            if done: break
        
        # Policy Gradient (REINFORCE) Update logic
        if len(ep_rewards) > 1:
            R = 0
            returns = []
            for r in reversed(ep_rewards):
                R = r + cfg.gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, device=device)
            if returns.numel() > 1 and returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = returns - returns.mean()
            
            loss = -(torch.stack(ep_log_probs) * returns.unsqueeze(1)).sum()
            env_train.optimizer.zero_grad()
            loss.backward()
            env_train.optimizer.step()
        
        # Linear annealing of the epsilon exploration rate
        env_train.epsilon = max(cfg.epsilon_end, env_train.epsilon - cfg.epsilon_decay)
        
        # Add this inside the 'for ep in range' loop of train_and_evaluate_worker
        if ep % 5 == 0:
            print(f"\r[Worker map_{width}_ag_{agents}_tg_{targets}_{mode} Ep: {ep}/{cfg.num_episodes} | Epsilon: {env_train.epsilon:.2f}", end="", flush=True)
    
        if ep % cfg.save_freq == 0 or ep == cfg.num_episodes - 1:
            env_train.save_checkpoint(ep, history_rewards, history_found)

    # --- PHASE 2: EVALUATION ---
    # Initialized ONLY after training completes to ensure model weights exist
    env_eval = EvalEnv(cfg, device)
    metric_success, metric_targets = [], []
    
    for ep in range(cfg.num_test_episodes):
        obs = env_eval.reset()
        step, done = 0, False
        while step < cfg.max_steps and not done:
            step += 1
            # Deterministic argmax selection for evaluation
            actions = env_eval.select_actions(obs)
            _, all_found, _ = env_eval.step(actions, step, cfg.max_steps)
            obs = env_eval.get_observations()
            if all_found: done = True
        
        found_total = sum(env_eval.found_targets)
        metric_success.append(1 if found_total == cfg.Nt else 0)
        metric_targets.append(found_total)

    stats = {
        "success_rate": np.mean(metric_success) * 100,
        "avg_targets": np.mean(metric_targets)
    }

    # --- PHASE 3: LOGGING ---
    with write_lock:
        with open(report_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([width, agents, targets, mode, f"{stats['success_rate']:.2f}", f"{stats['avg_targets']:.2f}"])
    
    return f"Finished: {width}_{agents}_{targets}_{mode}"

def main():
    # 'spawn' is necessary for CUDA safety in multiprocessing on Linux
    multiprocessing.set_start_method('spawn', force=True)
    
    master_cfg = AutoConfig()
    os.makedirs("logs", exist_ok=True)
    report_path = "logs/master_experiment_report.csv"
    
    # Initialize CSV header with the new Targets column
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Grid', 'Agents', 'Targets', 'Mode', 'Success%', 'Avg_Targets'])

    # Build tasks including Target Scaling
    tasks = []
    for width in master_cfg.WIDTH_OPTIONS:
        for agents in master_cfg.AGENT_COUNT_OPTIONS:
            for targets in master_cfg.TARGET_COUNT_OPTIONS:
                for mode in master_cfg.START_MODES:
                    tasks.append((width, agents, mode, targets, report_path))

    # Parallel workers (8 is safe for 6GB VRAM and 48GB RAM)
    num_workers = 36 
    
    print(f" Starting {len(tasks)} experiments in parallel on GPU...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered used to update progress immediately upon any task completion
        for _ in tqdm(pool.imap_unordered(train_and_evaluate_worker, tasks), 
                      total=len(tasks), 
                      desc="MASTER PROGRESS"):
            pass

if __name__ == "__main__":
    main()
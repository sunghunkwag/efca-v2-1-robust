import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import numpy as np
from efca.agent import EFCAgent
from torchvision import transforms

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    Reference: tests/test_fixed_ppo.py
    """
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
    returns = advantages + values
    return advantages, returns

def ppo_update(agent, optimizer, observations, actions, old_log_probs, advantages, returns,
               clip_eps=0.2, entropy_coef=0.01, batch_size=64, epochs=4):
    """
    Perform PPO update on the agent.
    Reference: tests/test_fixed_ppo.py logic adapted for EFCAgent
    """
    # Normalize advantages (Critical step)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert to tensors
    device = agent.device
    observations = torch.tensor(np.array(observations), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(device)
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(device)
    returns = torch.tensor(np.array(returns), dtype=torch.float32).to(device)

    # Dataset size
    N = len(observations)

    for _ in range(epochs):
        # Shuffle indices
        indices = np.random.permutation(N)

        for start_idx in range(0, N, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_obs = observations[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Forward pass (we need to handle state vs vision input)
            # Assuming state input for now as per instructions for "Real PPO"
            # For EFCAgent, we need to manage hidden state 'h'.
            # Standard PPO usually assumes i.i.d batches or uses truncated BPTT.
            # Here, we will reset 'h' for each batch or treat it as non-recurrent for the update
            # if we don't have stored hidden states.
            # Given the complexity, a common simplification for recurrent PPO is to just pass h=None
            # effectively treating each step as independent or relying on the short-term memory
            # in the forward pass if we don't backprop through time across batches.
            # However, EFCAgent uses CT-LNN which is recurrent.
            # For this implementation, to stick to the "Fix" scope, we'll pass h=None
            # which means the agent re-initializes hidden state.
            # Ideally, we should store and pass hidden states, but that requires a more complex buffer.

            # NOTE: Since CT-LNN is recurrent, simply passing h=None at every update step
            # might break the temporal dependency learning.
            # However, standard PPO implementations often ignore this for simplicity unless using LSTM-PPO.
            # We will proceed with h=None for the update step to ensure code runs,
            # but acknowledge this limitation.

            dist, value, _, _, perception_loss, _ = agent(batch_obs, h=None)

            # New log probs
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # Ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # Surrogate loss
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.functional.mse_loss(value.squeeze(), batch_returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy + perception_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

            optimizer.step()

            # Update target encoder
            if hasattr(agent.perception, 'update_target_encoder'):
                agent.perception.update_target_encoder()

def collect_rollouts(env, agent, steps_per_epoch):
    """
    Collect rollouts from the environment.
    """
    obs, _ = env.reset()
    h = None

    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []

    device = agent.device

    for _ in range(steps_per_epoch):
        # Prepare observation
        # Handle both image and state inputs logic
        if isinstance(obs, np.ndarray):
             obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(device)
        else:
             # Assume it's already tensor-like or needs handling
             obs_tensor = obs.float().unsqueeze(0).to(device)

        with torch.no_grad():
            dist, value, h, _, _, _ = agent(obs_tensor, h)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        action_cpu = action.squeeze().cpu().numpy()

        # Store data
        observations.append(obs) # Store original numpy obs
        # Check if action is scalar or array (Gym expects specific format)
        # For Pendulum/Ant (Continuous), it expects array.

        if isinstance(env.action_space, gym.spaces.Discrete):
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            actions.append(action.item())
        else:
             # Clip for continuous
             # Assuming action_cpu is (action_dim,)
             # Handle the case where action_cpu is 0-d array (scalar wrapped) for 1D action space
             if action_cpu.ndim == 0:
                 action_cpu = np.expand_dims(action_cpu, axis=0)

             clipped_action = np.clip(action_cpu, -1.0, 1.0)
             next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
             actions.append(action_cpu) # Store unclipped for log_prob consistency?
             # Usually we store the action generated by the policy.

        rewards.append(reward)
        dones.append(terminated or truncated)
        values.append(value.item())
        log_probs.append(log_prob.item()) # Assuming scalar log_prob sum or mean

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            h = None

    return observations, actions, rewards, dones, values, log_probs

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train EFCA-v2.1 Agent with PPO")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Fallback default config if file not found (e.g. during tests)
        print("Config file not found, using defaults.")
        config = {
            'phase': 0,
            'training': {
                'device': 'cpu',
                'learning_rate': 3e-4,
                'num_epochs': 100,
                'steps_per_epoch': 2048,
                'batch_size': 64,
                'update_epochs': 10
            },
            # Other sections would be needed but we rely on EFCAgent handling defaults or crashing if missing
        }

    # Setup Environment
    # Use Pendulum-v1 for verification as it is standard and simple
    env_name = "Pendulum-v1"
    env = gym.make(env_name)

    # Update config based on env
    config['input_type'] = 'state'
    config['input_dim'] = env.observation_space.shape[0]

    # Ensure sub-configs exist
    if 'task_policy' not in config: config['task_policy'] = {}
    config['task_policy']['action_dim'] = env.action_space.shape[0]
    config['task_policy']['action_space_type'] = 'continuous'

    if 'h_jepa' not in config: config['h_jepa'] = {}
    config['h_jepa']['embed_dim'] = 64 # Small for fast training

    if 'bottleneck' not in config: config['bottleneck'] = {'dim': 64, 'num_slots': 4}
    if 'ct_lnn' not in config: config['ct_lnn'] = {'input_dim': 64, 'hidden_dim': 64, 'output_dim': 64}
    if 'metacontrol' not in config: config['metacontrol'] = {'input_dim': 64}

    # Initialize agent
    agent = EFCAgent(config)
    
    optimizer = optim.Adam(agent.parameters(), lr=config['training'].get('learning_rate', 3e-4))
    
    num_epochs = config['training'].get('num_epochs', 50)
    steps_per_epoch = config['training'].get('steps_per_epoch', 1000)
    
    print(f"Starting PPO Training on {env_name}...")

    for epoch in range(num_epochs):
        # 1. Collect Rollouts
        observations, actions, rewards, dones, values, old_log_probs = collect_rollouts(env, agent, steps_per_epoch)
        
        # 2. Compute GAE
        advantages, returns = compute_gae(rewards, values, dones)
        
        # 3. Update
        ppo_update(agent, optimizer, observations, actions, old_log_probs, advantages, returns)
        
        # Logging
        avg_reward = np.sum(rewards) / (np.sum(dones) + 1e-8) # Approx avg reward per episode
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Reward/Step: {np.mean(rewards):.4f}, Est. Ep Reward: {avg_reward:.2f}")

    print("Training Complete.")
    env.close()

if __name__ == "__main__":
    main()

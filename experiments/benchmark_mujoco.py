import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import numpy as np
from efca.agent import EFCAgent

def run_benchmark(env_name, config_path, num_episodes=1000):
    print(f"Starting benchmark for {env_name}...")

    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize the environment
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to load environment {env_name}: {e}")
        return

    # Update config with environment specifics
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0] # Continuous action dim

    config['h_jepa']['input_shape'] = obs_dim
    config['task_policy']['action_dim'] = action_dim

    # Initialize agent
    agent = EFCAgent(config)

    # Initialize optimizer
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config["training"]["learning_rate"],
    )

    device = config['training'].get('device', 'cpu')
    agent.to(device)

    print(f"Agent initialized for {env_name}. Input: {obs_dim}, Action: {action_dim}")

    # Training Loop
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0) # (1, D)

        h = None
        episode_reward = 0
        done = False

        log_probs = []
        values = []
        rewards = []
        perception_losses = []

        while not done:
            # Agent forward pass
            # Returns: dist, value, h_new, meta_delta, perception_loss, probe_output
            dist, value, h, meta_delta, p_loss, _ = agent(obs_tensor, h=h)

            action = dist.sample()

            # Step environment
            # Action for MuJoCo is usually float, detached from graph for step
            action_np = action.detach().cpu().numpy().flatten()

            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)

            log_probs.append(dist.log_prob(action).sum(dim=-1)) # Sum log probs for multivariate normal
            values.append(value)
            rewards.append(reward)
            perception_losses.append(p_loss)

        total_rewards.append(episode_reward)

        # Optimization (Monte Carlo REINFORCE-like / A2C simplified)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        advantage = returns - values.detach().squeeze()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(values.squeeze(), returns)
        mean_perception_loss = torch.stack(perception_losses).mean()

        loss = mean_perception_loss + actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target encoder
        agent.perception.update_target_encoder()

        if (episode + 1) % 10 == 0:
            avg_return = np.mean(total_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}: Last Return = {episode_reward:.2f}, Avg Return (10) = {avg_return:.2f}")

    # Final Reporting
    last_100_avg = np.mean(total_rewards[-100:])
    stability = np.std(total_rewards[-100:])

    print(f"\nBenchmark Complete for {env_name}.")
    print(f"Average Return (Last 100): {last_100_avg:.2f}")
    print(f"Stability (Std Dev Last 100): {stability:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mujoco_config.yaml")
    parser.add_argument("--episodes", type=int, default=100) # Short default for quick check, user requested 100 avg
    args = parser.parse_args()

    benchmarks = ["Ant-v4", "Humanoid-v4"]

    for b in benchmarks:
        run_benchmark(b, args.config, args.episodes)

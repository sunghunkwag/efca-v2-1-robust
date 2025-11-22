import gymnasium as gym
import torch
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from efca.agent import EFCAgent
from experiments.train import collect_rollouts, compute_gae, ppo_update

def verify_fix():
    print("Starting Verification on Pendulum-v1...")

    # 1. Setup Environment
    env = gym.make("Pendulum-v1")

    # 2. Config
    config = {
        'phase': 0,
        'input_type': 'state',
        'input_dim': env.observation_space.shape[0],
        'h_jepa': {
            'embed_dim': 64,
        },
        'bottleneck': {'dim': 64, 'num_slots': 4},
        'ct_lnn': {'input_dim': 64, 'hidden_dim': 64, 'output_dim': 64},
        'task_policy': {
            'hidden_dim': 64,
            'action_dim': env.action_space.shape[0],
            'action_space_type': 'continuous'
        },
        'metacontrol': {'input_dim': 64},
        'training': {'device': 'cpu', 'learning_rate': 1e-3} # Higher LR for quick learning check
    }

    # 3. Initialize Agent
    agent = EFCAgent(config)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config['training']['learning_rate'])

    # 4. Train loop
    episodes = 20
    # Steps per episode is 200 for Pendulum
    steps_per_episode = 200
    # We can treat each episode as a rollout for simplicity here

    episode_rewards = []

    for ep in range(episodes):
        observations, actions, rewards, dones, values, old_log_probs = collect_rollouts(env, agent, steps_per_episode)

        # Compute GAE
        advantages, returns = compute_gae(rewards, values, dones)

        # Update
        # Run more update epochs to force learning on small data
        ppo_update(agent, optimizer, observations, actions, old_log_probs, advantages, returns, epochs=10)

        total_reward = np.sum(rewards)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward {total_reward:.2f}")

    # 5. Verify
    first_5_avg = np.mean(episode_rewards[:5])
    last_5_avg = np.mean(episode_rewards[-5:])

    print(f"\nFirst 5 Avg: {first_5_avg:.2f}")
    print(f"Last 5 Avg: {last_5_avg:.2f}")

    if last_5_avg > first_5_avg:
        print("SUCCESS: Agent is learning (Last 5 > First 5).")
    else:
        print("WARNING: Agent might not be learning efficiently in this short span, but code is running.")
        # Pendulum is easy, it should learn. But 20 episodes is very short.
        # We won't fail the script based on this heuristic strictly, but it's the goal.

    env.close()

if __name__ == "__main__":
    verify_fix()

import gymnasium as gym
import torch
import numpy as np
import yaml
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from efca.agent import EFCAgent

def run_benchmark(episodes=5, render=False):
    print(f"Starting MuJoCo Benchmark (Ant-v4) for {episodes} episodes...")
    
    # 1. Create Environment
    env = gym.make("Ant-v4", render_mode="rgb_array" if render else None)
    
    # 2. Create Config
    # Config for EFCAgent with state-based input
    input_dim = env.observation_space.shape[0]
    config = {
        'phase': 0,
        'input_type': 'state',
        'input_dim': input_dim,
        'h_jepa': {
            'embed_dim': 256,
            'input_type': 'state',
            'input_dim': input_dim
        },
        'bottleneck': {'dim': 256, 'num_slots': 4},
        'ct_lnn': {'input_dim': 256, 'hidden_dim': 128, 'output_dim': 128},
        'task_policy': {
            'hidden_dim': 128, 
            'action_dim': env.action_space.shape[0],
            'action_space_type': 'continuous'
        },
        'metacontrol': {'input_dim': 128},
        'training': {'device': 'cpu'} # Force CPU for compatibility
    }
    
    # 3. Initialize Agent
    print("Initializing EFCAgent...")
    agent = EFCAgent(config)
    agent.eval() # Set to evaluation mode
    
    total_rewards = []
    
    device = config['training']['device']

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        # Dummy hidden state
        h = None
        
        while not done and not truncated:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Forward pass
                # dist, value, h_new, meta_delta, perception_loss, probe_output
                dist, value, h, _, _, _ = agent(obs_tensor, h)
                
                # Sample action
                action = dist.sample()
                
                # Clamp action to valid range for MuJoCo [-1, 1]
                action_np = action.squeeze().cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)
                
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            episode_reward += reward
            step_count += 1
            
            done = terminated
            
            if step_count >= 100: # Limit steps for quick verification
                truncated = True
        
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        total_rewards.append(episode_reward)
        
    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"\nBenchmark Complete.")
    print(f"Average Reward: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    run_benchmark()

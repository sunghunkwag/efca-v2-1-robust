"""
EFCA-v2.1 Phase 1 Training Script
Implements "Curious" Agent with Meta-Controller enabled

This script follows specification Section 4 (Robust Learning Algorithm)
and implements Phase 1 from Section 5 (The "Curious" Agent).

Key Features:
- Meta-controller controls epsilon_explore
- Probe network monitors internal states
- Homeostatic reward mixing (survival vs curiosity mode)
- Low-frequency meta-actions (meta_interval)
- Correlation tracking between uncertainty and exploration
"""

import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import numpy as np
from pathlib import Path
from efca.agent import EFCAgent
from efca.metacontrol.parameter_adapter import ClampedDeltaAdapter
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='EFCA Phase 1 Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase1_config.yaml',
        help='Path to the configuration file'
    )
    return parser.parse_args()


def calculate_intrinsic_reward(probe_output: torch.Tensor, config: dict) -> float:
    """
    Calculate intrinsic reward based on internal uncertainty.
    
    Simple implementation: use probe output magnitude as uncertainty measure.
    Higher probe output = higher uncertainty = higher intrinsic reward.
    
    Args:
        probe_output: Output from probe network (B, output_dim)
        config: Configuration dictionary
    
    Returns:
        float: Intrinsic reward value
    """
    # Use L2 norm of probe output as uncertainty measure
    uncertainty = torch.norm(probe_output, p=2, dim=-1).mean().item()
    
    # Scale to reasonable range [0, 1]
    intrinsic_reward = min(1.0, uncertainty / 10.0)
    
    return intrinsic_reward


def homeostatic_reward_mixing(
    reward_ext: float,
    reward_int: float,
    avg_performance: float,
    survival_threshold: float = 0.3
) -> float:
    """
    Implements Homeostatic Reward Mixing from specification Section 2.4.
    
    r_total = r_ext + β_gate(P_avg) · r_intrinsic
    
    β_gate = 0 if P_avg < Threshold_survival, else 1
    
    Logic: If agent is failing basic task (dying), shut off introspection
    to focus purely on survival (extrinsic reward).
    
    Args:
        reward_ext: External (environment) reward
        reward_int: Intrinsic (curiosity) reward
        avg_performance: Moving average of recent external rewards
        survival_threshold: Threshold below which to disable intrinsic reward
    
    Returns:
        float: Total reward
    """
    # Homeostatic gate
    if avg_performance < survival_threshold:
        # Survival Mode: Focus only on external reward
        beta_gate = 0.0
    else:
        # Curiosity Mode: Allow intrinsic exploration
        beta_gate = 1.0
    
    total_reward = reward_ext + beta_gate * reward_int
    
    return total_reward


def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify Phase 1 configuration
    if config.get('phase', 0) != 1:
        print("WARNING: Config phase is not set to 1. Setting it now.")
        config['phase'] = 1
    
    # Set device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"
    
    # Initialize environment
    env = gym.make("CartPole-v1", render_mode=None)
    
    # Set action dimension in config
    config['task_policy']['action_dim'] = env.action_space.n
    
    # Initialize agent
    agent = EFCAgent(config).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config["training"]["learning_rate"],
    )
    
    # Initialize Clamped Delta Adapter for meta-control
    delta_adapter = ClampedDeltaAdapter(
        delta_max=config['metacontrol'].get('delta_max', 0.05)
    )
    
    # Setup checkpoint directory
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints_phase1')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize training variables
    epsilon_explore = 0.1  # Starting exploration rate
    avg_performance = 0.0  # Moving average of rewards
    performance_alpha = 0.1  # EMA coefficient for performance tracking
    
    # Tracking for Phase 1 success criteria
    uncertainty_history = []
    epsilon_history = []
    
    # Preprocessing for rendering CartPole as an image
    resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def preprocess(obs):
        obs_rgb = env.render()
        obs_tensor = resize(obs_rgb).unsqueeze(0).to(device)
        return obs_tensor
    
    print("=" * 80)
    print(f"Starting Phase 1: 'Curious' Agent training loop on {device}")
    print(f"Meta-Controller: ENABLED")
    print(f"Probe Network: ACTIVE")
    print(f"Training for {config['training']['num_episodes']} episodes")
    print("=" * 80)
    
    # Training loop
    for episode in range(config["training"]["num_episodes"]):
        obs, _ = env.reset()
        h = None  # Agent handles initialization if None
        total_reward = 0
        total_ext_reward = 0
        total_int_reward = 0
        done = False
        step = 0
        
        # Episode storage
        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        perception_losses = []
        probe_outputs = []
        
        while not done:
            step += 1
            
            # Preprocess observation
            obs_tensor = preprocess(obs)
            
            # Forward pass through agent
            dist, value, h_new, meta_delta, perception_loss, probe_output = agent(
                obs_tensor, meta_state=None, hidden_state=h
            )
            
            # Epsilon-greedy exploration based on current epsilon_explore
            if np.random.rand() < epsilon_explore:
                action = env.action_space.sample()  # Random action
            else:
                action = dist.sample().item()  # Policy action
            
            # Environment step
            next_obs, reward_ext, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Calculate intrinsic reward from probe output
            if probe_output is not None:
                reward_int = calculate_intrinsic_reward(probe_output, config)
            else:
                reward_int = 0.0
            
            # Apply homeostatic reward mixing (specification Section 2.4)
            reward_total = homeostatic_reward_mixing(
                reward_ext,
                reward_int * 0.1,  # Scale intrinsic reward
                avg_performance,
                survival_threshold=config.get('survival_threshold', 0.3)
            )
            
            # Store episode data
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(reward_total)
            log_probs.append(dist.log_prob(torch.tensor([action]).to(device)))
            values.append(value)
            if perception_loss is not None:
                perception_losses.append(perception_loss)
            if probe_output is not None:
                probe_outputs.append(probe_output)
            
            # Meta-control update (low frequency)
            meta_interval = config['metacontrol'].get('meta_interval', 10)
            if step % meta_interval == 0 and meta_delta is not None:
                # Apply clamped delta control to epsilon_explore
                epsilon_explore = delta_adapter.apply_delta(
                    epsilon_explore,
                    meta_delta
                )
                
                # Track for correlation analysis
                if probe_output is not None:
                    uncertainty = torch.norm(probe_output, p=2).item()
                    uncertainty_history.append(uncertainty)
                    epsilon_history.append(epsilon_explore)
            
            total_reward += reward_total
            total_ext_reward += reward_ext
            total_int_reward += reward_int
            obs = next_obs
            h = h_new
        
        # Update performance tracking
        avg_performance = performance_alpha * total_ext_reward + (1 - performance_alpha) * avg_performance
        
        # Compute loss for policy update
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        value_loss = []
        
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(value, torch.tensor([[R]]).to(device)))
        
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = torch.stack(value_loss).sum()
        
        # Add perception loss
        if perception_losses:
            mean_perception_loss = torch.stack(perception_losses).mean()
        else:
            mean_perception_loss = torch.tensor(0.0).to(device)
        
        loss = policy_loss + 0.5 * value_loss + 0.1 * mean_perception_loss
        
        # Backward pass with gradient clipping (specification Section 4)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update H-JEPA target encoder with EMA
        if hasattr(agent, 'perception') and hasattr(agent.perception, 'update_target_encoder'):
            ema_tau = config.get('h_jepa', {}).get('ema_tau', 0.996)
            agent.perception.update_target_encoder(tau=ema_tau)
        
        # Logging
        log_interval = config['training'].get('log_interval', 10)
        if (episode + 1) % log_interval == 0:
            # Calculate correlation for Phase 1 success criteria
            if len(uncertainty_history) > 10:
                correlation = np.corrcoef(uncertainty_history[-100:], epsilon_history[-100:])[0, 1]
            else:
                correlation = 0.0
            
            print(f"Episode {episode + 1}/{config['training']['num_episodes']}: "
                  f"ExtReward={total_ext_reward:.2f}, IntReward={total_int_reward:.2f}, "
                  f"Loss={loss.item():.4f}, Epsilon={epsilon_explore:.4f}, "
                  f"AvgPerf={avg_performance:.2f}, Correlation={correlation:.3f}")
                  
        # Save checkpoint periodically
        save_interval = config['training'].get('save_interval', 100)
        if (episode + 1) % save_interval == 0:
            checkpoint = {
                'episode': episode + 1,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon_explore': epsilon_explore,
                'avg_performance': avg_performance,
                'config': config
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final statistics
    if len(uncertainty_history) > 10:
        final_correlation = np.corrcoef(uncertainty_history, epsilon_history)[0, 1]
        print("\n" + "=" * 80)
        print(f"Phase 1 Training Complete!")
        print(f"Final Correlation (φ_unc vs ε_explore): {final_correlation:.3f}")
        print(f"Success Criteria (>0.4): {'✓ PASSED' if final_correlation > 0.4 else '✗ FAILED'}")
        print("=" * 80)
    
    # Cleanup
    if hasattr(agent, 'cleanup'):
        agent.cleanup()
    env.close()


if __name__ == "__main__":
    main()

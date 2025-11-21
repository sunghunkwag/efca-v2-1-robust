import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from pathlib import Path
from efca.agent import EFCAgent
from torchvision import transforms

def main():
    """
    Main training loop for the EFCA-v2.1 agent.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train EFCA-v2.1 Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize the environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Initialize the unified agent
    # Ensure action dim is set correctly from environment
    config['task_policy']['action_dim'] = env.action_space.n

    # Initialize agent
    agent = EFCAgent(config)

    # Initialize optimizers
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config["training"]["learning_rate"],
    )
    
    # Setup checkpoint directory
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint if exists
    start_episode = 0
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        print(f"Resuming from episode {start_episode}")

    # Preprocessing for rendering CartPole as an image
    resize = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    # Device handling for preprocessing (Agent handles its own device via config)
    device = config['training'].get('device', 'cpu')

    print("Agent initialized.")
    print(f"Starting Phase {config.get('phase', 0)}: 'Zombie' Agent training loop on {device}.")
    print(f"Training for {config['training']['num_episodes']} episodes")
    print("-" * 80)
    
    # Training loop
    for episode in range(start_episode, config["training"]["num_episodes"]):
        obs, _ = env.reset()
        h = None # Agent handles initialization if None
        total_reward = 0
        done = False

        log_probs = []
        values = []
        rewards = []
        perception_losses = []

        while not done:
            # Render the environment and preprocess the image
            screen = env.render()
            screen_tensor = resize(screen).unsqueeze(0).to(device)  # Add batch dimension (1, C, H, W)

            # --- Agent's forward pass ---
            # forward returns: dist, value, h_new, meta_delta, perception_loss, probe_output
            result = agent(screen_tensor, h=h)
            dist, value, h, meta_delta, p_loss, probe_output = result

            # NOTE: meta_delta is currently unused in Phase 0, but verified here.
            if meta_delta is not None:
                pass # Logic to use meta_delta would go here in later phases

            action = dist.sample()

            # --- Environment step ---
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            perception_losses.append(p_loss)

        # --- Optimization step ---
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        # Calculate Returns (Monte Carlo)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)

        # Advantage
        # values is (T, 1), returns is (T). Squeeze values to match.
        advantage = returns - values.detach().squeeze()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(values.squeeze(), returns)

        # Mean perception loss over the episode
        mean_perception_loss = torch.stack(perception_losses).mean()

        loss = mean_perception_loss + actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping as per specification Section 4 (line 162)
        # "Gradient Clipping is mandatory"
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update H-JEPA target encoder with EMA
        if hasattr(agent, 'perception') and hasattr(agent.perception, 'update_target_encoder'):
            ema_tau = config.get('h_jepa', {}).get('ema_tau', 0.996)
            agent.perception.update_target_encoder(tau=ema_tau)
        
        # Logging
        log_interval = config['training'].get('log_interval', 10)
        if (episode + 1) % log_interval == 0:
            print(f"Episode {episode + 1}/{config['training']['num_episodes']}: "
                  f"Reward={total_reward:.2f}, Loss={loss.item():.4f}, "
                  f"Perception Loss={mean_perception_loss.item():.4f}")
        
        # Save checkpoint periodically
        save_interval = config['training'].get('save_interval', 100)
        if (episode + 1) % save_interval == 0:
            checkpoint = {
                'episode': episode + 1,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")

    # Final checkpoint save
    final_checkpoint = {
        'episode': config['training']['num_episodes'],
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    final_path = os.path.join(checkpoint_dir, 'final_checkpoint.pt')
    torch.save(final_checkpoint, final_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_path}")
    
    # Cleanup
    if hasattr(agent, 'cleanup'):
        agent.cleanup()
    env.close()


if __name__ == "__main__":
    main()

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from efca.agent import EFCAgent
from torchvision import transforms

def main():
    """
    Main training loop for the EFCA-v2.1 agent.
    """
    # Load configuration
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize the environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Initialize the unified agent
    # Ensure action dim is set correctly from environment
    config['task_policy']['action_dim'] = env.action_space.n

    agent = EFCAgent(config)

    # Initialize optimizers
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # Preprocessing for rendering CartPole as an image
    resize = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    print("Agent initialized.")
    print(f"Starting Phase {config.get('phase', 0)}: 'Zombie' Agent training loop.")

    for episode in range(config["training"]["num_episodes"]):
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
            screen_tensor = resize(screen).unsqueeze(0)  # Add batch dimension (1, C, H, W)

            # --- Agent's forward pass ---
            # forward returns: dist, value, h_new, meta_delta, perception_loss
            dist, value, h, meta_delta, p_loss = agent(screen_tensor, h=h)

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
        returns = torch.tensor(returns)

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
        optimizer.step()

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Loss = {loss.item():.4f}")

    env.close()


if __name__ == "__main__":
    main()

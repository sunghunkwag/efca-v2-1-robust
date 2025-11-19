import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from efca.dynamics.ct_lnn import CTLNN
from efca.perception.h_jepa import HJEPA
from efca.policy.task_policy import TaskPolicy
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

    # Initialize the agent's modules
    perception = HJEPA(embed_dim=config["h_jepa"]["embed_dim"])
    dynamics = CTLNN(
        input_dim=config["ct_lnn"]["input_dim"],
        hidden_dim=config["ct_lnn"]["hidden_dim"],
        output_dim=config["ct_lnn"]["output_dim"],
    )
    policy = TaskPolicy(
        hidden_dim=config["task_policy"]["hidden_dim"],
        action_dim=env.action_space.n,
    )

    # Initialize optimizers
    optimizer = optim.Adam(
        list(perception.parameters())
        + list(dynamics.parameters())
        + list(policy.parameters()),
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

    print("Agent modules initialized.")
    print("Starting Phase 0: 'Zombie' Agent training loop.")

    for episode in range(config["training"]["num_episodes"]):
        obs, _ = env.reset()
        h = dynamics.init_state(batch_size=1)
        total_reward = 0
        done = False

        log_probs = []
        values = []
        rewards = []

        while not done:
            # Render the environment and preprocess the image
            screen = env.render()
            screen_tensor = resize(screen).unsqueeze(0)  # Add batch dimension

            # --- Agent's forward pass ---
            # 1. Perception
            perception_loss, online_features = perception(screen_tensor)
            perception_output = online_features.mean(dim=[2, 3]) # Global average pooling


            # 2. Dynamics
            h = dynamics.forward(h, perception_output)

            # 3. Policy
            dist, value = policy.forward(h)
            action = dist.sample()

            # --- Environment step ---
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)


        # --- Optimization step ---
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        advantage = returns - values.detach().squeeze()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(values.squeeze(), returns)

        loss = perception_loss + actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
import yaml
from efca.dynamics.ct_lnn import CTLNN
from efca.metacontrol.meta_controller import MetaController
from efca.perception.h_jepa import HJEPA
from efca.policy.task_policy import TaskPolicy
from efca.probe.probe_network import ProbeNetwork
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
    probe = ProbeNetwork(
        hidden_dim=config["ct_lnn"]["hidden_dim"],
        probe_dim=config["probe_network"]["probe_dim"],
    )
    meta_controller = MetaController(
        probe_dim=config["probe_network"]["probe_dim"],
    )

    # Initialize optimizers
    optimizer = optim.Adam(
        list(perception.parameters())
        + list(dynamics.parameters())
        + list(policy.parameters())
        + list(probe.parameters()),
        lr=config["training"]["learning_rate"],
    )
    meta_optimizer = optim.Adam(
        meta_controller.parameters(), lr=config["training"]["learning_rate"]
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
    print("Starting Phase 1: 'Curious' Agent training loop.")

    epsilon_explore = config["training"]["epsilon_start"]
    avg_performance = 0.0

    for episode in range(config["training"]["num_episodes"]):
        obs, _ = env.reset()
        h = dynamics.init_state(batch_size=1)
        total_reward = 0
        done = False

        log_probs, values, rewards, intrinsic_rewards, meta_deltas = [], [], [], [], []

        while not done:
            screen = env.render()
            screen_tensor = resize(screen).unsqueeze(0)

            # --- Agent's forward pass ---
            perception_loss, online_features = perception(screen_tensor)
            perception_output = online_features.mean(dim=[2, 3])

            h = dynamics.forward(h, perception_output)

            # --- Metacognition ---
            probe_output = probe(h)
            meta_delta = meta_controller(probe_output)
            epsilon_explore = max(
                0.01, min(1.0, epsilon_explore * (1 + meta_delta.item()))
            )

            # --- Action Selection ---
            if torch.rand(1).item() < epsilon_explore:
                action = torch.tensor([env.action_space.sample()])
                dist, value = policy(h)
            else:
                dist, value = policy(h)
                action = dist.sample()

            # --- Environment Step ---
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

            # --- Reward and Logging ---
            intrinsic_reward = (probe_output**2).mean()
            rewards.append(reward)
            intrinsic_rewards.append(intrinsic_reward)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            meta_deltas.append(meta_delta)

        # --- Optimization Step ---
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        meta_deltas = torch.cat(meta_deltas)

        # Homeostatic Gate
        avg_performance = 0.9 * avg_performance + 0.1 * total_reward

        # --- Calculate Returns and Advantages ---
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        intrinsic_returns = []
        R_intrinsic = 0
        for r_i in reversed(intrinsic_rewards):
            R_intrinsic = r_i + 0.99 * R_intrinsic
            intrinsic_returns.insert(0, R_intrinsic)
        intrinsic_returns = torch.stack(intrinsic_returns)

        advantage = returns - values.detach().squeeze()
        intrinsic_advantage = intrinsic_returns - values.detach().squeeze()

        # --- Calculate Losses ---
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(values.squeeze(), returns)

        loss = perception_loss + actor_loss + critic_loss

        # We need to retain the graph for the meta-controller's backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if avg_performance >= config["training"]["survival_threshold"]:
            meta_loss = -(meta_deltas * intrinsic_advantage).mean()
            meta_loss.backward()
            meta_optimizer.step()
            meta_optimizer.zero_grad()

        perception.update_target_encoder()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {epsilon_explore:.4f}")

    env.close()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any
import sys
import os

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym

# Import all the implemented modules
from efca.perception.h_jepa import HJEPA
from efca.bottleneck.s_gwt import sGWT
from efca.dynamics.ct_lnn import CTLNN
from efca.policy.actor_critic import ActorCriticPolicy
from efca.metacontrol.meta_controller import MetaControllerPolicy
from efca.probe.probe_network import ProbeNetwork
from efca.utils.replay_buffer import ReplayBuffer

@dataclass
class TrainingConfig:
    # --- Model Dimensions ---
    state_dim: int = 4 # Example for CartPole
    action_dim: int = 2 # Example for CartPole
    lnn_hidden_dim: int = 256
    probe_dim: int = 64
    meta_action_dim: int = 3 # lr, epsilon, lambda

    # --- Training Parameters ---
    warmup_episodes: int = 100
    total_episodes: int = 1000
    learning_rate: float = 3e-4
    meta_learning_rate: float = 1e-5
    gamma: float = 0.99

    # --- EFCA-Specific Parameters ---
    meta_interval: int = 10 # Update meta-controller every 10 steps
    survival_threshold: float = 50.0 # Min avg performance to enable intrinsic rewards
    clamp_delta: float = 0.05 # Max 5% change from meta-controller

    # --- Hardware ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class EFCANetwork(nn.Module):
    """ A wrapper class to hold all EFCA modules. """
    def __init__(self, config: TrainingConfig):
        super().__init__()
        # Note: Dimensions are placeholders and need to be wired correctly.
        # This is a simplified structural representation.
        self.perception = HJEPA()
        self.bottleneck = sGWT(input_dim=768, slot_dim=768)
        self.dynamics = CTLNN(input_dim=4*768, hidden_size=config.lnn_hidden_dim)
        self.policy = ActorCriticPolicy(state_dim=config.lnn_hidden_dim, action_dim=config.action_dim)
        self.probe = ProbeNetwork(lnn_hidden_dim=config.lnn_hidden_dim, perception_error_dim=1, probe_output_dim=config.probe_dim)
        self.meta_controller = MetaControllerPolicy(probe_state_dim=config.probe_dim, meta_action_dim=config.meta_action_dim)


class SimpleAgent(nn.Module):
    """ A simplified agent for end-to-end testing with low-dimensional environments. """
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.policy = ActorCriticPolicy(state_dim=config.state_dim, action_dim=config.action_dim)
        self.device = config.device

    def select_action(self, state):
        """ Selects an action based on the current state. """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, _ = self.policy(state_tensor)
        action = dist.sample()
        return action.item()

def run_episode(agent, env, buffer, config, meta_active=False):
    """
    Runs a single episode, interacts with the environment, and stores transitions.
    """
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not done and not truncated:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done or truncated)
        state = next_state
        total_reward += reward

    return total_reward

def update_policy(agent, buffer, optimizer, config):
    """
    Placeholder for the policy update step (e.g., PPO or A2C).
    """
    if len(buffer) < 32: # Minimum batch size
        return

    # states, actions, rewards, next_states, dones = buffer.sample(32)
    # Perform a gradient update step here.
    # For now, we just print a message.
    print("Updating task policy (placeholder)...")
    pass

def robust_training_loop(agent, env, config):
    """
    Main training loop for the simplified agent.
    """
    buffer = ReplayBuffer(capacity=10000)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    # --- Phase 0: Warm-up (Collect Experience) ---
    print("--- Phase 0: Warm-up ---")
    for episode in range(config.warmup_episodes):
        reward = run_episode(agent, env, buffer, config, meta_active=False)
        print(f"Warm-up Episode {episode + 1}/{config.warmup_episodes}, Reward: {reward}")

        # In a real scenario, you'd train the perception model here.
        # For the simple agent, we can directly update the policy.
        update_policy(agent, buffer, optimizer, config)

    print("\n--- Warm-up Complete ---")
    # A real implementation would transition to Phase 1 with the meta-controller here.
    # For this script, we'll stop after the warm-up to demonstrate functionality.

def main():
    parser = argparse.ArgumentParser(description="Train EFCA-v2.1 Agent")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig.from_yaml(args.config)

    # Initialize the environment
    env = gym.make("CartPole-v1")

    # Update config with env-specific dimensions
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n

    # Initialize the simplified agent for end-to-end testing
    agent = SimpleAgent(config).to(config.device)

    print("Starting EFCA-v2.1 Training Loop (Simplified Agent)")
    print(f"Using device: {config.device}")

    robust_training_loop(agent, env, config)

    env.close()

if __name__ == "__main__":
    main()

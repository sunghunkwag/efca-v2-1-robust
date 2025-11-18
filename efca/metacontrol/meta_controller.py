import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class MetaControllerPolicy(nn.Module):
    """
    The Meta-Controller Policy Network.

    This network implements the self-regulation mechanism of EFCA-v2.1. It takes a
    metacognitive state representation `phi` from the Probe Network and outputs a
    continuous action representing a *relative change* to the agent's internal parameters.

    Key features:
    - Actor-Critic architecture suitable for PPO.
    - Outputs parameters for a continuous action distribution (Gaussian).
    - Uses tanh to bound the output action, which is then clamped in the training loop
      to implement "Clamped Delta Control".
    """
    def __init__(self, probe_state_dim, meta_action_dim, hidden_dim=128):
        super().__init__()
        self.probe_state_dim = probe_state_dim
        self.meta_action_dim = meta_action_dim
        self.hidden_dim = hidden_dim

        # --- Shared Body ---
        self.body = nn.Sequential(
            nn.Linear(probe_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # --- Actor Head ---
        # Outputs the mean of the Gaussian action distribution.
        self.actor_mean = nn.Linear(hidden_dim, meta_action_dim)
        # Outputs the log standard deviation of the action distribution.
        self.actor_log_std = nn.Linear(hidden_dim, meta_action_dim)

        # --- Critic Head ---
        # Outputs the value of the metacognitive state `phi`.
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, probe_state):
        """
        Forward pass of the meta-controller policy.

        Args:
            probe_state (torch.Tensor): The metacognitive state, shape (B, probe_state_dim).

        Returns:
            torch.distributions.Normal: The action distribution.
            torch.Tensor: The estimated value of the metacognitive state, shape (B, 1).
        """
        # 1. Process probe state through the shared body
        features = self.body(probe_state)

        # 2. Get action distribution parameters from the actor head
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)

        # Clamp the log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # Create the Normal distribution
        action_dist = Normal(mean, std)

        # 3. Get state value from the critic head
        meta_state_value = self.critic_head(features)

        return action_dist, meta_state_value

    def sample_action(self, probe_state, clamp_delta=0.05):
        """
        Samples an action, applies tanh, and returns the clamped delta.
        This follows the "Clamped Delta Control" mechanism.

        Args:
            probe_state (torch.Tensor): The metacognitive state.
            clamp_delta (float): The maximum relative change (e.g., 0.05 for 5%).

        Returns:
            torch.Tensor: The final clamped delta for updating agent parameters.
            torch.Tensor: The log probability of the raw action (before tanh).
        """
        action_dist, _ = self.forward(probe_state)

        # Sample an action. Use rsample for reparameterization trick to allow gradients.
        raw_action = action_dist.rsample()
        log_prob = action_dist.log_prob(raw_action).sum(axis=-1)

        # Apply tanh to bound the action between [-1, 1]. This is a_t^meta.
        meta_action_tanh = torch.tanh(raw_action)

        # The training loop will then use this to update parameters:
        # Param_t+1 = Param_t * (1 + clamp(meta_action_tanh, -delta, +delta))
        # For simplicity, we can return the clamped value directly.
        clamped_output = torch.clamp(meta_action_tanh, -clamp_delta, clamp_delta)

        return clamped_output, log_prob


if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    PROBE_DIM = 64   # Example dimension of the metacognitive state phi
    META_ACTION_DIM = 3 # Example: controls [learning_rate, exploration_epsilon, jepa_lambda]
    BATCH_SIZE = 4

    # Create a dummy probe state tensor
    dummy_probe_state = torch.randn(BATCH_SIZE, PROBE_DIM).to(device)

    # Initialize the meta-controller
    meta_controller = MetaControllerPolicy(PROBE_DIM, META_ACTION_DIM).to(device)

    # Get action distribution and meta-state value
    dist, value = meta_controller(dummy_probe_state)

    print(f"Input probe state shape: {dummy_probe_state.shape}")
    print(f"Meta-state value shape: {value.shape}") # Should be (4, 1)
    print(f"Action distribution mean: {dist.mean.shape}") # Should be (4, 3)

    # Sample a clamped delta
    delta, log_p = meta_controller.sample_action(dummy_probe_state, clamp_delta=0.05)

    print(f"\n--- Clamped Delta Control ---")
    print(f"Sampled delta shape: {delta.shape}") # Should be (4, 3)
    print(f"Sampled delta values:\n{delta}")
    print(f"Log probability shape: {log_p.shape}")

    # Verify that the output is clamped correctly
    assert torch.all(delta >= -0.05) and torch.all(delta <= 0.05)
    print("\nAssertion passed: Delta values are correctly clamped between -0.05 and +0.05.")

    # Verify that gradients flow through
    loss = value.sum() + delta.sum() # Example loss
    loss.backward()

    has_grads = all(p.grad is not None for p in meta_controller.parameters())
    print(f"All parameters have gradients: {has_grads}") # Should be True

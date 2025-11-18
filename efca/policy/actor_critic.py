import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticPolicy(nn.Module):
    """
    A standard Actor-Critic policy network.

    This module takes a state representation (e.g., from the CT-LNN's hidden state)
    and outputs an action distribution (Actor) and a state value estimate (Critic).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # --- Shared Body ---
        # A common network to process the input state before splitting into actor and critic heads.
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # --- Actor Head ---
        # Outputs logits for the action probability distribution.
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # --- Critic Head ---
        # Outputs a single scalar value representing the value of the state.
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass of the policy network.

        Args:
            state (torch.Tensor): The input state, shape (B, state_dim).

        Returns:
            torch.distributions.Distribution: The action distribution.
            torch.Tensor: The estimated state value, shape (B, 1).
        """
        # 1. Process state through the shared body
        features = self.body(state)

        # 2. Get action logits from the actor head
        action_logits = self.actor_head(features)

        # Create a categorical distribution for sampling actions
        # This is suitable for discrete action spaces.
        action_dist = Categorical(logits=action_logits)

        # 3. Get state value from the critic head
        state_value = self.critic_head(features)

        return action_dist, state_value

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    STATE_DIM = 512  # Example: hidden state size from CT-LNN
    ACTION_DIM = 10  # Example: number of discrete actions in the environment
    BATCH_SIZE = 4

    # Create a dummy state tensor
    dummy_state = torch.randn(BATCH_SIZE, STATE_DIM).to(device)

    # Initialize the policy network
    policy = ActorCriticPolicy(STATE_DIM, ACTION_DIM).to(device)

    # Get action distribution and state value
    dist, value = policy(dummy_state)

    print(f"Input state shape: {dummy_state.shape}")
    print(f"State value output shape: {value.shape}") # Should be (4, 1)

    # Sample an action from the distribution
    action = dist.sample()
    print(f"Sampled action shape: {action.shape}") # Should be (4,)
    print(f"Sampled action: {action}")

    # Calculate log probability of the sampled action
    log_prob = dist.log_prob(action)
    print(f"Log probability shape: {log_prob.shape}") # Should be (4,)

    # Verify that gradients flow through
    loss = value.sum() + log_prob.sum() # Example loss
    loss.backward()

    has_grads = all(p.grad is not None for p in policy.parameters())
    print(f"\nAll parameters have gradients: {has_grads}") # Should be True

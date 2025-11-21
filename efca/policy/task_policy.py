from typing import Tuple, Literal

import torch
import torch.distributions as distributions
import torch.nn as nn


class TaskPolicy(nn.Module):
    """
    Standard Actor-Critic Task Policy supporting both Discrete and Continuous action spaces.

    This module takes a hidden state from the dynamics model and outputs:
    1.  An action distribution (Categorical for discrete, Normal for continuous).
    2.  A state value estimate.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_space_type: Literal['discrete', 'continuous'] = 'discrete'
    ) -> None:
        """
        Initializes the TaskPolicy module.

        Args:
            hidden_dim (int): The dimensionality of the input hidden state.
            action_dim (int): The number of actions (discrete) or action dimensions (continuous).
            action_space_type (str): 'discrete' or 'continuous'. Defaults to 'discrete'.
        """
        super().__init__()
        self.action_space_type = action_space_type

        # Common feature extractor (optional, but good for sharing weights)
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )

        # Actor network
        if self.action_space_type == 'discrete':
            self.actor_head = nn.Sequential(
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )
        elif self.action_space_type == 'continuous':
            # For continuous actions, we output mean and log_std
            self.actor_mean = nn.Linear(64, action_dim)
            # Learnable log standard deviation (state-independent for simplicity, or state-dependent)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            raise ValueError(f"Unsupported action_space_type: {action_space_type}")

        # Critic network
        self.critic_head = nn.Linear(64, 1)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[distributions.Distribution, torch.Tensor]:
        """
        Forward pass through the actor and critic networks.

        Args:
            h (torch.Tensor): The hidden state from the dynamics model (B, hidden_dim).

        Returns:
            dist (distributions.Distribution): Action distribution.
            value (torch.Tensor): State value estimate (B, 1).
        """
        features = self.feature_extractor(h)
        value = self.critic_head(features)

        if self.action_space_type == 'discrete':
            action_probs = self.actor_head(features)
            dist = distributions.Categorical(action_probs)
        else:
            mean = torch.tanh(self.actor_mean(features))  # Bound mean to [-1, 1]
            # Expand log_std to batch size
            log_std = self.actor_log_std.expand_as(mean)
            std = torch.exp(log_std)
            dist = distributions.Normal(mean, std)

        return dist, value

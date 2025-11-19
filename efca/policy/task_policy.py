from typing import Tuple

import torch
import torch.distributions as distributions
import torch.nn as nn


class TaskPolicy(nn.Module):
    """
    Standard Actor-Critic Task Policy.

    This module takes a hidden state from the dynamics model and outputs two things:
    1.  An action distribution (the "actor"), which can be sampled to select an action.
    2.  A state value (the "critic"), which estimates the expected return from the current state.
    """

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        """
        Initializes the TaskPolicy module.

        Args:
            hidden_dim (int): The dimensionality of the input hidden state from the dynamics model.
            action_dim (int): The number of possible actions in the environment.
        """
        super().__init__()

        # Actor network: Maps state to an action probability distribution
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),  # Outputs probabilities for each action
        )

        # Critic network: Maps state to a single value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[distributions.Distribution, torch.Tensor]:
        """
        Forward pass through the actor and critic networks.

        Args:
            h (torch.Tensor): The hidden state from the dynamics model.

        Returns:
            A tuple containing:
            - A Categorical distribution over actions.
            - The estimated value of the state.
        """
        action_probs = self.actor(h)
        dist = distributions.Categorical(action_probs)
        value = self.critic(h)
        return dist, value

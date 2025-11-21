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

    def __init__(self, hidden_dim: int, action_dim: int, continuous_action: bool = False) -> None:
        """
        Initializes the TaskPolicy module.

        Args:
            hidden_dim (int): The dimensionality of the input hidden state from the dynamics model.
            action_dim (int): The number of possible actions in the environment.
            continuous_action (bool): Whether the action space is continuous.
        """
        super().__init__()
        self.continuous_action = continuous_action

        if continuous_action:
            # Actor network for continuous actions: Maps state to mean and log_std
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh(),  # Assume actions are normalized to [-1, 1] usually
            )
            # Learnable log standard deviation
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
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
            - A distribution over actions (Categorical or Normal).
            - The estimated value of the state.
        """
        value = self.critic(h)

        if self.continuous_action:
            mean = self.actor_mean(h)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = distributions.Normal(mean, std)
        else:
            action_probs = self.actor(h)
            dist = distributions.Categorical(action_probs)

        return dist, value

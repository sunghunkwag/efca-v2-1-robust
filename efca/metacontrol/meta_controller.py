import torch
import torch.nn as nn
import torch.distributions as distributions


class MetaController(nn.Module):
    """
    The Meta-Controller is a self-regulatory module that adjusts the agent's
    hyperparameters based on its internal state.

    For Phase 1, its sole responsibility is to control the exploration rate
    (`epsilon_explore`) based on the output of the Probe Network. It uses a
    "Clamped Delta Control" mechanism to output a *relative change* to the
    parameter, preventing drastic, destabilizing updates.
    """

    def __init__(self, probe_dim: int, max_delta: float = 0.05) -> None:
        """
        Initializes the MetaController.

        Args:
            probe_dim (int): The dimensionality of the Probe Network's output.
            max_delta (float): The maximum percentage change allowed per step.
        """
        super().__init__()
        self.max_delta = max_delta

        # The policy network for the meta-controller
        self.meta_policy = nn.Sequential(
            nn.Linear(probe_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output is between -1 and 1
        )

    def forward(self, probe_output: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass to get the clamped delta for epsilon.

        Args:
            probe_output (torch.Tensor): The output from the Probe Network.

        Returns:
            torch.Tensor: The clamped delta value for `epsilon_explore`.
        """
        # The meta-policy outputs a value in [-1, 1]
        meta_action = self.meta_policy(probe_output)
        # The delta is clamped to the maximum allowed change
        delta = torch.clamp(meta_action, -self.max_delta, self.max_delta)
        return delta

from typing import Tuple

import torch
import torch.nn as nn


class ProbeNetwork(nn.Module):
    """
    The Probe Network is a metacognitive module that monitors the agent's internal state.

    It takes the hidden state of the dynamics model (CT-LNN) as input and outputs a
    representation of that state, which can be used by the Meta-Controller. A crucial
    feature is that gradients are not backpropagated from the probe into the dynamics
    model, ensuring that the act of observation does not interfere with the primary task.
    """

    def __init__(self, hidden_dim: int, probe_dim: int) -> None:
        """
        Initializes the ProbeNetwork.

        Args:
            hidden_dim (int): The dimensionality of the CT-LNN's hidden state.
            probe_dim (int): The dimensionality of the probe's output representation.
        """
        super().__init__()
        self.probe_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, probe_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the probe network.

        Args:
            h (torch.Tensor): The hidden state from the CT-LNN dynamics model.

        Returns:
            torch.Tensor: A tensor representing the probed internal state.
        """
        # The .detach() call is critical: it prevents gradients from flowing back
        # into the dynamics model, isolating the observer from the observed.
        return self.probe_net(h.detach())

import torch
import torch.nn as nn


class CTLNN(nn.Module):
    """
    Hybrid Continuous-Time Liquid Neural Network (CT-LNN) for dynamics modeling.

    This module simulates the dynamics of a system over time. For Phase 0-2, it
    uses a discrete approximation (Euler integration) for stability and ease of
    training. The full ODE-based mode is reserved for Phase 3.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dt: float = 0.01
    ) -> None:
        """
        Initializes the CT-LNN module.

        Args:
            input_dim (int): The dimensionality of the input from the perception module.
            hidden_dim (int): The dimensionality of the hidden state.
            output_dim (int): The dimensionality of the output (e.g., to the policy).
            dt (float): The time step for the discrete Euler integration.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt

        # Network layers
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

    def init_state(self, batch_size: int) -> torch.Tensor:
        """
        Initializes the hidden state to a tensor of zeros.

        Args:
            batch_size (int): The number of samples in the batch.

        Returns:
            torch.Tensor: The initial hidden state.
        """
        return torch.zeros(batch_size, self.hidden_dim)

    def forward_discrete(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a single discrete-time step of the CT-LNN using Euler integration.

        This approximates the continuous dynamics `dh/dt = f(h, x)`.

        Args:
            h (torch.Tensor): The current hidden state.
            x (torch.Tensor): The current input.

        Returns:
            torch.Tensor: The next hidden state.
        """
        # Calculate the derivative of the hidden state
        dh_dt = torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h))
        # Apply Euler integration to get the next hidden state
        h_next = h + self.dt * dh_dt
        return h_next

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass, which defaults to the discrete approximation for Phase 0-2.

        Args:
            h (torch.Tensor): The current hidden state.
            x (torch.Tensor): The current input.

        Returns:
            torch.Tensor: The next hidden state.
        """
        return self.forward_discrete(h, x)

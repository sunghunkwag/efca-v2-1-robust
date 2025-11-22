import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. ODE mode will be disabled.")


class CTLNN(nn.Module):
    """
    Hybrid Continuous-Time Liquid Neural Network (CT-LNN) for dynamics modeling.

    This module simulates the dynamics of a system over time. For Phase 0-2, it
    uses a discrete approximation (Euler integration) for stability and ease of
    training. The full ODE-based mode is reserved for Phase 3.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the CT-LNN module.

        Implements Hybrid CT-LNN from specification:
        - Discrete Mode (Phase 0-2): Standard BPTT with discrete approximation
        - ODE Mode (Phase 3): Full ODE solver via torchdiffeq

        Args:
            config (dict): Configuration dictionary containing parameters like
                           input_dim, hidden_dim, output_dim, dt, tau, and mode.
        """
        super().__init__()

        self.input_dim: int = config.get('input_dim', 768)
        self.hidden_dim: int = config.get('hidden_dim', 256)
        self.output_dim: int = config.get('output_dim', 256)
        self.dt: float = config.get('dt', 0.01)  # Δt in specification
        self.tau: float = config.get('tau', 0.1)  # Time constant τ

        # Mode: 'discrete' for Phase 0-2, 'ode' for Phase 3
        self.mode: str = config.get('mode', 'discrete')

        if self.mode == 'ode' and not TORCHDIFFEQ_AVAILABLE:
            print("Warning: ODE mode requested but torchdiffeq not available. Falling back to discrete mode.")
            self.mode = 'discrete'

        # Network layers
        self.input_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_to_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden_to_output = nn.Linear(self.hidden_dim, self.output_dim)

    def init_state(self, batch_size: int) -> torch.Tensor:
        """
        Initializes the hidden state to a tensor of zeros.

        Args:
            batch_size (int): The number of samples in the batch.

        Returns:
            torch.Tensor: The initial hidden state (B, hidden_dim).
        """
        return torch.zeros(batch_size, self.hidden_dim)

    def _ode_func(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        The ODE function dh/dt = f(h, u) for the CT-LNN.

        Equation: dh/dt = (-h + tanh(W_in * u + W_rec * h)) / tau

        Note: 'u' (self._current_input) is assumed constant over the integration step.

        Args:
            t (torch.Tensor): Current time (scalar).
            h (torch.Tensor): Current hidden state (B, hidden_dim).

        Returns:
            torch.Tensor: Derivative of hidden state (B, hidden_dim).
        """
        # dh/dt = (-h + f(h, u)) / tau
        # self._current_input must be set before calling this via odeint
        f_hu = torch.tanh(self.input_to_hidden(self._current_input) + self.hidden_to_hidden(h))
        dh_dt = (-h + f_hu) / self.tau
        return dh_dt

    def forward_discrete(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a single discrete-time step of the CT-LNN using Euler integration.

        This approximates the continuous dynamics `dh/dt = f(h, x)`.

        Args:
            h (torch.Tensor): The current hidden state.
            x (torch.Tensor): The current input.

        Returns:
            torch.Tensor: The next hidden state (B, hidden_dim).
        """
        # Calculate the derivative of the hidden state: dh/dt = (-h + f(h, x)) / tau
        f_hx = torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h))
        dh_dt = (-h + f_hx) / self.tau
        # Apply Euler integration to get the next hidden state
        h_next = h + self.dt * dh_dt

        # Note: We return the hidden state directly for recurrence.
        # The output projection (hidden_to_output) is available but not applied here
        # to preserve state dimensions for the next step.
        return h_next

    def forward_ode(self, h: torch.Tensor, x: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ODE-based integration using torchdiffeq (Phase 3).

        Args:
            h: Current hidden state (B, hidden_dim)
            x: Input from perception (B, input_dim)
            t_span: Time span for integration

        Returns:
            torch.Tensor: Next hidden state (B, hidden_dim)
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise RuntimeError("ODE mode requires torchdiffeq. Install: pip install torchdiffeq")

        if t_span is None:
            t_span = torch.tensor([0.0, self.dt], device=h.device)

        # Store input
        self._current_input = x

        # ODE function
        def ode_func(t, state):
            f_hx = torch.tanh(self.input_to_hidden(self._current_input) + self.hidden_to_hidden(state))
            dh_dt = (-state + f_hx) / self.tau
            return dh_dt

        # Integrate
        h_trajectory = odeint(ode_func, h, t_span, method='dopri5')
        h_next = h_trajectory[-1]

        return h_next

    def forward(self, h: Optional[torch.Tensor], x: torch.Tensor, t_span: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with automatic mode selection.

        Args:
            h: Current hidden state or None
            x: Input from perception (B, input_dim)
            t_span: Time span for ODE mode

        Returns:
            torch.Tensor: Next hidden state (B, output_dim)
        """
        batch_size = x.shape[0]

        if h is None:
            h = self.init_state(batch_size).to(x.device)

        # Select mode
        if self.mode == 'discrete':
            return self.forward_discrete(h, x)
        elif self.mode == 'ode':
            return self.forward_ode(h, x, t_span)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_mode(self, mode: str) -> None:
        """Switch between discrete and ODE modes."""
        if mode not in ['discrete', 'ode']:
            raise ValueError(f"Invalid mode: {mode}")
        if mode == 'ode' and not TORCHDIFFEQ_AVAILABLE:
            raise RuntimeError("ODE mode requires torchdiffeq")
        self.mode = mode
        print(f"CT-LNN mode: {mode}")

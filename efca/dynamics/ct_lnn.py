import torch
import torch.nn as nn

class LTCCell(nn.Module):
    """
    Liquid Time Constant (LTC) Cell using a discrete update rule for stability.

    This cell implements the discrete approximation formula from EFCA-v2.1:
    h_t = h_{t-1} + (Δt / τ) * f_theta(h_{t-1}, u_t)

    The time constant τ is learned and is dependent on the current input and hidden state,
    making the cell's dynamics input-dependent.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # A single network to process the combined input and hidden state
        # This will be used to derive the gate, time constant, and state update
        self.input_mapper = nn.Linear(input_size + hidden_size, hidden_size * 2)

        # The core dynamics function f_theta
        self.f_theta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_size, hidden_size)
        )

        # Network to compute the time constant tau. Softplus ensures positivity.
        self.tau_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus() # Ensure tau is always positive
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, u_t, h_tm1, dt=0.01):
        """
        Performs a single discrete step of the LTC-Cell.

        Args:
            u_t (torch.Tensor): The current input, shape (B, input_size).
            h_tm1 (torch.Tensor): The previous hidden state, shape (B, hidden_size).
            dt (float): The time step, Δt.

        Returns:
            torch.Tensor: The next hidden state, h_t, shape (B, hidden_size).
        """
        combined = torch.cat([u_t, h_tm1], dim=-1)

        # Map combined input to an intermediate representation
        mapped_input = self.input_mapper(combined)

        # Split the mapped input into a gate and a core signal
        g, core_signal = torch.chunk(mapped_input, 2, dim=-1)
        g = torch.sigmoid(g) # Gating mechanism

        # 1. Compute time constant τ (input-dependent)
        # We add a small epsilon for numerical stability to avoid division by zero.
        tau = self.tau_net(core_signal) + 1e-6

        # 2. Compute f_theta(h_{t-1}, u_t)
        # Here, we use the gated core signal as the input to f_theta
        f_output = self.f_theta(g * torch.tanh(core_signal))

        # 3. Apply the discrete update rule
        h_t = h_tm1 + (dt / tau) * f_output

        # Apply layer normalization for stability
        h_t = self.norm(h_t)

        return h_t

class CTLNN(nn.Module):
    """
    Continuous-Time Liquid Neural Network (CT-LNN) wrapper.

    This module unrolls an LTCCell over a sequence of inputs. For now, it only
    supports 'discrete' mode, as specified for Phases 0-2 of the EFCA-v2.1 roadmap.
    """
    def __init__(self, input_size, hidden_size, mode='discrete'):
        super().__init__()
        if mode != 'discrete':
            raise NotImplementedError("Only 'discrete' mode is implemented for Phase 0-2.")

        self.hidden_size = hidden_size
        self.cell = LTCCell(input_size, hidden_size)

    def forward(self, input_sequence, h_0=None):
        """
        Processes a sequence of inputs.

        Args:
            input_sequence (torch.Tensor): Input of shape (B, T, D_in).
            h_0 (torch.Tensor, optional): Initial hidden state of shape (B, D_hidden).
                                          Defaults to zeros if None.

        Returns:
            torch.Tensor: A tensor of all hidden states, shape (B, T, D_hidden).
        """
        B, T, _ = input_sequence.shape

        if h_0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=input_sequence.device)
        else:
            h_t = h_0

        outputs = []
        for t in range(T):
            u_t = input_sequence[:, t, :]
            h_t = self.cell(u_t, h_t)
            outputs.append(h_t)

        # Stack the outputs along the time dimension
        return torch.stack(outputs, dim=1)

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Let's assume the input is a sequence of slot states from s-GWT
    # For simplicity, we'll flatten the slots into a single vector per time step
    # Input: (Batch, Seq_Len, Num_Slots * Slot_Dim)
    dummy_input_seq = torch.randn(4, 10, 4 * 768).to(device)

    # Initialize the dynamics module
    dynamics_model = CTLNN(input_size=4 * 768, hidden_size=512, mode='discrete').to(device)

    # Get the sequence of hidden states
    hidden_states = dynamics_model(dummy_input_seq)

    print(f"Input sequence shape: {dummy_input_seq.shape}")
    print(f"Output hidden states shape: {hidden_states.shape}") # Should be (4, 10, 512)

    # Verify that gradients flow through
    hidden_states.sum().backward()

    has_grads = all(p.grad is not None for p in dynamics_model.parameters())
    print(f"All parameters have gradients: {has_grads}") # Should be True

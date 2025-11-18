import torch
import torch.nn as nn

class ProbeNetwork(nn.Module):
    """
    The Probe Network for Metacognition.

    This module observes the agent's internal states to create a metacognitive
    representation, `phi`. As per the EFCA-v2.1 specification, it operates with a
    "Freezed Encoder" to isolate the observer from the observed.
    """
    def __init__(self, lnn_hidden_dim, perception_error_dim, probe_output_dim, hidden_dim=128):
        super().__init__()
        self.lnn_hidden_dim = lnn_hidden_dim
        self.perception_error_dim = perception_error_dim
        self.probe_output_dim = probe_output_dim

        # The input to the probe is the concatenation of the LNN hidden state and perception errors.
        input_dim = lnn_hidden_dim + perception_error_dim

        # A simple MLP to process the concatenated internal states.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, probe_output_dim)
        )

    def forward(self, lnn_hidden_state, perception_errors):
        """
        Forward pass for the Probe Network.

        Args:
            lnn_hidden_state (torch.Tensor): The hidden state from the CT-LNN, shape (B, lnn_hidden_dim).
            perception_errors (torch.Tensor): A tensor representing errors from the H-JEPA module,
                                              shape (B, perception_error_dim).

        Returns:
            torch.Tensor: The metacognitive state representation `phi`, shape (B, probe_output_dim).
        """
        # --- "Freezed Encoder" Implementation ---
        # Detach the inputs from the current computation graph to prevent gradients
        # from flowing back into the LNN and H-JEPA modules.
        h_t_detached = lnn_hidden_state.detach()
        errors_detached = perception_errors.detach()

        # 1. Concatenate the detached internal states.
        combined_state = torch.cat([h_t_detached, errors_detached], dim=-1)

        # 2. Process through the network to get the metacognitive state `phi`.
        phi = self.network(combined_state)

        return phi

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    LNN_DIM = 512       # Example hidden dim from CT-LNN
    PERCEPTION_ERROR_DIM = 128 # Example dimension of an error vector from H-JEPA
    PROBE_DIM = 64      # Desired output dimension for phi
    BATCH_SIZE = 4

    # --- Create dummy inputs that require gradients ---
    # This simulates the real scenario where these tensors are part of a larger graph.
    dummy_h_t = torch.randn(BATCH_SIZE, LNN_DIM, device=device, requires_grad=True)
    dummy_errors = torch.randn(BATCH_SIZE, PERCEPTION_ERROR_DIM, device=device, requires_grad=True)

    # Initialize the probe network
    probe = ProbeNetwork(
        lnn_hidden_dim=LNN_DIM,
        perception_error_dim=PERCEPTION_ERROR_DIM,
        probe_output_dim=PROBE_DIM
    ).to(device)

    # Get the metacognitive state phi
    phi = probe(dummy_h_t, dummy_errors)

    print(f"Input LNN state shape: {dummy_h_t.shape}")
    print(f"Input perception error shape: {dummy_errors.shape}")
    print(f"Output phi shape: {phi.shape}") # Should be (4, 64)

    # --- Verify the "Freezed Encoder" mechanism ---
    # The gradient should only flow through the probe's own parameters,
    # not back to the original input tensors.
    phi.sum().backward()

    probe_has_grads = all(p.grad is not None for p in probe.parameters())
    print(f"\nProbe network has gradients: {probe_has_grads}") # Should be True

    # The original tensors should NOT have gradients from this operation.
    h_t_has_grad = dummy_h_t.grad is not None
    errors_has_grad = dummy_errors.grad is not None
    print(f"LNN hidden state has gradients: {h_t_has_grad}") # Should be False
    print(f"Perception errors have gradients: {errors_has_grad}") # Should be False

    assert probe_has_grads
    assert not h_t_has_grad
    assert not errors_has_grad
    print("\nAssertion passed: Gradients are correctly isolated within the Probe Network.")

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTopK(nn.Module):
    """
    Differentiable Soft Top-K routing.
    This implementation uses a simple softmax over the routing scores
    to create a "soft" assignment of inputs to slots. It's a simplified
    interpretation of a "Soft-TopK router" that maintains differentiability.
    """
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, scores):
        """
        Args:
            scores (torch.Tensor): A tensor of scores of shape (B, N, K_slots).
                                   B=batch, N=num_inputs, K_slots=num_slots.
        Returns:
            torch.Tensor: A tensor of soft assignments of shape (B, N, K_slots).
        """
        # Using softmax to get a probability distribution over slots for each input token
        # This acts as a soft assignment.
        soft_assignments = F.softmax(scores, dim=-1)

        # To emulate "Top-K", we can zero out the probabilities for non-top-k slots.
        # However, to keep it simple and differentiable, we'll use the soft assignments directly.
        # A more complex implementation might use Gumbel-Softmax or a differentiable sorting network.
        # For now, softmax is a reasonable and stable interpretation of "soft selection".

        # In this simplified version, we don't strictly enforce K, but allow soft routing.
        # The "top_k" parameter is kept for future, more complex implementations.
        return soft_assignments


class sGWT(nn.Module):
    """
    Slotted Global Workspace Theory (s-GWT) Bottleneck.

    This module acts as an information bottleneck, forcing competition among
    input features to be written to a limited number of "slots".

    As per EFCA-v2.1, it uses a Soft-TopK router for stable early-stage training.
    """
    def __init__(self, num_slots=4, input_dim=768, slot_dim=768):
        super().__init__()
        self.num_slots = num_slots
        self.input_dim = input_dim
        self.slot_dim = slot_dim

        # Learnable slot embeddings (keys)
        self.slots_k = nn.Parameter(torch.randn(1, self.num_slots, self.slot_dim))

        # Learnable slot embeddings (values) - initialized to zeros
        self.slots_v = nn.Parameter(torch.zeros(1, self.num_slots, self.slot_dim))

        self.soft_topk_router = SoftTopK(top_k=num_slots) # k isn't strictly used here yet

        # Layer normalization for stability
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

    def forward(self, x):
        """
        Forward pass for the s-GWT module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D), where B is the batch size,
                              N is the number of input tokens/patches, and D is the feature dimension.

        Returns:
            torch.Tensor: The state of the slots after processing the input, shape (B, K, D_slot).
        """
        B, N, D = x.shape

        # 1. Normalize inputs for stable dot-product attention
        x_norm = self.norm_input(x)
        slots_k_norm = self.norm_slots(self.slots_k)

        # 2. Compute routing scores (dot-product similarity between inputs and slot keys)
        # (B, N, D) @ (1, K, D).transpose(1,2) -> (B, N, K)
        scores = torch.matmul(x_norm, slots_k_norm.transpose(-2, -1))

        # 3. Get soft assignments from the router
        assignments = self.soft_topk_router(scores) # Shape: (B, N, K)

        # 4. Compute weighted sum of inputs for each slot
        # assignments.transpose(1,2) @ x -> (B, K, N) @ (B, N, D) -> (B, K, D)
        updates = torch.matmul(assignments.transpose(-2, -1), x)

        # 5. Update slot values (simple additive update)
        # Start with the initialized slot values and add the updates
        updated_slots = self.slots_v.repeat(B, 1, 1) + updates

        return updated_slots


if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Let's assume the input is the flattened feature map from H-JEPA's backbone
    # ConvNext-Tiny on 224x224 -> 7x7 feature map with 768 dims
    dummy_input = torch.randn(4, 49, 768).to(device) # (Batch, Num Patches, Dim)

    # Initialize the bottleneck with K_slot=4 as specified
    bottleneck = sGWT(num_slots=4, input_dim=768, slot_dim=768).to(device)

    # Get the output slot states
    slot_outputs = bottleneck(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output slots shape: {slot_outputs.shape}") # Should be (4, 4, 768)

    # Verify that gradients flow through
    slot_outputs.sum().backward()

    has_grads = all(p.grad is not None for p in bottleneck.parameters())
    print(f"All parameters have gradients: {has_grads}") # Should be True

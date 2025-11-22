import torch
import torch.nn as nn
import torch.nn.functional as F

class SGWT(nn.Module):
    """
    Slot-based Global Workspace Theory (s-GWT) Bottleneck.
    Implements Slot Attention to route information from input (Perception) to slots (Bottleneck).
    """
    def __init__(self, config):
        super().__init__()
        self.num_slots = config.get('num_slots', 4)
        self.dim = config.get('dim', 128)
        self.iters = config.get('iters', 3) # Number of attention iterations
        self.hidden_dim = config.get('hidden_dim', self.dim * 2)
        self.eps = 1e-8
        self.topk = config.get('topk', 2)

        # Parameters for Slot Attention
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.dim))
        
        # Linear projections
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        # GRU for iterative updates
        self.gru = nn.GRUCell(self.dim, self.dim)

        # MLP block for slots
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim)
        )

        self.norm_input = nn.LayerNorm(self.dim)
        self.norm_slots = nn.LayerNorm(self.dim)
        self.norm_pre_ff = nn.LayerNorm(self.dim)



    def soft_topk(self, x, k, dim=-1, temperature=1.0):
        """
        Differentiable Soft-TopK approximation.
        
        Args:
            x: Input tensor
            k: Number of top elements to keep
            dim: Dimension to apply TopK
            temperature: Softmax temperature
            
        Returns:
            Soft-TopK probabilities
        """
        # Get top-k indices and values
        topk_val, _ = torch.topk(x, k, dim=dim)
        
        # Create a mask for non-top-k values
        # We use the k-th largest value as the threshold
        # We use the k-th largest value as the threshold
        # topk_val is sorted, so the k-th largest is at index k-1
        threshold = topk_val.select(dim, k-1).unsqueeze(dim)
        
        # Mask values below threshold with large negative number
        mask = x < threshold
        masked_x = x.clone()
        masked_x[mask] = -float('inf')
        
        # Apply Softmax
        return F.softmax(masked_x / temperature, dim=dim)

    def forward(self, inputs):
        """
        Args:
            inputs: (Batch_Size, Num_Inputs, Dim) - e.g., spatial features flattened
        Returns:
            slots: (Batch_Size, Num_Slots, Dim)
        """
        b, n, d = inputs.shape

        # Initialize slots
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.to_q(slots)

            # Dot product attention
            # (B, Num_Slots, Dim) x (B, Num_Inputs, Dim) -> (B, Num_Slots, Num_Inputs)
            dots = torch.einsum('bid,bjd->bij', q, k) * (self.dim ** -0.5)
            
            # Soft-TopK Routing
            # Each input should only route to Top-K slots
            # We apply Soft-TopK over the slots dimension (dim=1)
            k_routing = min(self.num_slots, self.topk)
            attn = self.soft_topk(dots, k=k_routing, dim=1) + self.eps
            
            # Normalize over inputs (standard Slot Attention normalization)
            # Note: The original paper normalizes over slots (dim=1) for the weights, 
            # but then normalizes over inputs (dim=2) for the weighted mean?
            # Actually, original Slot Attention:
            # attn = softmax(dots, dim=1)  # Competition between slots for each input
            # updates = weighted_mean(attn + epsilon, v, dim=2)
            # Let's stick to the standard but with Soft-TopK replacing Softmax
            
            attn = attn / attn.sum(dim=-1, keepdim=True) # Normalize over inputs

            # Weighted mean
            updates = torch.einsum('bjd,bij->bid', v, attn)

            # GRU Update
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, self.num_slots, d)

            # MLP Update
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

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
            # Corrected einsum expression
            dots = torch.einsum('bid,bjd->bij', q, k) * (self.dim ** -0.5)
            attn = dots.softmax(dim=1) + self.eps
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

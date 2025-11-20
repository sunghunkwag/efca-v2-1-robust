import torch
import torch.nn as nn

class SGWT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_slots = config.get('num_slots', 4)
        self.dim = config.get('dim', 128)
        self.slots = nn.Parameter(torch.randn(1, self.num_slots, self.dim))
        
    def forward(self, x):
        # Simple bottleneck implementation
        # In a real GWT, this would involve attention mechanisms
        return x + self.slots.mean(dim=1, keepdim=True)

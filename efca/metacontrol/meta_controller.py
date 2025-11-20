import torch
import torch.nn as nn

class MetaController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_dim = config.get('output_dim', 1)
        self.net = nn.Linear(64, self.output_dim) # Input dim placeholder
        
    def forward(self, x):
        return torch.tanh(self.net(x)) # Clamped output

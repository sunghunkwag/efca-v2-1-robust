import torch
import torch.nn as nn

class ClampedMetaController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_dim = config.get('output_dim', 1)
        # Input dim is currently hardcoded to 64 in the placeholder,
        # but in a real scenario it should come from the Probe or config.
        # I'll leave it as 64 for now or see if I can make it configurable.
        self.input_dim = config.get('input_dim', 64)
        self.net = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        return torch.tanh(self.net(x)) # Clamped output

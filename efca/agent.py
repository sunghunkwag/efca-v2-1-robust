import torch
import torch.nn as nn
from .perception.h_jepa import HJEPA
from .dynamics.ct_lnn import CTLNN
from .policy.task_policy import TaskPolicy

class EFCAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.perception = HJEPA(config.perception)
        self.dynamics = CTLNN(config.dynamics)
        self.policy = TaskPolicy(config.policy)
        
    def forward(self, obs):
        # Core Logic: Perception -> Dynamics -> Action
        z = self.perception(obs)
        s = self.dynamics(z)
        action = self.policy(s)
        return action

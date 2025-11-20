import torch
import torch.nn as nn
from .perception.h_jepa import HJEPA
from .dynamics.ct_lnn import CTLNN
from .policy.task_policy import TaskPolicy
from .bottleneck.s_gwt import SGWT
from .metacontrol.meta_controller import ClampedMetaController

class EFCAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.perception = HJEPA(config.perception)
        self.bottleneck = SGWT(config.bottleneck)
        self.dynamics = CTLNN(config.dynamics)
        self.policy = TaskPolicy(config.policy)
        self.meta_controller = ClampedMetaController(config.metacontrol)
        
    def forward(self, obs, meta_state=None):
        z = self.perception(obs)
        s_gwt = self.bottleneck(z)
        s_pooled = s_gwt.mean(dim=1)
        s_dyn = self.dynamics(s_pooled)
        action = self.policy(s_dyn)
        meta_delta = None
        if meta_state is not None:
            meta_delta = self.meta_controller(meta_state)
        return action, meta_delta

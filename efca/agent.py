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
        self.config = config
        
        # Determine if we are in Phase 0 (Meta-Controller disabled)
        # Check for 'phase' in config, default to 0 if not present for safety
        self.phase = config.get('phase', 0)
        self.meta_enabled = (self.phase > 0)

        # Initialize modules
        # Note: HJEPA and others might need specific kwargs unwrapped from config sub-dicts
        # depending on how they are implemented. Assuming they take a config dict or kwargs.
        # Based on previous file reads:
        # HJEPA(embed_dim=...)
        # CTLNN(input_dim=..., hidden_dim=..., output_dim=...)
        # TaskPolicy(hidden_dim=..., action_dim=...)
        # SGWT(config) - takes a dict/object
        # ClampedMetaController(config) - takes a dict/object

        # To handle the discrepancy between dictionary access and object attribute access
        # (config.perception vs config['perception']), I will convert config to a standard format
        # or handle it robustly. Assuming config is a dictionary from yaml.safe_load.

        # Helper to access config safely whether it's dict or object
        def get_cfg(c, key):
            if isinstance(c, dict):
                return c.get(key)
            return getattr(c, key, None)

        self.perception = HJEPA(embed_dim=get_cfg(get_cfg(config, 'h_jepa'), 'embed_dim'))

        # SGWT expects a config object/dict itself in its __init__
        self.bottleneck = SGWT(get_cfg(config, 'bottleneck'))

        ct_lnn_cfg = get_cfg(config, 'ct_lnn')
        self.dynamics = CTLNN(
            input_dim=get_cfg(ct_lnn_cfg, 'input_dim'),
            hidden_dim=get_cfg(ct_lnn_cfg, 'hidden_dim'),
            output_dim=get_cfg(ct_lnn_cfg, 'output_dim')
        )

        policy_cfg = get_cfg(config, 'task_policy')
        self.policy = TaskPolicy(
            hidden_dim=get_cfg(policy_cfg, 'hidden_dim'),
            action_dim=get_cfg(policy_cfg, 'action_dim')
        )

        if self.meta_enabled:
            self.meta_controller = ClampedMetaController(get_cfg(config, 'metacontrol'))
        else:
            self.meta_controller = None

    def forward(self, obs, h=None, meta_state=None):
        """
        Args:
            obs: Input observation (Image tensor)
            h: Hidden state for dynamics (optional)
            meta_state: State for meta-controller (optional)
        Returns:
            dist: Action distribution
            value: Value estimate
            h_new: New hidden state
            meta_delta: Meta-controller output (or None)
            perception_loss: Loss from perception module
        """
        # 1. Perception
        perception_loss, online_features = self.perception(obs)

        # online_features shape: (B, C, H, W). We need to flatten spatial dims for SGWT
        # SGWT expects (B, N, D).
        B, C, H, W = online_features.shape
        flat_features = online_features.permute(0, 2, 3, 1).reshape(B, H*W, C)

        # 2. Bottleneck (s-GWT)
        s_gwt = self.bottleneck(flat_features) # (B, Num_Slots, Dim)

        # Pool slots for Dynamics (e.g., mean pooling or another mechanism)
        # For now, simple mean pooling as in previous code
        s_pooled = s_gwt.mean(dim=1)

        # 3. Dynamics (CT-LNN)
        # CT-LNN expects (h, input). If h is None, it initializes.
        if h is None:
            h = self.dynamics.init_state(batch_size=B)
        h_new = self.dynamics(h, s_pooled)

        # 4. Policy
        dist, value = self.policy(h_new)

        # 5. Meta-Control
        meta_delta = None
        if self.meta_enabled and self.meta_controller is not None and meta_state is not None:
            meta_delta = self.meta_controller(meta_state)

        return dist, value, h_new, meta_delta, perception_loss

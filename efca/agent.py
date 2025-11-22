import torch
import torch.nn as nn
from .perception.h_jepa import HJEPA
from .dynamics.ct_lnn import CTLNN
from .policy.task_policy import TaskPolicy
from .bottleneck.s_gwt import SGWT
from .metacontrol.meta_controller import ClampedMetaController
from .probe.probe_network import ProbeNetwork
from .browser_interface import BrowserController

class EFCAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Determine if we are in Phase 0 (Meta-Controller disabled)
        # Check for 'phase' in config, default to 0 if not present for safety
        self.phase = config.get('phase', 0)
        self.meta_enabled = (self.phase > 0)

        # Helper to access config safely whether it's dict or object
        def get_cfg(c, key):
            if isinstance(c, dict):
                return c.get(key)
            return getattr(c, key, None)

        # Handle H-JEPA configuration
        h_jepa_cfg = get_cfg(config, 'h_jepa')
        if isinstance(h_jepa_cfg, dict):
            # Ensure input_type and input_dim are propagated if they exist in root config
            # and are not already in h_jepa_cfg
            if 'input_type' not in h_jepa_cfg and get_cfg(config, 'input_type'):
                h_jepa_cfg['input_type'] = get_cfg(config, 'input_type')
            if 'input_dim' not in h_jepa_cfg and get_cfg(config, 'input_dim'):
                h_jepa_cfg['input_dim'] = get_cfg(config, 'input_dim')

        self.perception = HJEPA(config=h_jepa_cfg)

        # SGWT expects a config object/dict itself in its __init__
        self.bottleneck = SGWT(get_cfg(config, 'bottleneck'))

        ct_lnn_cfg = get_cfg(config, 'ct_lnn')
        self.dynamics = CTLNN(config=ct_lnn_cfg)

        policy_cfg = get_cfg(config, 'task_policy')
        self.policy = TaskPolicy(
            hidden_dim=get_cfg(policy_cfg, 'hidden_dim'),
            action_dim=get_cfg(policy_cfg, 'action_dim'),
            action_space_type=get_cfg(policy_cfg, 'action_space_type') or 'discrete'
        )

        if self.meta_enabled:
            # Initialize Probe Network for metacognitive monitoring
            probe_config = get_cfg(config, 'probe')
            if probe_config is None:
                # Create default probe config if not provided
                probe_config = {
                    'h_jepa_dim': get_cfg(get_cfg(config, 'h_jepa'), 'embed_dim'),
                    'gwt_dim': get_cfg(get_cfg(config, 'bottleneck'), 'dim'),
                    'lnn_dim': get_cfg(get_cfg(config, 'ct_lnn'), 'output_dim'),
                    'output_dim': get_cfg(get_cfg(config, 'metacontrol'), 'input_dim'),
                    'hidden_dim': 128
                }
            self.probe = ProbeNetwork(probe_config)
            self.meta_controller = ClampedMetaController(get_cfg(config, 'metacontrol'))
        else:
            self.probe = None
            self.meta_controller = None

        # Device handling
        self.device = get_cfg(get_cfg(config, 'training'), 'device')
        if self.device:
            self.to(self.device)

        # Browser Controller
        if config.get('enable_browser', False):
            browser_config = get_cfg(config, 'browser')
            headless = browser_config.get('headless', False) if browser_config else False
            timeout = browser_config.get('timeout', 30000) if browser_config else 30000
            self.browser = BrowserController(headless=headless, default_timeout=timeout)
        else:
            self.browser = None

    def forward(self, obs, h=None, meta_state=None):
        """
        Args:
            obs: Input observation (Image tensor)
            h: Hidden state for dynamics (optional)
            meta_state: State for meta-controller (optional, deprecated - probe now generates it)
        Returns:
            dist: Action distribution
            value: Value estimate
            h_new: New hidden state
            meta_delta: Meta-controller output (or None)
            perception_loss: Loss from perception module
            probe_output: Probe network output (or None if meta disabled)
        """
        # Ensure inputs are on the correct device
        if obs.device != self.device:
            obs = obs.to(self.device)

        # 1. Perception
        perception_loss, online_features = self.perception(obs)

        # online_features shape:
        # - (B, C, H, W) for vision
        # - (B, D) for state
        # SGWT expects (B, N, D).

        if len(online_features.shape) == 4:
            B, C, H, W = online_features.shape
            flat_features = online_features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:
            # State input: (B, D) -> (B, 1, D)
            B, D = online_features.shape
            flat_features = online_features.unsqueeze(1)

        # 2. Bottleneck (s-GWT)
        s_gwt = self.bottleneck(flat_features) # (B, Num_Slots, Dim)

        # Pool slots for Dynamics (e.g., mean pooling or another mechanism)
        # For now, simple mean pooling as in previous code
        s_pooled = s_gwt.mean(dim=1)

        # 3. Dynamics (CT-LNN)
        # CT-LNN expects (h, input). If h is None, it initializes.
        if h is None:
            h = self.dynamics.init_state(batch_size=B).to(self.device)
        h_new = self.dynamics(h, s_pooled)

        # 4. Policy
        dist, value = self.policy(h_new)

        # 5. Probe Network (if meta-control enabled)
        probe_output = None
        if self.meta_enabled and self.probe is not None:
            probe_output = self.probe(
                h_jepa_features=online_features,
                gwt_slots=s_gwt,
                lnn_state=h_new,
                h_jepa_loss=perception_loss
            )
        
        # 6. Meta-Control
        meta_delta = None
        if self.meta_enabled and self.meta_controller is not None:
            # Use probe output as meta_state if available, otherwise use provided meta_state
            meta_input = probe_output if probe_output is not None else meta_state
            if meta_input is not None:
                meta_delta = self.meta_controller(meta_input)

        return dist, value, h_new, meta_delta, perception_loss, probe_output

    def execute_browser_action(self, action_type, **kwargs):
        if self.browser:
            if action_type == 'navigate':
                return self.browser.navigate(kwargs.get('url'))
            elif action_type == 'click':
                return self.browser.click(kwargs.get('selector'))
            elif action_type == 'type':
                return self.browser.type(kwargs.get('selector'), kwargs.get('text'))
            elif action_type == 'screenshot':
                return self.browser.screenshot(kwargs.get('path'))
            elif action_type == 'get_title':
                return self.browser.get_title()
            elif action_type == 'get_content':
                return self.browser.get_content()
        return "Browser not enabled"
    
    def cleanup(self):
        """Clean up resources, especially browser if enabled."""
        if self.browser is not None:
            try:
                self.browser.close()
            except Exception as e:
                print(f"Error closing browser: {e}")

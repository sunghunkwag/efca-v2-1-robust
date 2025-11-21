import torch


class ClampedDeltaAdapter:
    """
    Implements Clamped Delta Control from specification Section 2.3.
    
    Applies meta-controller outputs as percentage changes with hard clipping
    to prevent parameter explosion:
    
    Param_{t+1} = Param_t · (1 + clip(a_t^meta, -δ, +δ))
    
    Where:
    - a_t^meta: Output from meta-controller (tanh activation)
    - δ: Maximum change per step (e.g., 0.05 = 5%)
    """
    
    def __init__(self, delta_max: float = 0.05):
        """
        Initialize the parameter adapter.
        
        Args:
            delta_max (float): Maximum percentage change per step (default: 0.05 = 5%)
        """
        self.delta_max = delta_max
    
    def apply_delta(self, current_param: float, meta_action: torch.Tensor) -> float:
        """
        Apply clamped delta update to a parameter.
        
        Args:
            current_param (float): Current parameter value
            meta_action (torch.Tensor): Meta-controller output (already tanh-activated)
        
        Returns:
            float: Updated parameter value
        """
        # Extract scalar value if tensor
        if isinstance(meta_action, torch.Tensor):
            meta_action = meta_action.item()
        
        # Clip to maximum delta
        delta_clipped = torch.clamp(
            torch.tensor(meta_action), 
            -self.delta_max, 
            self.delta_max
        ).item()
        
        # Apply percentage change: Param_new = Param_old * (1 + delta)
        new_param = current_param * (1.0 + delta_clipped)
        
        # Safety bounds to prevent extreme values
        new_param = max(1e-6, min(10.0, new_param))
        
        return new_param
    
    def apply_delta_batch(self, current_params: dict, meta_actions: torch.Tensor) -> dict:
        """
        Apply clamped delta updates to multiple parameters.
        
        Args:
            current_params (dict): Dictionary of current parameter values
            meta_actions (torch.Tensor): Meta-controller outputs (B, num_params)
        
        Returns:
            dict: Updated parameter values
        """
        updated_params = {}
        param_names = list(current_params.keys())
        
        for i, param_name in enumerate(param_names):
            if i < meta_actions.shape[-1]:
                updated_params[param_name] = self.apply_delta(
                    current_params[param_name],
                    meta_actions[0, i] if meta_actions.dim() > 1 else meta_actions[i]
                )
            else:
                # Keep unchanged if no meta-action for this parameter
                updated_params[param_name] = current_params[param_name]
        
        return updated_params

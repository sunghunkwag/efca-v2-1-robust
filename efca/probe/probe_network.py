import torch
import torch.nn as nn
from typing import Dict, Optional


class ProbeNetwork(nn.Module):
    """
    Probe Network for metacognitive monitoring.
    
    This module monitors the internal states of the EFCA agent's components
    (H-JEPA, s-GWT, CT-LNN) and generates a structured representation for the
    meta-controller to use in adjusting agent parameters.
    
    Essential for Phase 1+ where the agent needs self-awareness of its internal
    processing states.
    """
    
    def __init__(self, config: Dict) -> None:
        """
        Initializes the Probe Network.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - h_jepa_dim: Dimension of H-JEPA features
                - gwt_dim: Dimension of GWT slot features
                - lnn_dim: Dimension of CT-LNN hidden state
                - output_dim: Output dimension for meta-controller
                - hidden_dim: Hidden dimension for internal processing
        """
        super().__init__()
        
        # Extract configuration
        self.h_jepa_dim = config.get('h_jepa_dim', 768)
        self.gwt_dim = config.get('gwt_dim', 768)
        self.lnn_dim = config.get('lnn_dim', 256)
        self.output_dim = config.get('output_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Probe modules for each component
        # H-JEPA Probe: Monitors perception uncertainty
        self.h_jepa_probe = nn.Sequential(
            nn.Linear(self.h_jepa_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # GWT Probe: Monitors bottleneck activity patterns
        self.gwt_probe = nn.Sequential(
            nn.Linear(self.gwt_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # CT-LNN Probe: Monitors dynamics state statistics
        self.lnn_probe = nn.Sequential(
            nn.Linear(self.lnn_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        
        # Aggregation network: Combines all probe outputs
        combined_dim = 3 * (self.hidden_dim // 2)
        self.aggregator = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
    def probe_h_jepa(self, features: torch.Tensor, loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Probe H-JEPA perception module.
        
        Monitors the perceptual features and optionally the reconstruction loss
        to assess perception uncertainty.
        
        Args:
            features (torch.Tensor): Online features from H-JEPA (B, C, H, W)
            loss (Optional[torch.Tensor]): Reconstruction loss (scalar)
            
        Returns:
            torch.Tensor: Probe output (B, hidden_dim // 2)
        """
        # Spatial pooling to get global features
        B, C, H, W = features.shape
        pooled = features.mean(dim=[2, 3])  # (B, C)
        
        # Process through probe network
        probe_out = self.h_jepa_probe(pooled)
        
        return probe_out
    
    def probe_gwt(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Probe s-GWT bottleneck module.
        
        Monitors the slot attention patterns to assess information competition
        and workspace activity.
        
        Args:
            slots (torch.Tensor): GWT slots (B, Num_Slots, Dim)
            
        Returns:
            torch.Tensor: Probe output (B, hidden_dim // 2)
        """
        # Pool over slots to get workspace summary
        B, N, D = slots.shape
        pooled = slots.mean(dim=1)  # (B, D)
        
        # Process through probe network
        probe_out = self.gwt_probe(pooled)
        
        return probe_out
    
    def probe_lnn(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Probe CT-LNN dynamics module.
        
        Monitors the hidden state of the dynamics module to assess temporal
        processing and state evolution.
        
        Args:
            hidden_state (torch.Tensor): CT-LNN hidden state (B, D)
            
        Returns:
            torch.Tensor: Probe output (B, hidden_dim // 2)
        """
        # Process through probe network
        probe_out = self.lnn_probe(hidden_state)
        
        return probe_out
    
    def forward(
        self, 
        h_jepa_features: torch.Tensor,
        gwt_slots: torch.Tensor,
        lnn_state: torch.Tensor,
        h_jepa_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Probe Network.
        
        Aggregates information from all monitored components into a single
        state representation for the meta-controller.
        
        Implements "Freezed Encoder" from specification Section 3:
        The probe reads h(t) but does NOT backpropagate gradients into the
        monitored modules (H-JEPA, s-GWT, CT-LNN). This isolates the observer
        from the observed.
        
        Args:
            h_jepa_features (torch.Tensor): Online features from H-JEPA (B, C, H, W)
            gwt_slots (torch.Tensor): GWT slots (B, Num_Slots, Dim)
            lnn_state (torch.Tensor): CT-LNN hidden state (B, D)
            h_jepa_loss (Optional[torch.Tensor]): H-JEPA reconstruction loss
            
        Returns:
            torch.Tensor: Aggregated probe output for meta-controller (B, output_dim)
        """
        # CRITICAL: Detach inputs to prevent gradient backpropagation
        # This implements the "Freezed Encoder" requirement
        h_jepa_features_detached = h_jepa_features.detach()
        gwt_slots_detached = gwt_slots.detach()
        lnn_state_detached = lnn_state.detach()
        
        # Probe each component with detached inputs
        h_jepa_probe = self.probe_h_jepa(h_jepa_features_detached, h_jepa_loss)
        gwt_probe = self.probe_gwt(gwt_slots_detached)
        lnn_probe = self.probe_lnn(lnn_state_detached)
        
        # Concatenate all probe outputs
        combined = torch.cat([h_jepa_probe, gwt_probe, lnn_probe], dim=-1)
        
        # Aggregate into final output
        output = self.aggregator(combined)
        
        return output
    
    def get_statistics(
        self,
        h_jepa_features: torch.Tensor,
        gwt_slots: torch.Tensor,
        lnn_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract statistical information from internal states.
        
        Useful for logging and visualization during training.
        
        Args:
            h_jepa_features (torch.Tensor): Online features from H-JEPA
            gwt_slots (torch.Tensor): GWT slots
            lnn_state (torch.Tensor): CT-LNN hidden state
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of statistics
        """
        stats = {}
        
        # H-JEPA statistics
        stats['h_jepa_mean'] = h_jepa_features.mean()
        stats['h_jepa_std'] = h_jepa_features.std()
        stats['h_jepa_max'] = h_jepa_features.max()
        
        # GWT statistics
        stats['gwt_mean'] = gwt_slots.mean()
        stats['gwt_std'] = gwt_slots.std()
        stats['gwt_slot_variance'] = gwt_slots.var(dim=1).mean()
        
        # CT-LNN statistics
        stats['lnn_mean'] = lnn_state.mean()
        stats['lnn_std'] = lnn_state.std()
        stats['lnn_max'] = lnn_state.max()
        
        return stats

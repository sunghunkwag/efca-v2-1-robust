from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import numpy as np


class StateEncoder(nn.Module):
    """
    Simple MLP encoder for state-based inputs.
    Reference: dhc_ssm/adapters/rl_policy_v2.py
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        # Key feature: LayerNorm and Tanh for RL stability
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        # Orthogonal initialization (crucial for PPO)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

    def forward(self, x):
        return self.net(x)


class HJEPA(nn.Module):
    """
    Hierarchical Joint-Embedding Predictive Architecture (H-JEPA) for perception.

    This module uses a ConvNeXt-Tiny backbone to learn hierarchical representations
    of input data through self-supervised predictive coding. It consists of an
    online encoder, a target encoder (updated via EMA), and a predictor network.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the H-JEPA module.

        Args:
            config (dict): Configuration dictionary containing parameters like
                           embed_dim, predictor_depth, gamma_reg, and use_hinge_loss.
        """
        super().__init__()
        self.embed_dim = config.get('embed_dim', 768)
        predictor_depth = config.get('predictor_depth', 2)
        # L2 Regularization parameter (γ_reg in specification)
        self.gamma_reg = config.get('gamma_reg', 0.01)
        # Hinge loss fallback (specification Section 3)
        self.use_hinge_loss = config.get('use_hinge_loss', False)
        self.hinge_margin = config.get('hinge_margin', 1.0)

        # Input type handling
        self.input_type = config.get('input_type', 'vision')
        self.input_dim = config.get('input_dim', None)

        if self.input_type == 'state':
            if self.input_dim is None:
                raise ValueError("input_dim must be provided for state input")

            # Use StateEncoder for state inputs
            self.online_encoder = StateEncoder(self.input_dim, self.embed_dim)
            self.target_encoder = StateEncoder(self.input_dim, self.embed_dim)

            # Projections are identity or integrated in StateEncoder
            # For compatibility with update_target_encoder, we can keep them as identity or
            # simply skip them if we adapt update_target_encoder
            # But the original code had projections separate.
            # StateEncoder outputs embed_dim directly.

            # Let's create dummy identity projections to satisfy existing structure or set to None
            # The original update_target_encoder iterates over projection parameters.
            # We should probably define them as Identity or handle it.
            self.online_projection = nn.Identity()
            self.target_projection = nn.Identity()

        else:
            # Online Encoder (ConvNeXt-Tiny)
            self.online_encoder: nn.Module = timm.create_model(
                "convnext_tiny", pretrained=True, features_only=True
            )

            # Target Encoder (ConvNeXt-Tiny)
            self.target_encoder: nn.Module = timm.create_model(
                "convnext_tiny", pretrained=True, features_only=True
            )

            # Predictor Network (MLP)
            # Note: The predictor must handle the same embedding dimension as the encoder output.
            # ConvNeXt-Tiny output dim is 768
            self.backbone_dim = 768

            # Projections to embedding dimension
            self.online_projection = nn.Linear(self.backbone_dim, self.embed_dim)
            self.target_projection = nn.Linear(self.backbone_dim, self.embed_dim)

            # Initialize target projection with online projection weights
            self.target_projection.load_state_dict(self.online_projection.state_dict())
            for param in self.target_projection.parameters():
                param.requires_grad = False

        # Freeze target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor Network (MLP)
        self.predictor: nn.Module = self._build_predictor(self.embed_dim, predictor_depth)

    def _build_predictor(self, embed_dim: int, depth: int) -> nn.Module:
        """
        Builds the predictor network as a simple Multi-Layer Perceptron (MLP).

        Args:
            embed_dim (int): The input and output dimensionality of the MLP.
            depth (int): The number of linear layers in the MLP.

        Returns:
            nn.Module: The constructed predictor network.
        """
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(embed_dim, embed_dim), nn.ReLU()])

        if len(layers) > 0:
            # Remove last ReLU
            layers.pop()

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the H-JEPA module.

        Args:
            x (torch.Tensor): The input tensor.
                - Shape (B, C, H, W) for vision.
                - Shape (B, D) for state.

        Returns:
            A tuple containing:
            - The reconstruction loss.
            - The online features.
        """
        if self.input_type == 'state':
            return self._forward_state(x)
        else:
            return self._forward_vision(x)

    def _forward_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Get feature representations
        with torch.no_grad():
            target_features = self.target_encoder(x)

        online_features = self.online_encoder(x)

        # 2. Generate mask (element-wise on the feature vector)
        # Mask shape: (B, embed_dim)
        mask = torch.rand_like(online_features).ge(0.5).to(x.device)

        # 3. Apply mask
        masked_online_features = online_features * mask.float()

        # 4. Predict
        predicted_features = self.predictor(masked_online_features)

        # 5. Loss
        if self.use_hinge_loss:
            diff = predicted_features * (1 - mask.float()) - target_features * (1 - mask.float())
            reconstruction_loss = torch.mean(torch.clamp(diff.norm(dim=1) - self.hinge_margin, min=0))
        else:
            reconstruction_loss = nn.functional.mse_loss(
                predicted_features * (1 - mask.float()),
                target_features * (1 - mask.float()),
            )

        l2_regularization = self.gamma_reg * (online_features ** 2).mean()
        loss = reconstruction_loss + l2_regularization

        return loss, online_features

    def _forward_vision(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Get feature representations from both encoders
        with torch.no_grad():
            # (B, 768, H, W) -> (B, H, W, 768)
            target_backbone = self.target_encoder(x)[-1].permute(0, 2, 3, 1)
            # Project to embed_dim: (B, H, W, 256)
            target_features = self.target_projection(target_backbone)
            # Back to (B, 256, H, W)
            target_features = target_features.permute(0, 3, 1, 2)

        # Online path
        online_backbone = self.online_encoder(x)[-1].permute(0, 2, 3, 1)
        online_features = self.online_projection(online_backbone).permute(0, 3, 1, 2)

        # 2. Generate a mask
        mask = (
            torch.rand(
                online_features.shape[0],
                1,
                online_features.shape[2],
                online_features.shape[3],
            )
            .to(x.device)
            .ge(0.5)
        )

        # 3. Apply the mask to the online features
        masked_online_features = online_features * mask.float()

        # 4. Predict the target features from the masked online features
        # Permute to (B, H, W, C) for the linear layer
        masked_online_features_permuted = masked_online_features.permute(0, 2, 3, 1)
        predicted_features_permuted = self.predictor(masked_online_features_permuted)
        # Permute back to (B, C, H, W)
        predicted_features = predicted_features_permuted.permute(0, 3, 1, 2)

        # 5. Calculate the reconstruction loss on the masked patches
        if self.use_hinge_loss:
            diff = predicted_features * (1 - mask.float()) - target_features * (1 - mask.float())
            reconstruction_loss = torch.mean(torch.clamp(diff.norm(dim=1) - self.hinge_margin, min=0))
        else:
            reconstruction_loss = nn.functional.mse_loss(
                predicted_features * (1 - mask.float()),
                target_features * (1 - mask.float()),
            )
        
        # Add L2 regularization term to prevent representation collapse
        l2_regularization = self.gamma_reg * (online_features ** 2).mean()
        
        # Total loss with regularization
        loss = reconstruction_loss + l2_regularization

        return loss, online_features

    def update_target_encoder(self, tau: float = 0.996) -> None:
        """
        Update the target encoder's weights using an exponential moving average (EMA).

        Args:
            tau (float): The momentum parameter for the EMA update.
        """
        # Update backbone
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.copy_(
                tau * target_param.data + (1 - tau) * online_param.data
            )
            
        # Update projection (only if not identity)
        if not isinstance(self.online_projection, nn.Identity):
            for online_param, target_param in zip(
                self.online_projection.parameters(), self.target_projection.parameters()
            ):
                target_param.data.copy_(
                    tau * target_param.data + (1 - tau) * online_param.data
                )

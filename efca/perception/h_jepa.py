from typing import List, Tuple

import timm
import torch
import torch.nn as nn


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

        # Online Encoder (ConvNeXt-Tiny)
        self.online_encoder: nn.Module = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True
        )

        # Target Encoder (ConvNeXt-Tiny)
        self.target_encoder: nn.Module = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True
        )

        # Freeze target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor Network (MLP)
        # Note: The predictor must handle the same embedding dimension as the encoder output.
        # ConvNeXt-Tiny output dim is 768.
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

        # Add final projection to ensure output matches embed_dim (if last layer shouldn't have ReLU)
        # or just to be explicit.
        # The user prompt said "Check predictor dimensions".
        # The current loop adds `depth` layers of Linear+ReLU.
        # A predictor usually ends with a linear layer (no activation) to match target values unrestricted.
        # So I will modify the last layer to be linear only.

        if len(layers) > 0:
            # Remove last ReLU
            layers.pop()

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the H-JEPA module.

        This implementation performs masked patch prediction and returns both the
        reconstruction loss and the feature representation from the online encoder.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            A tuple containing:
            - The reconstruction loss.
            - The online features.
        """
        # 1. Get feature representations from both encoders
        with torch.no_grad():
            target_features = self.target_encoder(x)[-1]

        online_features = self.online_encoder(x)[-1]

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
        # Implements specification Section 2.1:
        # L_JEPA = Σ_k λ_k (||ẑ_k - sg(z_k^target)||² + γ_reg ||z_k||²)
        
        if self.use_hinge_loss:
            # Hinge loss fallback (specification Section 3 table)
            # Use if L2 collapses
            diff = predicted_features * (1 - mask.float()) - target_features * (1 - mask.float())
            reconstruction_loss = torch.mean(torch.clamp(diff.norm(dim=1) - self.hinge_margin, min=0))
        else:
            # Standard MSE loss
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

        This is a common technique in self-supervised learning to create a slowly
        progressing, more stable target representation.

        Args:
            tau (float): The momentum parameter for the EMA update.
        """
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.copy_(
                tau * target_param.data + (1 - tau) * online_param.data
            )

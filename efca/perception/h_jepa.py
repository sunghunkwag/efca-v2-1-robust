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

    def __init__(
        self,
        embed_dim: int = 768,
        predictor_depth: int = 2,
        input_shape: Tuple[int, ...] = None,
    ) -> None:
        """
        Initializes the H-JEPA module.

        Args:
            embed_dim (int): The dimensionality of the embedding space.
            predictor_depth (int): The number of layers in the predictor network.
            input_shape (Tuple[int, ...]): The shape of the input tensor.
                If 1D (e.g. (D,)), uses MLP encoder.
                If 3D (e.g. (C, H, W)) or None, uses ConvNeXt.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_shape = input_shape

        # Determine if we are in state mode or vision mode
        self.is_state_input = (
            input_shape is not None and len(input_shape) == 1
        )

        if self.is_state_input:
            input_dim = input_shape[0]
            self.online_encoder = self._build_mlp_encoder(input_dim, embed_dim)
            self.target_encoder = self._build_mlp_encoder(input_dim, embed_dim)
        else:
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
        self.predictor: nn.Module = self._build_predictor(embed_dim, predictor_depth)

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

    def _build_mlp_encoder(self, input_dim: int, embed_dim: int) -> nn.Module:
        """
        Builds a simple MLP encoder for state inputs.
        """
        return nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

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
        if self.is_state_input:
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
        loss = nn.functional.mse_loss(
            predicted_features * (1 - mask.float()),
            target_features * (1 - mask.float()),
        )

        return loss, online_features

    def _forward_vision(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        loss = nn.functional.mse_loss(
            predicted_features * (1 - mask.float()),
            target_features * (1 - mask.float()),
        )

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

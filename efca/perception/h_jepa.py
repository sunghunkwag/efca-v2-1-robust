import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block

class HJEPA(nn.Module):
    """
    Hierarchical Joint-Embedding Predictive Architecture (H-JEPA)

    This module uses a ConvNeXt-Tiny backbone as specified in the EFCA-v2.1
    document. It learns by predicting representations of a target block
    from a context block within the same image.

    Key features for stability:
    - Stop-Gradient on the target encoder's output.
    - Exponential Moving Average (EMA) for updating the target encoder.
    """
    def __init__(self, model_name='convnext_tiny', pretrained=True, ema_decay=0.996):
        super().__init__()
        self.ema_decay = ema_decay

        # --- Online Network ---
        self.online_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # We want feature maps, not a final classification
        )

        # --- Target Network ---
        self.target_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )

        # Synchronize target and online networks initially
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

        # Freeze the target encoder parameters
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- Predictor ---
        # The predictor is a series of Transformer blocks that operate on patch embeddings
        # The embedding dimension depends on the output of the ConvNeXt backbone
        # For ConvNeXt-Tiny, the output feature dimensions are (96, 192, 384, 768)
        # Let's use the last feature map, which has a dimension of 768
        embed_dim = self.online_encoder.feature_info.info[-1]['num_chs']
        self.predictor = nn.Sequential(
            *[Block(dim=embed_dim, num_heads=8) for _ in range(3)]
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def _update_target_network(self):
        """
        Update the target network using an exponential moving average (EMA)
        of the online network's weights.
        """
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.ema_decay * target_param.data + (1 - self.ema_decay) * online_param.data

    def forward(self, x, context_mask, target_mask):
        """
        Forward pass for H-JEPA.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            context_mask (torch.Tensor): A boolean mask indicating the context patches.
            target_mask (torch.Tensor): A boolean mask indicating the target patches.

        Returns:
            torch.Tensor: The prediction loss.
            dict: A dictionary containing debugging info.
        """
        B, C, H, W = x.shape
        embed_dim = self.online_encoder.feature_info.info[-1]['num_chs']

        # --- 1. Get Representations from Encoders ---
        # Get online features for context
        online_features_flat = self.online_encoder(x)[-1].flatten(2).permute(0, 2, 1)

        # Get target features for targets (with no gradient)
        with torch.no_grad():
            self._update_target_network() # EMA update
            target_features_flat = self.target_encoder(x)[-1].flatten(2).permute(0, 2, 1)

        # --- 2. Select Context and Target Patches ---
        # This assumes a fixed number of patches per item in the batch, which is true for the example.
        num_context_patches = context_mask[0].sum().item()
        num_target_patches = target_mask[0].sum().item()

        context_embeddings = online_features_flat[context_mask].reshape(B, num_context_patches, embed_dim)
        with torch.no_grad():
            target_embeddings = target_features_flat[target_mask].reshape(B, num_target_patches, embed_dim).detach()

        # --- 3. Predict Target from Context (MAE-style) ---
        # Prepare input for the predictor by combining context with learnable mask tokens
        mask_tokens = self.mask_token.repeat(B, num_target_patches, 1)
        predictor_input = torch.cat([context_embeddings, mask_tokens], dim=1)

        # Get predictions
        predictions_all = self.predictor(predictor_input)

        # We only care about the outputs corresponding to the mask tokens
        predicted_embeddings = predictions_all[:, -num_target_patches:, :]

        # --- 4. Calculate Loss ---
        loss = self.loss_fn(predicted_embeddings, target_embeddings)

        # L2 regularization on online encoder's latent vectors for stability
        # Note: This is a simplified interpretation of the spec's regularization term.
        l2_reg = 0.0
        for param in self.online_encoder.parameters():
            l2_reg += torch.norm(param)

        total_loss = loss + 1e-6 * l2_reg # Small regularization factor

        return total_loss, {"loss": loss.item(), "l2_reg": l2_reg.item()}

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HJEPA().to(device)

    # Create a dummy image
    dummy_image = torch.randn(4, 3, 224, 224).to(device)

    # Create dummy masks for a 7x7 patch grid (output of ConvNeXt-Tiny for 224x224)
    num_patches = 49

    # Create boolean masks
    context_mask_bool = torch.zeros(4, num_patches, dtype=torch.bool, device=device)
    context_mask_bool[:, :25] = True # Use first ~50% as context

    target_mask_bool = torch.zeros(4, num_patches, dtype=torch.bool, device=device)
    target_mask_bool[:, 25:40] = True # Use next ~30% as target

    loss, info = model(dummy_image, context_mask_bool, target_mask_bool)

    print(f"Loss: {loss.item()}")
    print(f"Info: {info}")

    # Check that gradients are flowing only to the online encoder and predictor
    loss.backward()

    online_grad = any(p.grad is not None for p in model.online_encoder.parameters())
    predictor_grad = any(p.grad is not None for p in model.predictor.parameters())
    target_grad = any(p.grad is not None for p in model.target_encoder.parameters())

    print(f"Online encoder has gradients: {online_grad}")       # Should be True
    print(f"Predictor has gradients: {predictor_grad}")         # Should be True
    print(f"Target encoder has gradients: {target_grad}")       # Should be False

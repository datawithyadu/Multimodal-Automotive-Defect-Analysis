"""
Concatenation fusion classifier for multimodal severity prediction.

Architecture:
    EfficientNet-B0  → image features (1280-dim)
                                                  ┐
                                             Concat (2048-dim)
                                                  ┘
    DistilBERT       → text  features  (768-dim)

    → Dropout → Linear(2048 → 512) → ReLU → Dropout → Linear(512 → 3)
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel


class ConcatFusionClassifier(nn.Module):
    """
    Concatenation fusion model for image + text severity prediction.

    Both encoders are partially frozen (same strategy as Phase 1/2).
    The classification head operates on the concatenated feature vector.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.4,
        # Image encoder settings
        freeze_image_backbone: bool = True,
        unfreeze_image_last_n_blocks: int = 3,
        # Text encoder settings
        freeze_text_backbone: bool = True,
        unfreeze_text_last_n_layers: int = 2,
    ):
        super().__init__()

        # ── Image encoder: EfficientNet-B0 ────────────────────────
        self.image_encoder = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Remove the built-in classifier head — we only want features
        self.image_encoder.classifier = nn.Identity()

        if freeze_image_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Unfreeze last N feature blocks
            total_blocks = len(self.image_encoder.features)
            for i in range(total_blocks - unfreeze_image_last_n_blocks, total_blocks):
                for param in self.image_encoder.features[i].parameters():
                    param.requires_grad = True

        image_feature_dim = 1280  # EfficientNet-B0 avgpool output

        # ── Text encoder: DistilBERT ───────────────────────────────
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_text_backbone:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            total_layers = len(self.text_encoder.transformer.layer)
            for i in range(total_layers - unfreeze_text_last_n_layers, total_layers):
                for param in self.text_encoder.transformer.layer[i].parameters():
                    param.requires_grad = True

        text_feature_dim = self.text_encoder.config.hidden_size  # 768

        # ── Classification head ────────────────────────────────────
        fused_dim = image_feature_dim + text_feature_dim  # 1280 + 768 = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 1280-dim image features via EfficientNet-B0.

        Returns: (batch, 1280)
        """
        features = self.image_encoder.features(images)
        features = self.image_encoder.avgpool(features)
        features = torch.flatten(features, 1)
        return features

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract 768-dim [CLS] text features via DistilBERT.

        Returns: (batch, 768)
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images:         (batch, 3, H, W)
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_classes)
        """
        img_features  = self.get_image_features(images)                  # (batch, 1280)
        text_features = self.get_text_features(input_ids, attention_mask)  # (batch,  768)
        fused = torch.cat([img_features, text_features], dim=1)           # (batch, 2048)
        return self.classifier(fused)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

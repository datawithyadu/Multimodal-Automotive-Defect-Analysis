"""
EfficientNet-B0 image classifier for severity prediction.

Uses transfer learning with frozen early layers and fine-tuned later layers.
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for car damage severity.

    Architecture:
        EfficientNet-B0 (pre-trained, partially frozen)
        → Adaptive Average Pooling
        → Dropout
        → Linear (1280 → num_classes)
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
    ):
        super().__init__()

        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze the last N blocks of the features
            # EfficientNet-B0 has 8 blocks (features[0] through features[8])
            if unfreeze_last_n_blocks > 0:
                total_blocks = len(self.backbone.features)
                for i in range(total_blocks - unfreeze_last_n_blocks, total_blocks):
                    for param in self.backbone.features[i].parameters():
                        param.requires_grad = True

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features  # 1280 for B0
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (before the classification head).
        Useful for fusion architectures later.

        Returns: Tensor of shape (batch_size, 1280)
        """
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features

    def count_trainable_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

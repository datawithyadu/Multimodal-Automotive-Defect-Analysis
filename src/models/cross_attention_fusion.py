"""
Cross-Attention Fusion classifier for multimodal severity prediction.

Architecture:
    Image features (7x7x1280) → Flatten → Linear → Query (Q)
    Text features (seqx768)   → Linear → Key (K), Value (V)
    Cross-Attention(Q, K, V)
    → Global Average Pooling
    → Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel


class CrossAttentionFusionClassifier(nn.Module):
    """
    Cross-attention fusion model for image + text severity prediction.

    Image spatial features act as Queries.
    Text token features act as Keys and Values.
    This allows each image region to attend to relevant text context.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.4,
        d_k: int = 512,
        num_heads: int = 8,
        # Modality dropout — randomly zero out an entire modality during training
        modality_dropout_p: float = 0.1,
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
        # Remove the built-in classifier head and average pooling
        self.image_encoder.classifier = nn.Identity()
        self.image_encoder.avgpool = nn.Identity()

        if freeze_image_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # Unfreeze last N feature blocks
            total_blocks = len(self.image_encoder.features)
            for i in range(total_blocks - unfreeze_image_last_n_blocks, total_blocks):
                for param in self.image_encoder.features[i].parameters():
                    param.requires_grad = True

        image_feature_dim = 1280  # EfficientNet-B0 feature map depth

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

        # ── Modality Dropout ────────────────────────────────────────
        # During training, randomly zero out entire image or text features
        # to prevent the model from over-relying on a single modality
        self.modality_dropout_p = modality_dropout_p

        # ── Projections ────────────────────────────────────────────
        self.d_k = d_k
        self.query_proj = nn.Linear(image_feature_dim, d_k)
        self.key_proj = nn.Linear(text_feature_dim, d_k)
        self.value_proj = nn.Linear(text_feature_dim, d_k)

        # ── Cross-Attention ────────────────────────────────────────
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_k,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate,
        )

        # ── Classification head ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(d_k, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass.

        Args:
            images:         (batch, 3, H, W)
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_attention_weights: whether to return the cross-attention weights

        Returns:
            logits: (batch, num_classes)
            (optional) attn_weights: (batch, num_image_regions, seq_len)
        """
        # 1. Image Features (Queries)
        # EfficientNet-B0 features: (batch, 1280, 7, 7) for 224x224 input
        img_features = self.image_encoder.features(images)
        batch_size, c, h, w = img_features.shape

        # Flatten spatial dims: (batch, 1280, 49)
        img_features = img_features.view(batch_size, c, -1)
        # Transpose to sequence format: (batch, 49, 1280)
        img_features = img_features.transpose(1, 2)

        # L2 normalize image features before projection
        img_features = F.normalize(img_features, p=2, dim=-1)

        # 2. Text Features (Keys, Values)
        # DistilBERT outputs: (batch, seq_len, 768)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_outputs.last_hidden_state

        # L2 normalize text features before projection
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 3. Modality Dropout (training only)
        # Randomly zero out an entire modality to encourage robust learning
        if self.training and self.modality_dropout_p > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout_p:
                # Drop image: zero out queries, model must rely on text
                img_features = torch.zeros_like(img_features)
            elif r < 2 * self.modality_dropout_p:
                # Drop text: zero out keys/values, model must rely on image
                text_features = torch.zeros_like(text_features)

        # Project to d_k: (batch, 49, d_k)
        queries = self.query_proj(img_features)

        # Project to d_k: (batch, seq_len, d_k)
        keys = self.key_proj(text_features)
        values = self.value_proj(text_features)

        # 4. Cross-Attention
        # key_padding_mask expects True for padding tokens (which we want to ignore).
        # transformers attention_mask is 1 for real tokens, 0 for pad.
        key_padding_mask = (attention_mask == 0)

        attn_output, attn_weights = self.cross_attention(
            query=queries,
            key=keys,
            value=values,
            key_padding_mask=key_padding_mask,
        )  # attn_output: (batch, 49, d_k), attn_weights: (batch, 49, seq_len)

        # 5. Aggregation
        # Global average pool over the spatial sequence dimension
        # (batch, 49, d_k) -> (batch, d_k)
        fused_features = attn_output.mean(dim=1)

        # 5. Classification
        logits = self.classifier(fused_features)

        if return_attention_weights:
            return logits, attn_weights
        return logits

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

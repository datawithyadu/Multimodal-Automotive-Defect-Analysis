"""
DistilBERT text classifier for severity prediction.

Uses transfer learning with frozen early transformer layers
and fine-tuned later layers.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    """
    DistilBERT-based classifier for car damage severity.

    Architecture:
        DistilBERT (pre-trained, partially frozen)
        → [CLS] token embedding (768-dim)
        → Dropout
        → Linear (768 → num_classes)
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 2,
    ):
        super().__init__()

        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

            # Unfreeze the last N transformer layers
            # DistilBERT has 6 transformer layers (0-5)
            if unfreeze_last_n_layers > 0:
                total_layers = len(self.bert.transformer.layer)
                for i in range(total_layers - unfreeze_last_n_layers, total_layers):
                    for param in self.bert.transformer.layer[i].parameters():
                        param.requires_grad = True

        # Classification head
        hidden_size = self.bert.config.hidden_size  # 768 for distilbert-base
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) — tokenized text
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, num_classes)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token is the first token
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits = self.classifier(cls_embedding)
        return logits

    def get_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract [CLS] token features (before classification head).
        Useful for fusion architectures later.

        Returns: Tensor of shape (batch_size, 768)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def get_token_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract ALL token embeddings (for cross-attention fusion).
        Returns full sequence output, not just [CLS].

        Returns: Tensor of shape (batch_size, seq_len, 768)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def count_trainable_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

"""
PyTorch Dataset for text-only severity classification.

Loads synthetic text descriptions from multimodal_dataset.csv
and tokenizes them using DistilBERT's tokenizer.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

# Same label mapping as image dataset
LABEL_MAP = {"minor": 0, "moderate": 1, "severe": 2}
LABEL_NAMES = ["minor", "moderate", "severe"]
NUM_CLASSES = 3

# Default max token length — our texts average ~30 words ≈ ~40 tokens
MAX_TOKEN_LENGTH = 64


class TextDamageDataset(Dataset):
    """
    Dataset for text-only severity classification.

    Each sample returns:
        input_ids:      (max_length,) — tokenized text
        attention_mask: (max_length,) — 1 for real tokens, 0 for padding
        label:          scalar — severity class (0=minor, 1=moderate, 2=severe)

    Also stores image_paths for group-aware fold splitting (all 3 text
    variants of the same image must stay in the same fold).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        image_paths: list[str],
        tokenizer: DistilBertTokenizer = None,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        assert len(texts) == len(labels) == len(image_paths), \
            "texts, labels, and image_paths must have same length"

        self.texts = texts
        self.labels = labels
        self.image_paths = image_paths
        self.max_length = max_length
        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            encoding["input_ids"].squeeze(0),       # (max_length,)
            encoding["attention_mask"].squeeze(0),   # (max_length,)
            torch.tensor(label, dtype=torch.long),
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        split: str = None,
        tokenizer: DistilBertTokenizer = None,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        """
        Create dataset from multimodal_dataset.csv.

        Args:
            csv_path: Path to multimodal_dataset.csv
            split: Filter by split ('training' or 'testing'). None = all data.
            tokenizer: Pre-initialized tokenizer (shared across datasets).
            max_length: Max token length for tokenizer.
        """
        df = pd.read_csv(csv_path)

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)

        texts = df["text"].tolist()
        labels = df["severity_encoded"].tolist()
        image_paths = df["image_path"].tolist()

        return cls(texts, labels, image_paths, tokenizer, max_length)

"""
Multimodal PyTorch Dataset for image + text severity classification.

Each sample is one row from multimodal_dataset.csv:
    image: loaded from image_path and transformed
    text:  tokenized with DistilBertTokenizer
    label: severity_encoded (0=minor, 1=moderate, 2=severe)

Since each image has 3 text variants, GroupKFold by image_path
ensures all 3 variants of the same image stay in the same fold.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer

LABEL_MAP = {"minor": 0, "moderate": 1, "severe": 2}
LABEL_NAMES = ["minor", "moderate", "severe"]
NUM_CLASSES = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MAX_TOKEN_LENGTH = 64


def get_train_transforms(image_size: int = 224):
    """Training transforms with augmentation — same as Phase 1."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation/test transforms — no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MultimodalDamageDataset(Dataset):
    """
    Dataset for multimodal (image + text) severity classification.

    Each sample returns:
        image:          (3, H, W) image tensor
        input_ids:      (max_length,) tokenized text
        attention_mask: (max_length,) 1 for real tokens, 0 for padding
        label:          scalar severity class

    Also exposes image_paths for GroupKFold splitting.
    """

    def __init__(
        self,
        image_paths: list[str],
        texts: list[str],
        labels: list[int],
        tokenizer: DistilBertTokenizer,
        image_transform=None,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        assert len(image_paths) == len(texts) == len(labels), \
            "image_paths, texts and labels must have same length"

        self.image_paths = image_paths
        self.labels = labels
        self.image_transform = image_transform or get_val_transforms()
        self.texts = texts  # Keep raw texts for subset creation

        # Pre-tokenize ALL texts at init time (batch operation, much faster than per-item)
        print(f"    Tokenizing {len(texts)} texts...", end=" ", flush=True)
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids      = encodings["input_ids"]       # (N, max_length)
        self.attention_mask = encodings["attention_mask"]  # (N, max_length)
        print("done.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ── Image ──────────────────────────────────────────────────
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)

        # ── Text (pre-tokenized) ───────────────────────────────────
        input_ids      = self.input_ids[idx]       # (max_length,)
        attention_mask = self.attention_mask[idx]  # (max_length,)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, input_ids, attention_mask, label

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        split: str = None,
        tokenizer: DistilBertTokenizer = None,
        image_transform=None,
        max_length: int = MAX_TOKEN_LENGTH,
    ):
        """
        Create dataset from multimodal_dataset.csv.

        Args:
            csv_path:         Path to multimodal_dataset.csv
            split:            Filter by split ('training' or 'testing'). None = all.
            tokenizer:        Pre-initialized DistilBertTokenizer.
            image_transform:  Torchvision transform pipeline.
            max_length:       Max token length for tokenizer.
        """
        df = pd.read_csv(csv_path)
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)

        tok = tokenizer or DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        return cls(
            image_paths=df["image_path"].tolist(),
            texts=df["text"].tolist(),
            labels=df["severity_encoded"].tolist(),
            tokenizer=tok,
            image_transform=image_transform,
            max_length=max_length,
        )

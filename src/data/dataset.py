"""
PyTorch Dataset for car damage severity classification.

Supports both image-only and multimodal (image + text) modes.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Label mapping
LABEL_MAP = {"minor": 0, "moderate": 1, "severe": 2}
LABEL_NAMES = ["minor", "moderate", "severe"]
NUM_CLASSES = 3

# ImageNet normalization (required for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """Training transforms with aggressive data augmentation for small datasets."""
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
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Cutout augmentation
    ])


def get_val_transforms(image_size: int = 224):
    """Validation/test transforms — no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CarDamageDataset(Dataset):
    """
    Dataset for car damage severity classification.

    Can be initialized from:
    1. A directory structure: root/{01-minor, 02-moderate, 03-severe}/*.jpg
    2. A list of (image_path, label) tuples
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
    ):
        assert len(image_paths) == len(labels), "Paths and labels must have same length"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_val_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    @classmethod
    def from_directory(cls, root_dir: str | Path, transform=None):
        """
        Create dataset from directory structure:
        root_dir/{01-minor, 02-moderate, 03-severe}/*.jpg
        """
        root = Path(root_dir)
        image_paths = []
        labels = []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue

            # Extract label from folder name (e.g., "01-minor" -> "minor")
            label_name = class_dir.name.split("-", 1)[-1] if "-" in class_dir.name else class_dir.name
            label_name = label_name.lower()

            if label_name not in LABEL_MAP:
                print(f"Warning: Skipping unknown class folder '{class_dir.name}'")
                continue

            label = LABEL_MAP[label_name]

            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    image_paths.append(str(img_path))
                    labels.append(label)

        return cls(image_paths, labels, transform)

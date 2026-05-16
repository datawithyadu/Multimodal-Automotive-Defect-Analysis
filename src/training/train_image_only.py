"""
Phase 1: Image-only baseline training with Stratified 5-Fold Cross-Validation.

Trains EfficientNet-B0 on the car damage severity dataset using only images.
Reports mean ± std of accuracy and macro F1 across all folds.

Usage:
    conda run -n genai python src/training/train_image_only.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.dataset import (
    CarDamageDataset,
    get_train_transforms,
    get_val_transforms,
    LABEL_NAMES,
    NUM_CLASSES,
)
from models.image_classifier import EfficientNetClassifier

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_ROOT / "training"
TEST_DIR = DATA_ROOT / "Validation"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
NUM_FOLDS = 5
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4   # Lower LR for fine-tuning (was 1e-3)
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
PATIENCE = 7  # Early stopping patience (increased from 5)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model. Returns dict with loss, accuracy, f1, predictions, labels."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "predictions": all_preds,
        "labels": all_labels,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }


def train_fold(
    fold_idx: int,
    train_dataset: CarDamageDataset,
    train_indices: list,
    val_indices: list,
) -> dict:
    """Train and evaluate one fold. Returns metrics dict."""
    print(f"\n{'='*50}")
    print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"{'='*50}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples:   {len(val_indices)}")

    # Create data subsets with appropriate transforms
    train_subset_paths = [train_dataset.image_paths[i] for i in train_indices]
    train_subset_labels = [train_dataset.labels[i] for i in train_indices]
    val_subset_paths = [train_dataset.image_paths[i] for i in val_indices]
    val_subset_labels = [train_dataset.labels[i] for i in val_indices]

    train_ds = CarDamageDataset(train_subset_paths, train_subset_labels, get_train_transforms(IMAGE_SIZE))
    val_ds = CarDamageDataset(val_subset_paths, val_subset_labels, get_val_transforms(IMAGE_SIZE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model
    model = EfficientNetClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=0.4,       # Slightly higher dropout for regularization
        freeze_backbone=True,
        unfreeze_last_n_blocks=3,  # Unfreeze more layers (was 2)
    ).to(DEVICE)

    if fold_idx == 0:
        trainable = model.count_trainable_params()
        total = model.count_total_params()
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6,
    )

    # Training loop with early stopping
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping check
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Load best model and get final metrics
    model.load_state_dict(best_model_state)
    final_metrics = evaluate(model, val_loader, criterion, DEVICE)

    print(f"\n  Best Val F1: {final_metrics['f1_macro']:.4f}")
    print(f"  Best Val Acc: {final_metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  {final_metrics['confusion_matrix']}")

    # Save best model for this fold
    model_path = RESULTS_DIR / f"image_only_fold{fold_idx+1}.pt"
    torch.save(best_model_state, model_path)

    return {
        "fold": fold_idx + 1,
        "best_val_accuracy": final_metrics["accuracy"],
        "best_val_f1_macro": final_metrics["f1_macro"],
        "best_val_loss": final_metrics["loss"],
        "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
        "model_path": str(model_path),
    }


def evaluate_held_out_test(best_fold_model_path: str) -> dict:
    """Evaluate the best model on the held-out test set."""
    print(f"\n{'='*50}")
    print(f"  HELD-OUT TEST SET EVALUATION")
    print(f"{'='*50}")

    # Load test data
    test_dataset = CarDamageDataset.from_directory(TEST_DIR, get_val_transforms(IMAGE_SIZE))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_dataset)}")

    # Load best model
    model = EfficientNetClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(best_fold_model_path, map_location=DEVICE, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(metrics["labels"], metrics["predictions"], target_names=LABEL_NAMES))
    print(f"  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")

    return {
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_loss": metrics["loss"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": classification_report(
            metrics["labels"], metrics["predictions"],
            target_names=LABEL_NAMES, output_dict=True,
        ),
    }


def main():
    print("=" * 60)
    print("  Phase 1: Image-Only Baseline (EfficientNet-B0)")
    print("  Stratified 5-Fold Cross-Validation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Train dir: {TRAIN_DIR}")
    print(f"  Test dir: {TEST_DIR}")
    print()

    start_time = time.time()

    # Load training data (all train images — we'll split with KFold)
    full_train_dataset = CarDamageDataset.from_directory(TRAIN_DIR, transform=None)
    print(f"Total training images: {len(full_train_dataset)}")

    labels_array = np.array(full_train_dataset.labels)

    # Print class distribution
    for i, name in enumerate(LABEL_NAMES):
        count = (labels_array == i).sum()
        print(f"  {name}: {count} ({100*count/len(labels_array):.1f}%)")
    print()

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(labels_array, labels_array)):
        result = train_fold(fold_idx, full_train_dataset, train_indices.tolist(), val_indices.tolist())
        fold_results.append(result)

    # Summary across folds
    accuracies = [r["best_val_accuracy"] for r in fold_results]
    f1_scores = [r["best_val_f1_macro"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['best_val_accuracy']:.4f}, F1={r['best_val_f1_macro']:.4f}")
    print(f"  {'─'*40}")
    print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Mean F1 Macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    # Evaluate on held-out test set using the best fold's model
    best_fold_idx = np.argmax(f1_scores)
    best_model_path = fold_results[best_fold_idx]["model_path"]
    print(f"\n  Best fold: {best_fold_idx + 1} (F1={f1_scores[best_fold_idx]:.4f})")

    test_results = evaluate_held_out_test(best_model_path)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Save all results
    all_results = {
        "model": "EfficientNet-B0 (image-only)",
        "config": {
            "num_folds": NUM_FOLDS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "patience": PATIENCE,
        },
        "cv_results": {
            "fold_results": fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_f1_macro": float(np.mean(f1_scores)),
            "std_f1_macro": float(np.std(f1_scores)),
        },
        "test_results": test_results,
        "elapsed_seconds": elapsed,
    }

    results_file = RESULTS_DIR / "phase1_image_only_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    main()

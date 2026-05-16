"""
Phase 2: Text-only baseline (leakage test) with Stratified Group 5-Fold CV.

Trains DistilBERT on the synthetic text descriptions only (no images)
to verify that text does NOT encode severity information.

TARGET: ~33% accuracy (random chance for 3 classes).
If accuracy is significantly higher, it indicates label leakage.

Usage:
    conda run -n genai python src/training/train_text_only.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from transformers import DistilBertTokenizer
from tqdm import tqdm

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.text_dataset import TextDamageDataset, LABEL_NAMES, NUM_CLASSES
from models.text_classifier import DistilBertClassifier

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
CSV_PATH = DATA_ROOT / "multimodal_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
NUM_FOLDS = 5
NUM_EPOCHS = 20          # Text models converge faster than vision models
BATCH_SIZE = 32
LEARNING_RATE = 2e-5     # Standard BERT fine-tuning learning rate
WEIGHT_DECAY = 1e-2
MAX_TOKEN_LENGTH = 64    # Our texts are ~30 words ≈ ~40 tokens
PATIENCE = 5             # Early stopping patience
WARMUP_RATIO = 0.1       # 10% of total steps for linear warmup

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — standard for transformer fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

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
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
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


def get_linear_warmup_scheduler(optimizer, num_warmup_steps, num_total_steps):
    """Linear warmup then linear decay scheduler."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_total_steps - current_step) /
            float(max(1, num_total_steps - num_warmup_steps)),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_fold(
    fold_idx: int,
    full_dataset: TextDamageDataset,
    train_indices: list,
    val_indices: list,
    tokenizer: DistilBertTokenizer,
) -> dict:
    """Train and evaluate one fold. Returns metrics dict."""
    print(f"\n{'='*50}")
    print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"{'='*50}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples:   {len(val_indices)}")

    # Create data subsets
    train_texts = [full_dataset.texts[i] for i in train_indices]
    train_labels = [full_dataset.labels[i] for i in train_indices]
    train_paths = [full_dataset.image_paths[i] for i in train_indices]

    val_texts = [full_dataset.texts[i] for i in val_indices]
    val_labels = [full_dataset.labels[i] for i in val_indices]
    val_paths = [full_dataset.image_paths[i] for i in val_indices]

    train_ds = TextDamageDataset(train_texts, train_labels, train_paths, tokenizer, MAX_TOKEN_LENGTH)
    val_ds = TextDamageDataset(val_texts, val_labels, val_paths, tokenizer, MAX_TOKEN_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model
    model = DistilBertClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=0.3,
        freeze_backbone=True,
        unfreeze_last_n_layers=2,
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

    # Linear warmup scheduler
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # Training loop with early stopping
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"LR: {current_lr:.2e}"
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
    model_path = RESULTS_DIR / f"text_only_fold{fold_idx+1}.pt"
    torch.save(best_model_state, model_path)

    return {
        "fold": fold_idx + 1,
        "best_val_accuracy": float(final_metrics["accuracy"]),
        "best_val_f1_macro": float(final_metrics["f1_macro"]),
        "best_val_loss": float(final_metrics["loss"]),
        "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
        "model_path": str(model_path),
    }


def evaluate_held_out_test(
    best_fold_model_path: str,
    tokenizer: DistilBertTokenizer,
) -> dict:
    """Evaluate the best model on the held-out test set."""
    print(f"\n{'='*50}")
    print(f"  HELD-OUT TEST SET EVALUATION")
    print(f"{'='*50}")

    # Load test data
    test_dataset = TextDamageDataset.from_csv(
        CSV_PATH, split="testing", tokenizer=tokenizer, max_length=MAX_TOKEN_LENGTH,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_dataset)}")

    # Load best model
    model = DistilBertClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(best_fold_model_path, map_location=DEVICE, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(metrics["labels"], metrics["predictions"], target_names=LABEL_NAMES))
    print(f"  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")

    # Leakage assessment
    print(f"\n  {'='*40}")
    print(f"  LEAKAGE ASSESSMENT")
    print(f"  {'='*40}")
    test_acc = metrics["accuracy"]
    if test_acc <= 0.40:
        print(f"  [PASS] Accuracy {test_acc:.1%} is near random chance (33.3%)")
        print(f"  -> Text descriptions are severity-neutral. Safe to proceed to fusion.")
    elif test_acc <= 0.50:
        print(f"  [WARNING] Accuracy {test_acc:.1%} is above chance.")
        print(f"  -> Inspect confusion matrix for systematic patterns.")
        print(f"  -> May indicate mild leakage in text generation pipeline.")
    else:
        print(f"  [FAIL] Accuracy {test_acc:.1%} significantly exceeds chance.")
        print(f"  -> LEAKAGE DETECTED. Text encodes severity information.")
        print(f"  -> Investigate what signal BERT is learning from text.")

    return {
        "test_accuracy": float(metrics["accuracy"]),
        "test_f1_macro": float(metrics["f1_macro"]),
        "test_loss": float(metrics["loss"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": classification_report(
            metrics["labels"], metrics["predictions"],
            target_names=LABEL_NAMES, output_dict=True,
        ),
    }


def main():
    print("=" * 60)
    print("  Phase 2: Text-Only Baseline (DistilBERT) — Leakage Test")
    print("  Group-Aware 5-Fold Cross-Validation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  CSV: {CSV_PATH}")
    print(f"  Target: ~33% accuracy (random chance = no leakage)")
    print()

    start_time = time.time()

    # Initialize tokenizer once (shared across all folds)
    print("  Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load training data
    full_train_dataset = TextDamageDataset.from_csv(
        CSV_PATH, split="training", tokenizer=tokenizer, max_length=MAX_TOKEN_LENGTH,
    )
    print(f"  Total training text samples: {len(full_train_dataset)}")
    print(f"  Unique images: {len(set(full_train_dataset.image_paths))}")

    labels_array = np.array(full_train_dataset.labels)
    groups_array = np.array(full_train_dataset.image_paths)

    # Print class distribution
    for i, name in enumerate(LABEL_NAMES):
        count = (labels_array == i).sum()
        print(f"  {name}: {count} ({100*count/len(labels_array):.1f}%)")
    print()

    # Group K-Fold — all 3 text variants of same image stay together
    gkf = GroupKFold(n_splits=NUM_FOLDS)

    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(
        gkf.split(labels_array, labels_array, groups=groups_array)
    ):
        # Verify no group leakage
        train_groups = set(groups_array[train_indices])
        val_groups = set(groups_array[val_indices])
        overlap = train_groups & val_groups
        assert len(overlap) == 0, f"Group leakage detected! {len(overlap)} images in both train and val"

        result = train_fold(fold_idx, full_train_dataset, train_indices.tolist(), val_indices.tolist(), tokenizer)
        fold_results.append(result)

    # Summary across folds
    accuracies = [r["best_val_accuracy"] for r in fold_results]
    f1_scores_list = [r["best_val_f1_macro"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['best_val_accuracy']:.4f}, F1={r['best_val_f1_macro']:.4f}")
    print(f"  {'-'*40}")
    print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Mean F1 Macro: {np.mean(f1_scores_list):.4f} ± {np.std(f1_scores_list):.4f}")

    # Evaluate on held-out test set using the best fold's model
    best_fold_idx = np.argmax(f1_scores_list)
    best_model_path = fold_results[best_fold_idx]["model_path"]
    print(f"\n  Best fold: {best_fold_idx + 1} (F1={f1_scores_list[best_fold_idx]:.4f})")

    test_results = evaluate_held_out_test(best_model_path, tokenizer)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Save all results
    all_results = {
        "model": "DistilBERT (text-only) — leakage test",
        "purpose": "Verify synthetic text does NOT encode severity. Target: ~33% accuracy.",
        "config": {
            "num_folds": NUM_FOLDS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_token_length": MAX_TOKEN_LENGTH,
            "patience": PATIENCE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "freeze_strategy": "freeze embeddings + first 4 layers, fine-tune last 2 + classifier",
            "group_splitting": "GroupKFold by image_path (all 3 variants stay together)",
        },
        "cv_results": {
            "fold_results": fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_f1_macro": float(np.mean(f1_scores_list)),
            "std_f1_macro": float(np.std(f1_scores_list)),
        },
        "test_results": test_results,
        "elapsed_seconds": elapsed,
    }

    results_file = RESULTS_DIR / "phase2_text_only_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    main()

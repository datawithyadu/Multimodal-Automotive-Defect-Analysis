"""
Phase 3: Concatenation Fusion Baseline with Group-Aware 5-Fold CV.

Trains a model that concatenates EfficientNet-B0 image features (1280-dim)
with DistilBERT text features (768-dim) and classifies through a dense head.

This is the simplest multimodal baseline — no cross-modal attention.
Its purpose is to establish whether ANY multimodal fusion beats both
unimodal baselines:
    Phase 1: Image-only   → 70.2% test accuracy, F1=68.4%
    Phase 2: Text-only    → 65.3% test accuracy, F1=65.8%
    Phase 3: Concat (this) → ???

Usage:
    conda run -n genai python src/training/train_concat_fusion.py
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

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.multimodal_dataset import (
    MultimodalDamageDataset,
    get_train_transforms,
    get_val_transforms,
    LABEL_NAMES,
    NUM_CLASSES,
    MAX_TOKEN_LENGTH,
)
from models.concat_fusion import ConcatFusionClassifier

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT / "data"
CSV_PATH     = DATA_ROOT / "multimodal_dataset.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
NUM_FOLDS      = 5
NUM_EPOCHS     = 25
BATCH_SIZE     = 16    # Reduced from 32 — two large models in memory simultaneously
PATIENCE       = 7
IMAGE_SIZE     = 224
MAX_TOKEN_LEN  = MAX_TOKEN_LENGTH  # 64

# Two learning rates: lower for pre-trained encoders, higher for fresh classifier head
ENCODER_LR     = 1e-5   # Fine-tuning LR for EfficientNet + DistilBERT layers
CLASSIFIER_LR  = 1e-4   # Higher LR for the new fusion classification head
WEIGHT_DECAY   = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# Training utilities
# -------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, input_ids, attention_mask, labels in loader:
        images         = images.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss    = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
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
    total_loss  = 0.0
    num_batches = 0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in loader:
            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            outputs = model(images, input_ids, attention_mask)
            loss    = criterion(outputs, labels)

            total_loss  += loss.item()
            num_batches += 1

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss":             total_loss / max(num_batches, 1),
        "accuracy":         accuracy_score(all_labels, all_preds),
        "f1_macro":         f1_score(all_labels, all_preds, average="macro"),
        "predictions":      all_preds,
        "labels":           all_labels,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }


def get_warmup_scheduler(optimizer, num_warmup_steps, num_total_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_total_steps - step) /
            float(max(1, num_total_steps - num_warmup_steps)),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------------------
# Fold training
# -------------------------------------------------------------------

def train_fold(
    fold_idx: int,
    full_dataset: MultimodalDamageDataset,
    train_indices: list,
    val_indices: list,
    tokenizer: DistilBertTokenizer,
) -> dict:
    """Train and evaluate one fold. Returns metrics dict."""
    print(f"\n{'='*55}")
    print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"{'='*55}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples:   {len(val_indices)}")

    # Build fold sub-datasets (train gets augmentation, val gets clean transforms)
    def make_subset(indices, transform):
        return MultimodalDamageDataset(
            image_paths=[full_dataset.image_paths[i] for i in indices],
            texts=[full_dataset.texts[i] for i in indices],
            labels=[full_dataset.labels[i] for i in indices],
            tokenizer=tokenizer,
            image_transform=transform,
            max_length=MAX_TOKEN_LEN,
        )

    train_ds = make_subset(train_indices, get_train_transforms(IMAGE_SIZE))
    val_ds   = make_subset(val_indices,   get_val_transforms(IMAGE_SIZE))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Initialize model
    model = ConcatFusionClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=0.4,
        freeze_image_backbone=True,
        unfreeze_image_last_n_blocks=3,
        freeze_text_backbone=True,
        unfreeze_text_last_n_layers=2,
    ).to(DEVICE)

    if fold_idx == 0:
        trainable = model.count_trainable_params()
        total     = model.count_total_params()
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Separate parameter groups — different LRs for encoders vs classifier
    encoder_params    = list(model.image_encoder.parameters()) + \
                        list(model.text_encoder.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder_params    if p.requires_grad], "lr": ENCODER_LR},
        {"params": [p for p in classifier_params if p.requires_grad], "lr": CLASSIFIER_LR},
    ], weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()

    # Training loop with early stopping (monitor val F1)
    best_val_f1    = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss  = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        enc_lr  = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(
            f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"LR: {enc_lr:.1e}/{head_lr:.1e}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1      = val_metrics["f1_macro"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Reload best and get final metrics
    model.load_state_dict(best_model_state)
    final_metrics = evaluate(model, val_loader, criterion, DEVICE)

    print(f"\n  Best Val F1:  {final_metrics['f1_macro']:.4f}")
    print(f"  Best Val Acc: {final_metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix:\n  {final_metrics['confusion_matrix']}")

    model_path = RESULTS_DIR / f"concat_fusion_fold{fold_idx+1}.pt"
    torch.save(best_model_state, model_path)

    return {
        "fold":               fold_idx + 1,
        "best_val_accuracy":  float(final_metrics["accuracy"]),
        "best_val_f1_macro":  float(final_metrics["f1_macro"]),
        "best_val_loss":      float(final_metrics["loss"]),
        "confusion_matrix":   final_metrics["confusion_matrix"].tolist(),
        "model_path":         str(model_path),
    }


# -------------------------------------------------------------------
# Held-out test evaluation
# -------------------------------------------------------------------

def evaluate_held_out_test(
    best_model_path: str,
    tokenizer: DistilBertTokenizer,
) -> dict:
    """Evaluate best model on the held-out test set."""
    print(f"\n{'='*55}")
    print(f"  HELD-OUT TEST SET EVALUATION")
    print(f"{'='*55}")

    test_ds = MultimodalDamageDataset.from_csv(
        CSV_PATH, split="testing",
        tokenizer=tokenizer,
        image_transform=get_val_transforms(IMAGE_SIZE),
        max_length=MAX_TOKEN_LEN,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_ds)}")

    model = ConcatFusionClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))

    criterion = nn.CrossEntropyLoss()
    metrics   = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n  Test Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Test F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(metrics["labels"], metrics["predictions"], target_names=LABEL_NAMES))
    print(f"  Confusion Matrix:\n  {metrics['confusion_matrix']}")

    # Comparison with unimodal baselines
    print(f"\n  {'='*45}")
    print(f"  COMPARISON VS UNIMODAL BASELINES")
    print(f"  {'='*45}")
    print(f"  Image-only (Phase 1): Acc=70.2%  F1=68.4%")
    print(f"  Text-only  (Phase 2): Acc=65.3%  F1=65.8%")
    print(f"  Concat fusion:        Acc={metrics['accuracy']*100:.1f}%  F1={metrics['f1_macro']*100:.1f}%")

    baseline_f1 = max(0.684, 0.658)  # best unimodal
    if metrics["f1_macro"] > baseline_f1:
        gain = (metrics["f1_macro"] - baseline_f1) * 100
        print(f"\n  [PASS] Fusion beats best unimodal by +{gain:.1f}% F1")
        print(f"  -> Multimodal fusion provides real benefit.")
    else:
        gap = (baseline_f1 - metrics["f1_macro"]) * 100
        print(f"\n  [NOTE] Concat fusion is {gap:.1f}% F1 below best unimodal.")
        print(f"  -> Simple concatenation insufficient. Cross-attention may do better.")

    return {
        "test_accuracy":         float(metrics["accuracy"]),
        "test_f1_macro":         float(metrics["f1_macro"]),
        "test_loss":             float(metrics["loss"]),
        "confusion_matrix":      metrics["confusion_matrix"].tolist(),
        "classification_report": classification_report(
            metrics["labels"], metrics["predictions"],
            target_names=LABEL_NAMES, output_dict=True,
        ),
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Phase 3: Concatenation Fusion (Image + Text Baseline)")
    print("  Group-Aware 5-Fold Cross-Validation")
    print("=" * 60)
    print(f"  Device:       {DEVICE}")
    print(f"  CSV:          {CSV_PATH}")
    print(f"  Batch size:   {BATCH_SIZE} (reduced for dual-encoder memory)")
    print(f"  Encoder LR:   {ENCODER_LR}  |  Classifier LR: {CLASSIFIER_LR}")
    print()

    start_time = time.time()

    # Shared tokenizer
    print("  Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load full training data (no transform yet — assigned per fold)
    full_dataset = MultimodalDamageDataset.from_csv(
        CSV_PATH, split="training",
        tokenizer=tokenizer,
        image_transform=get_val_transforms(IMAGE_SIZE),  # placeholder
        max_length=MAX_TOKEN_LEN,
    )
    print(f"  Total training samples: {len(full_dataset)}")
    print(f"  Unique images:          {len(set(full_dataset.image_paths))}")

    labels_array = np.array(full_dataset.labels)
    groups_array = np.array(full_dataset.image_paths)

    for i, name in enumerate(LABEL_NAMES):
        count = (labels_array == i).sum()
        print(f"  {name}: {count} ({100*count/len(labels_array):.1f}%)")
    print()

    # GroupKFold — all 3 text variants of same image stay together
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(labels_array, labels_array, groups=groups_array)
    ):
        # Sanity check: no image group in both train and val
        train_groups = set(groups_array[train_idx])
        val_groups   = set(groups_array[val_idx])
        overlap = train_groups & val_groups
        assert len(overlap) == 0, f"Group leakage detected! {len(overlap)} images in both folds"

        result = train_fold(
            fold_idx, full_dataset,
            train_idx.tolist(), val_idx.tolist(),
            tokenizer,
        )
        fold_results.append(result)

    # CV summary
    accuracies    = [r["best_val_accuracy"] for r in fold_results]
    f1_scores_list = [r["best_val_f1_macro"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['best_val_accuracy']:.4f}, F1={r['best_val_f1_macro']:.4f}")
    print(f"  {'-'*40}")
    print(f"  Mean Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"  Mean F1 Macro: {np.mean(f1_scores_list):.4f} +/- {np.std(f1_scores_list):.4f}")

    # Held-out test evaluation with best fold
    best_fold_idx  = np.argmax(f1_scores_list)
    best_model_path = fold_results[best_fold_idx]["model_path"]
    print(f"\n  Best fold: {best_fold_idx + 1} (F1={f1_scores_list[best_fold_idx]:.4f})")

    test_results = evaluate_held_out_test(best_model_path, tokenizer)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Save all results
    all_results = {
        "model":   "Concatenation Fusion (EfficientNet-B0 + DistilBERT)",
        "purpose": "Multimodal baseline — concatenate image (1280) + text (768) features",
        "config": {
            "num_folds":             NUM_FOLDS,
            "num_epochs":            NUM_EPOCHS,
            "batch_size":            BATCH_SIZE,
            "encoder_lr":            ENCODER_LR,
            "classifier_lr":         CLASSIFIER_LR,
            "image_size":            IMAGE_SIZE,
            "max_token_length":      MAX_TOKEN_LEN,
            "patience":              PATIENCE,
            "image_unfreeze_blocks": 3,
            "text_unfreeze_layers":  2,
            "group_splitting":       "GroupKFold by image_path",
        },
        "cv_results": {
            "fold_results":  fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy":  float(np.std(accuracies)),
            "mean_f1_macro": float(np.mean(f1_scores_list)),
            "std_f1_macro":  float(np.std(f1_scores_list)),
        },
        "test_results":   test_results,
        "elapsed_seconds": elapsed,
        "unimodal_baselines": {
            "image_only_test_f1":  0.684,
            "text_only_test_f1":   0.658,
        },
    }

    results_file = RESULTS_DIR / "phase3_concat_fusion_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    main()

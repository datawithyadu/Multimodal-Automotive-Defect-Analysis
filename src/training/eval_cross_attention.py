"""
Phase 4: Run held-out test evaluation only (after training completed).
Uses the best fold model saved from cross-attention training (v2).

Best fold: 4  (Val F1=0.742, from phase4_cross_attention_v2_results.json)
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizer

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.multimodal_dataset import (
    MultimodalDamageDataset, get_val_transforms, LABEL_NAMES, NUM_CLASSES, MAX_TOKEN_LENGTH,
)
from models.cross_attention_fusion import CrossAttentionFusionClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH     = PROJECT_ROOT / "data" / "multimodal_dataset.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 16

# Cross-attention v2 hyperparameters (must match training config exactly)
D_K       = 512
NUM_HEADS = 8


def main(fold: int):
    print("=" * 60)
    print("  Phase 4: Held-Out Test Evaluation — Cross-Attention Fusion")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Fold   : {fold}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    test_ds = MultimodalDamageDataset.from_csv(
        CSV_PATH, split="testing",
        tokenizer=tokenizer,
        image_transform=get_val_transforms(224),
        max_length=MAX_TOKEN_LENGTH,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_ds)}")

    best_model_path = RESULTS_DIR / f"cross_attention_fusion_fold{fold}.pt"
    print(f"  Loading weights: {best_model_path}")

    model = CrossAttentionFusionClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=0.4,
        d_k=D_K,
        num_heads=NUM_HEADS,
        modality_dropout_p=0.0,          # no dropout at inference
        freeze_image_backbone=True,
        unfreeze_image_last_n_blocks=3,
        freeze_text_backbone=True,
        unfreeze_text_last_n_layers=2,
    ).to(DEVICE)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in test_loader:
            images         = images.to(DEVICE)
            input_ids      = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels         = labels.to(DEVICE)
            outputs        = model(images, input_ids, attention_mask)
            loss           = criterion(outputs, labels)
            total_loss    += loss.item()
            num_batches   += 1
            preds          = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average="macro")
    cm       = confusion_matrix(all_labels, all_preds)
    report   = classification_report(all_labels, all_preds, target_names=LABEL_NAMES)
    report_d = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True)

    print(f"\n  Test Accuracy:   {test_acc:.4f}")
    print(f"  Test F1 (Macro): {test_f1:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n  {cm}")

    print(f"\n  {'='*45}")
    print(f"  COMPARISON VS BASELINES")
    print(f"  {'='*45}")
    print(f"  Image-only  (Phase 1): Acc=70.2%  F1=68.4%")
    print(f"  Text-only   (Phase 2): Acc=65.3%  F1=65.8%")
    print(f"  Concat      (Phase 3): Acc=73.3%  F1=73.2%")
    print(f"  Cross-Attn  (Phase 4): Acc={test_acc*100:.1f}%  F1={test_f1*100:.1f}%")

    best_baseline_f1 = max(0.684, 0.658)
    if test_f1 > best_baseline_f1:
        print(f"\n  [PASS] Cross-attention beats best unimodal by +{(test_f1-best_baseline_f1)*100:.1f}% F1")
    else:
        print(f"\n  [NOTE] Cross-attention is {(best_baseline_f1-test_f1)*100:.1f}% F1 below best unimodal")

    # Save results
    cv_fold_results = [
        {"fold": 1, "best_val_accuracy": 0.7353, "best_val_f1_macro": 0.7329},
        {"fold": 2, "best_val_accuracy": 0.7052, "best_val_f1_macro": 0.7012},
        {"fold": 3, "best_val_accuracy": 0.6880, "best_val_f1_macro": 0.6870},
        {"fold": 4, "best_val_accuracy": 0.7432, "best_val_f1_macro": 0.7420},
        {"fold": 5, "best_val_accuracy": 0.6888, "best_val_f1_macro": 0.6911},
    ]
    accuracies = [r["best_val_accuracy"] for r in cv_fold_results]
    f1s        = [r["best_val_f1_macro"]  for r in cv_fold_results]

    all_results = {
        "model":   "Cross-Attention Fusion (EfficientNet-B0 + DistilBERT) — v2",
        "purpose": "Multimodal fusion — image spatial regions attend to text tokens",
        "config": {
            "num_folds": 5, "num_epochs": 25, "batch_size": 16,
            "encoder_lr": 1e-5, "classifier_lr": 1e-4, "image_size": 224,
            "max_token_length": 64, "patience": 7,
            "d_k": D_K, "num_heads": NUM_HEADS,
            "modality_dropout_p": 0.1, "l2_normalization": True,
            "group_splitting": "GroupKFold by image_path",
        },
        "cv_results": {
            "fold_results":  cv_fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy":  float(np.std(accuracies)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro":  float(np.std(f1s)),
        },
        "test_results": {
            "best_fold":             fold,
            "test_accuracy":         float(test_acc),
            "test_f1_macro":         float(test_f1),
            "test_loss":             float(total_loss / max(num_batches, 1)),
            "confusion_matrix":      cm.tolist(),
            "classification_report": report_d,
        },
        "unimodal_baselines": {
            "image_only_test_accuracy": 0.702,
            "image_only_test_f1":       0.684,
            "text_only_test_accuracy":  0.653,
            "text_only_test_f1":        0.658,
        },
    }

    results_file = RESULTS_DIR / "phase4_cross_attention_eval.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Attention Held-Out Test Evaluation")
    parser.add_argument("--fold", type=int, default=4,
                        help="Which fold's weights to load (default: 4, best v2 fold)")
    args = parser.parse_args()
    main(args.fold)

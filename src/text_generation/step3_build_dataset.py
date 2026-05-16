"""
Step 3: Build the final dataset CSV from generated descriptions.

Converts the JSON output from Step 2 into a clean CSV ready for model training.
Each row = one image-text pair (multiple rows per image if variants exist).

Output: data/multimodal_dataset.csv
"""

import json
import random
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "synthetic_descriptions.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "multimodal_dataset.csv"

# Label encoding
LABEL_MAP = {"minor": 0, "moderate": 1, "severe": 2}


def main():
    print("=" * 60)
    print("Step 3: Building final multimodal dataset CSV")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run Steps 1 and 2 first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} image records.")

    # Build rows: one row per (image, text_variant) pair
    rows = []
    for record in records:
        image_path = record["image_path"]
        split = record["split"]
        severity = record["severity_label"]
        label_encoded = LABEL_MAP.get(severity, -1)
        descriptions = record.get("descriptions", [])

        if not descriptions:
            # If no descriptions were generated, add a placeholder
            descriptions = ["Vehicle damage observed on inspection."]

        for variant_idx, text in enumerate(descriptions):
            rows.append({
                "image_path": image_path,
                "text": text,
                "variant_idx": variant_idx,
                "severity_label": severity,
                "severity_encoded": label_encoded,
                "split": split,
            })

    df = pd.DataFrame(rows)

    # Shuffle within each split (important for training)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Print summary
    print()
    print("Dataset Summary:")
    print("-" * 40)
    print(f"Total image-text pairs: {len(df)}")
    print(f"Unique images: {df['image_path'].nunique()}")
    print()
    print("By split:")
    print(df.groupby("split").size().to_string())
    print()
    print("By severity label:")
    print(df.groupby("severity_label").size().to_string())
    print()
    print("By split × severity:")
    print(df.groupby(["split", "severity_label"]).size().unstack(fill_value=0).to_string())
    print()
    print(f"Saved to: {OUTPUT_FILE}")

    # Sample preview
    print()
    print("Sample entries:")
    print("-" * 40)
    for _, row in df.head(5).iterrows():
        img_name = Path(row["image_path"]).name
        print(f"  [{row['severity_label']}] {img_name}")
        print(f"    Text: {row['text']}")
        print()


if __name__ == "__main__":
    main()

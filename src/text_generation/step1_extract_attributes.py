"""
Step 1: Extract structured attributes from car damage images using GPT-4o vision.

This script sends each image to GPT-4o and asks it to identify ONLY factual,
observable attributes (component, damage type, location) — NOT severity.

Output: data/image_attributes.json
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(r"C:\D_Folder\Defect seviority dataset\dataset")
OUTPUT_FILE = PROJECT_ROOT / "data" / "image_attributes.json"

# Load API key from .env
load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GPT-4o vision model
MODEL = "gpt-4o"

# Rate limiting: pause between API calls (seconds)
DELAY_BETWEEN_CALLS = 0.5

# -------------------------------------------------------------------
# System prompt for attribute extraction
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are a neutral vehicle inspection technician. 
Your job is to examine a photo of a damaged vehicle and identify ONLY the 
observable, factual attributes of the damage.

RULES:
1. Do NOT assess how severe or serious the damage is.
2. Do NOT use emotional language or subjective judgments.
3. Do NOT describe whether the car is drivable or safe.
4. ONLY describe what you can physically see in the image.

Respond with a JSON object containing exactly these fields:
{
    "component": "the primary vehicle part affected (e.g., front bumper, hood, door)",
    "damage_type": "the type of physical damage visible (e.g., scratch, dent, crack, deformation)",
    "location_on_component": "where on the component the damage is (e.g., lower left, center, upper edge)",
    "material_visible": "what material you can see (e.g., metal, plastic, glass, paint)",
    "secondary_component": "another affected part if visible, or null",
    "secondary_damage_type": "damage type on secondary component, or null"
}

Return ONLY the JSON object, no other text."""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_image_to_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """Get the MIME type for an image based on its extension."""
    ext = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


def extract_attributes(image_path: Path) -> dict:
    """
    Send an image to GPT-4o vision and extract structured damage attributes.

    Returns a dict with the extracted attributes, or an error dict if the call fails.
    """
    base64_image = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}",
                                "detail": "low",  # Use low detail to save tokens/cost
                            },
                        },
                        {
                            "type": "text",
                            "text": "Examine this vehicle image and extract the damage attributes as specified.",
                        },
                    ],
                },
            ],
            max_tokens=300,
            temperature=0.2,  # Low temperature for consistent factual extraction
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            # Remove ```json and ``` markers
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        attributes = json.loads(content)
        return attributes

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_response": content}
    except Exception as e:
        return {"error": str(e)}


def collect_image_paths(data_root: Path) -> list[dict]:
    """
    Walk the dataset directory and collect all image paths with their labels.

    Expected structure: data_root / {train,test} / {01-minor,02-moderate,03-severe} / *.jpg
    """
    records = []
    for split_name in ["training", "testing"]:
        split_dir = data_root / split_name
        if not split_dir.is_dir():
            print(f"Warning: {split_dir} not found, skipping.")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            # Extract clean label from folder name (e.g., "01-minor" -> "minor")
            label = class_dir.name.split("-", 1)[-1] if "-" in class_dir.name else class_dir.name

            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({
                        "image_path": str(img_path),
                        "split": split_name,
                        "severity_label": label,
                        "class_folder": class_dir.name,
                    })

    return records


def main(auto_yes: bool = False):
    print("=" * 60)
    print("Step 1: Extracting damage attributes from images via GPT-4o")
    print("=" * 60)

    # Load existing progress if any (resume support)
    existing_results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_results = {r["image_path"]: r for r in existing_data}
        print(f"Found existing progress: {len(existing_results)} images already processed.")

    # Collect all image paths
    records = collect_image_paths(DATA_ROOT)
    print(f"Total images found: {len(records)}")

    # Filter out already-processed images
    to_process = [r for r in records if r["image_path"] not in existing_results]
    print(f"Images remaining to process: {len(to_process)}")

    if not to_process:
        print("All images already processed! Nothing to do.")
        return

    # Estimate cost
    estimated_cost = len(to_process) * 0.003  # ~$0.003 per image at low detail
    print(f"Estimated API cost: ~${estimated_cost:.2f}")
    print()

    if not auto_yes:
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
    else:
        print("Auto-confirmed with --yes flag.")

    # Process images
    results = list(existing_results.values())
    errors = 0

    for record in tqdm(to_process, desc="Extracting attributes"):
        image_path = Path(record["image_path"])
        attributes = extract_attributes(image_path)

        if "error" in attributes:
            errors += 1
            tqdm.write(f"  Error for {image_path.name}: {attributes['error']}")

        record["attributes"] = attributes
        results.append(record)

        # Save progress after every 10 images
        if len(results) % 10 == 0:
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY_BETWEEN_CALLS)

    # Final save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done! Processed {len(to_process)} images ({errors} errors).")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract damage attributes from images via GPT-4o")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    main(auto_yes=args.yes)

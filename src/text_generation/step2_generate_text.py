"""
Step 2: Generate severity-neutral text descriptions from extracted attributes.

This script reads the structured attributes from Step 1 and generates
controlled, severity-neutral damage descriptions. The text generator
NEVER sees the severity label or the original image.

Output: data/synthetic_descriptions.json
"""

import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocabulary import BANNED_SEVERITY_WORDS, validate_text

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "image_attributes.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "synthetic_descriptions.json"

load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use GPT-4o-mini for text generation (cheaper, sufficient for this task)
MODEL = "gpt-4o-mini"

# Number of paraphrased variants per image
NUM_VARIANTS = 3

# Rate limiting
DELAY_BETWEEN_CALLS = 0.3

# -------------------------------------------------------------------
# System prompt for text generation
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are generating synthetic vehicle damage reports for a 
machine learning dataset. Your goal is to create realistic but SEVERITY-NEUTRAL 
descriptions of vehicle damage.

STRICT RULES:
1. Write as a neutral vehicle inspection report — third person, passive voice.
2. Do NOT assess severity, urgency, or safety implications.
3. Do NOT use emotional language.
4. NEVER use these words or similar: minor, moderate, severe, serious, critical, 
   significant, slight, major, extreme, terrible, destroyed, totaled, wrecked, 
   massive, tiny, huge, devastating, alarming, urgent, dangerous, unsafe.
5. Each description MUST be exactly 2-3 sentences and 25-40 words.
6. Only describe the PHYSICAL STATE of the damage — what you can observe.
7. Vary the writing style: some formal, some casual, some like a technician's note.

Given the vehicle damage attributes, write {num_variants} different descriptions.
Each description should say the same factual content but in a different style.

Return a JSON array of strings, one per variant. Example:
["Description variant 1.", "Description variant 2.", "Description variant 3."]

Return ONLY the JSON array, no other text."""


def sanitize_attributes(attributes: dict) -> dict:
    """
    Remove any potentially leaky information from the extracted attributes.
    This is the sanitization layer between Step 1 and Step 2.
    """
    sanitized = {}

    # Keep only the allowed fields
    allowed_fields = [
        "component", "damage_type", "location_on_component",
        "material_visible",
    ]

    for field in allowed_fields:
        if field in attributes and attributes[field] is not None:
            value = str(attributes[field]).lower().strip()
            # Remove any severity-related words that might have leaked in
            for banned in BANNED_SEVERITY_WORDS:
                value = value.replace(banned, "")
            sanitized[field] = value.strip()

    # Optionally include secondary component (limit to max 1 extra)
    if attributes.get("secondary_component") and random.random() < 0.5:
        sanitized["secondary_component"] = str(attributes["secondary_component"]).lower().strip()

    return sanitized


def generate_descriptions(attributes: dict) -> list[str]:
    """
    Generate severity-neutral text descriptions from sanitized attributes.

    The model only sees the sanitized attributes — never the severity label
    or the original image.
    """
    # Build the attribute prompt
    attr_text = "\n".join(f"- {k}: {v}" for k, v in attributes.items() if v)

    prompt = f"Vehicle damage attributes:\n{attr_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(num_variants=NUM_VARIANTS),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.8,  # Higher temp for stylistic diversity
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        descriptions = json.loads(content)

        # Validate each description for leakage
        validated = []
        for desc in descriptions:
            is_valid, banned_found = validate_text(desc)
            if is_valid:
                validated.append(desc)
            else:
                # Re-clean by removing banned words
                cleaned = desc
                for word in banned_found:
                    cleaned = cleaned.replace(word, "").replace(word.title(), "")
                # Remove double spaces
                cleaned = " ".join(cleaned.split())
                validated.append(cleaned)

        return validated

    except json.JSONDecodeError:
        return [f"Vehicle inspection notes damage on the {attributes.get('component', 'vehicle')}."]
    except Exception as e:
        return [f"Error: {e}"]


def add_controlled_noise(text: str) -> str:
    """
    Add realistic noise to a description to simulate real user input.
    Applied randomly to some variants.
    """
    noise_options = [
        # Filler phrases (prepend)
        lambda t: random.choice([
            "I think ", "Not sure but ", "I noticed that ", "It looks like ",
            "Upon inspection, ", "Basically, ",
        ]) + t[0].lower() + t[1:],

        # Typo injection (swap two adjacent chars)
        lambda t: _inject_typo(t),

        # Casual tone (remove periods, add informal endings)
        lambda t: t.rstrip(".") + random.choice(["", " tbh", " basically", ""]),

        # No change (keep as is)
        lambda t: t,
    ]

    # Apply 0-1 noise transformations
    if random.random() < 0.4:
        noise_fn = random.choice(noise_options)
        text = noise_fn(text)

    return text


def _inject_typo(text: str) -> str:
    """Randomly swap two adjacent characters in one word."""
    words = text.split()
    if len(words) < 3:
        return text
    # Pick a random word (not first or last)
    idx = random.randint(1, len(words) - 2)
    word = words[idx]
    if len(word) > 3:
        pos = random.randint(1, len(word) - 2)
        word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        words[idx] = word
    return " ".join(words)


def main():
    print("=" * 60)
    print("Step 2: Generating severity-neutral text descriptions")
    print("=" * 60)

    # Load attributes from Step 1
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run Step 1 first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} image records.")

    # Load existing progress
    existing_results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_results = {r["image_path"]: r for r in existing_data}
        print(f"Found existing progress: {len(existing_results)} images already processed.")

    # Filter records
    to_process = [r for r in records if r["image_path"] not in existing_results]
    # Skip records that had errors in Step 1
    to_process = [r for r in to_process if "error" not in r.get("attributes", {})]

    print(f"Images remaining to process: {len(to_process)}")

    if not to_process:
        print("All images already processed!")
        return

    results = list(existing_results.values())
    errors = 0

    for record in tqdm(to_process, desc="Generating descriptions"):
        # Sanitize attributes (leakage firewall)
        sanitized_attrs = sanitize_attributes(record.get("attributes", {}))

        if not sanitized_attrs:
            errors += 1
            continue

        # Generate descriptions (Step 2 model never sees severity label or image)
        descriptions = generate_descriptions(sanitized_attrs)

        # Apply controlled noise to simulate real user input
        noisy_descriptions = [add_controlled_noise(desc) for desc in descriptions]

        result = {
            "image_path": record["image_path"],
            "split": record["split"],
            "severity_label": record["severity_label"],
            "sanitized_attributes": sanitized_attrs,
            "descriptions": noisy_descriptions,
        }
        results.append(result)

        # Save progress every 20 records
        if len(results) % 20 == 0:
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY_BETWEEN_CALLS)

    # Final save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary statistics
    total_descriptions = sum(len(r.get("descriptions", [])) for r in results)
    print()
    print(f"Done! Generated {total_descriptions} descriptions for {len(results)} images.")
    print(f"  Variants per image: ~{NUM_VARIANTS}")
    print(f"  Errors: {errors}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

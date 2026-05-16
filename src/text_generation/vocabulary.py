"""
Controlled per-attribute vocabulary pools for synthetic text generation.

Organized by ATTRIBUTE TYPE (not severity) to prevent label leakage.
The same words can appear across any severity level — the model must
rely on the image to determine how bad the damage is.
"""

# What type of damage is visible
DAMAGE_TYPES = [
    "scratch", "scuff", "abrasion", "dent", "ding",
    "crack", "fracture", "split", "puncture", "hole",
    "deformation", "buckling", "warping", "crumpling",
    "chipping", "peeling", "fading", "discoloration",
    "shatter", "breakage",
]

# Which vehicle component is affected
COMPONENTS = [
    "front bumper", "rear bumper", "hood", "trunk",
    "front door", "rear door", "driver door", "passenger door",
    "front fender", "rear fender", "side panel",
    "windshield", "rear window", "side window", "side mirror",
    "headlight", "taillight", "fog light", "turn signal",
    "roof", "pillar", "rocker panel", "wheel well",
    "grille", "radiator support", "quarter panel",
]

# How much area is affected (spatial extent)
EXTENT = [
    "localized", "small area", "limited section",
    "partial", "moderate area", "section of",
    "spanning", "full width", "across the entire",
    "multiple areas", "widespread",
]

# Observable physical state of the material
MATERIAL_STATE = [
    "intact structural form", "surface disruption",
    "visible surface marks", "paint transfer present",
    "material separation", "structural compromise",
    "layer exposure", "substrate visible",
    "surface continuity broken", "finish damage",
    "component displacement", "alignment shift",
]

# Location descriptors (where on the component)
LOCATION = [
    "upper", "lower", "center", "left side", "right side",
    "corner", "edge", "along the seam", "near the joint",
    "front-facing surface", "rear-facing surface",
]

# Banned words — these MUST NOT appear in generated text
# They directly encode severity and will cause label leakage
BANNED_SEVERITY_WORDS = [
    # Direct severity terms
    "minor", "moderate", "severe", "mild", "serious",
    "critical", "significant", "slight", "light", "heavy",
    "major", "extreme", "catastrophic", "devastating",
    "minimal", "negligible", "substantial", "considerable",
    # Emotional / urgency terms
    "terrible", "horrible", "awful", "devastating",
    "frightening", "scary", "alarming", "urgent",
    "emergency", "dangerous", "unsafe", "totaled",
    "destroyed", "ruined", "wrecked", "mangled",
    # Subjective assessment terms
    "barely noticeable", "very bad", "not too bad",
    "looks fine", "needs immediate", "write-off",
    "cosmetic only", "just a small", "huge",
    "tiny", "massive", "enormous",
]


def validate_text(text: str) -> tuple[bool, list[str]]:
    """
    Check if generated text contains any banned severity words.

    Returns:
        (is_valid, list_of_found_banned_words)
    """
    text_lower = text.lower()
    found = [word for word in BANNED_SEVERITY_WORDS if word in text_lower]
    return len(found) == 0, found

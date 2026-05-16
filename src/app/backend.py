"""
Flask backend API for multimodal defect severity prediction.

Loads fusion models AND unimodal fallback models so users can
submit image-only, text-only, or both for prediction.
Includes LLM-powered contextual explanations.
"""

import io
import sys
import os
import json
import base64
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import openai

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms
from transformers import DistilBertTokenizer

from src.models.concat_fusion import ConcatFusionClassifier
from src.models.cross_attention_fusion import CrossAttentionFusionClassifier
from src.models.image_classifier import EfficientNetClassifier
from src.models.text_classifier import DistilBertClassifier

# ── Constants ──────────────────────────────────────────────────────
LABEL_NAMES = ["Minor", "Moderate", "Severe"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MAX_TOKEN_LENGTH = 64
IMAGE_SIZE = 224

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# ── Inference transforms (no augmentation) ─────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model loading ─────────────────────────────────────────────────
def load_fusion_model(model_type="concat", fold=1, device="cpu"):
    """Load a trained fusion model from saved weights."""

    if model_type == "concat":
        model = ConcatFusionClassifier(
            num_classes=3,
            dropout_rate=0.4,
            freeze_image_backbone=False,
            unfreeze_image_last_n_blocks=3,
            freeze_text_backbone=False,
            unfreeze_text_last_n_layers=2,
        )
        weight_path = os.path.join(RESULTS_DIR, f"concat_fusion_fold{fold}.pt")
    elif model_type == "cross_attention":
        model = CrossAttentionFusionClassifier(
            num_classes=3,
            dropout_rate=0.4,
            d_k=512,
            num_heads=8,
            modality_dropout_p=0.0,  # No dropout at inference
            freeze_image_backbone=False,
            unfreeze_image_last_n_blocks=3,
            freeze_text_backbone=False,
            unfreeze_text_last_n_layers=2,
        )
        weight_path = os.path.join(RESULTS_DIR, f"cross_attention_fusion_fold{fold}.pt")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"  Loading fusion weights: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_image_model(fold=1, device="cpu"):
    """Load the image-only EfficientNet classifier."""
    model = EfficientNetClassifier(
        num_classes=3,
        dropout_rate=0.3,
        freeze_backbone=False,
        unfreeze_last_n_blocks=3,
    )
    weight_path = os.path.join(RESULTS_DIR, f"image_only_fold{fold}.pt")
    print(f"  Loading image-only weights: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_text_model(fold=1, device="cpu"):
    """Load the text-only DistilBERT classifier."""
    model = DistilBertClassifier(
        num_classes=3,
        dropout_rate=0.3,
        freeze_backbone=False,
        unfreeze_last_n_layers=2,
    )
    weight_path = os.path.join(RESULTS_DIR, f"text_only_fold{fold}.pt")
    print(f"  Loading text-only weights: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── Flask App ─────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Global models, tokenizer, and OpenAI client (loaded on startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
oai_client = None
models = {}      # {"concat": model, "cross_attention": model, "image_only": model, "text_only": model}
tokenizer = None


def generate_explanation(prediction, confidence, probabilities, user_text=None, image_bytes=None):
    """
    Use GPT-4o-mini to generate a context-specific explanation
    based on the prediction results and user inputs.
    """
    if oai_client is None:
        return None

    prob_str = ", ".join(f"{k}: {v}%" for k, v in probabilities.items())

    prompt = (
        f"You are an automotive damage assessment AI assistant. "
        f"A vehicle damage analysis model predicted: {prediction} severity "
        f"with {confidence}% confidence.\n"
        f"Class probabilities: {prob_str}.\n"
    )

    if user_text:
        prompt += f'\nThe user described the damage as: "{user_text}"\n'

    prompt += (
        "\nWrite a 2-3 sentence explanation that is SPECIFIC to this case. "
        "Reference the user's description if provided. "
        "Include a practical recommendation. "
        "Do NOT use markdown formatting. Keep it concise."
    )

    messages = [{"role": "user", "content": []}]

    # If image was provided, include it for visual context
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"},
        })

    messages[0]["content"].append({"type": "text", "text": prompt})

    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM explanation failed: {e}")
        return None

def validate_vehicle_image(image_bytes):
    """
    Use GPT-4o-mini vision to verify the image contains a vehicle or vehicle damage.
    Returns (is_valid, reason).
    """
    if oai_client is None:
        # If no OpenAI key, skip validation
        return True, ""

    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Does this image show a vehicle (car, truck, van, motorcycle, etc.) "
                            "or vehicle damage/parts? Answer ONLY with a JSON object: "
                            '{\"is_vehicle\": true/false, \"reason\": \"brief explanation\"}. '
                            "No other text."
                        ),
                    },
                ],
            }],
            max_tokens=80,
            temperature=0.0,
        )

        answer = resp.choices[0].message.content.strip()
        # Parse the JSON response
        result = json.loads(answer)
        return result.get("is_vehicle", True), result.get("reason", "")

    except Exception as e:
        print(f"Image validation failed: {e}")
        # On error, allow the image through
        return True, ""


@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    """Serve static frontend files."""
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict severity from an uploaded image and/or text description.

    Supports three modes:
        - Multimodal (image + text) → uses fusion model
        - Image-only              → uses EfficientNet classifier
        - Text-only               → uses DistilBERT classifier

    Expects multipart form data:
        - image: an image file (optional)
        - text:  a damage description string (optional)
        - model_type: 'concat' or 'cross_attention' (for fusion, default: concat)
    """
    try:
        # Parse inputs
        has_image = "image" in request.files and request.files["image"].filename != ""
        text = request.form.get("text", "").strip()
        has_text = len(text) > 0

        if not has_image and not has_text:
            return jsonify({"error": "Please provide at least an image or text description."}), 400

        model_type = request.form.get("model_type", "concat")

        # Keep raw image bytes for LLM explanation
        raw_image_bytes = None

        # Determine which mode to use
        if has_image and has_text:
            mode = "multimodal"
        elif has_image:
            mode = "image_only"
        else:
            mode = "text_only"

        # ── Preprocess image (if provided) ──
        image_tensor = None
        if has_image:
            image_file = request.files["image"]
            raw_image_bytes = image_file.read()

            # Validate: reject non-vehicle images
            is_vehicle, reason = validate_vehicle_image(raw_image_bytes)
            if not is_vehicle:
                return jsonify({
                    "error": f"This image does not appear to show a vehicle. {reason}. Please upload a vehicle damage photo."
                }), 400

            image = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
            image_tensor = inference_transform(image).unsqueeze(0).to(device)

        # ── Tokenize text (if provided) ──
        input_ids = None
        attention_mask = None
        if has_text:
            encoding = tokenizer(
                text,
                max_length=MAX_TOKEN_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

        # ── Run inference ──
        with torch.no_grad():
            if mode == "multimodal":
                model = models.get(model_type)
                if model is None:
                    return jsonify({"error": f"Fusion model '{model_type}' not loaded."}), 500
                logits = model(image_tensor, input_ids, attention_mask)

            elif mode == "image_only":
                model = models.get("image_only")
                if model is None:
                    return jsonify({"error": "Image-only model not loaded."}), 500
                logits = model(image_tensor)

            else:  # text_only
                model = models.get("text_only")
                if model is None:
                    return jsonify({"error": "Text-only model not loaded."}), 500
                logits = model(input_ids, attention_mask)

            probabilities = F.softmax(logits, dim=1).squeeze(0)

        # Build response
        probs_list = probabilities.cpu().tolist()
        predicted_idx = int(torch.argmax(probabilities).item())

        pred_name = LABEL_NAMES[predicted_idx]
        conf = round(probs_list[predicted_idx] * 100, 1)
        prob_dict = {
            name: round(prob * 100, 1)
            for name, prob in zip(LABEL_NAMES, probs_list)
        }

        # Generate LLM explanation specific to this input
        explanation = generate_explanation(
            prediction=pred_name,
            confidence=conf,
            probabilities=prob_dict,
            user_text=text if has_text else None,
            image_bytes=raw_image_bytes,
        )

        response = {
            "prediction": pred_name,
            "confidence": conf,
            "probabilities": prob_dict,
            "mode": mode,
            "model_used": model_type if mode == "multimodal" else mode,
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys()),
        "device": str(device),
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Defect Severity Prediction API")
    parser.add_argument("--fold", type=int, default=4,
                        help="Which fold's weights to load (best fold)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    args = parser.parse_args()

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        oai_client = openai.OpenAI(api_key=api_key)
        print("OpenAI client initialized (contextual explanations enabled)")
    else:
        print("WARNING: No OPENAI_API_KEY found. Explanations will use fallback text.")

    print("Initializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    print(f"\nLoading all models (fold {args.fold})...")

    print("[1/4] Concat Fusion")
    models["concat"] = load_fusion_model("concat", fold=args.fold, device=device)

    print("[2/4] Cross-Attention Fusion")
    models["cross_attention"] = load_fusion_model("cross_attention", fold=args.fold, device=device)

    print("[3/4] Image-Only (EfficientNet-B0)")
    models["image_only"] = load_image_model(fold=args.fold, device=device)

    print("[4/4] Text-Only (DistilBERT)")
    models["text_only"] = load_text_model(fold=args.fold, device=device)

    print(f"\nAll models loaded! ({len(models)} models on {device})")
    print(f"Server starting on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

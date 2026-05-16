/**
 * AutoDefect AI — Frontend Application Logic
 *
 * Handles image upload, text input, form submission,
 * API communication, and animated results display.
 * Supports image-only, text-only, and multimodal predictions.
 */

(() => {
    "use strict";

    // ── Config ──────────────────────────────────────────────
    const API_BASE = window.location.origin;

    const SEVERITY_INFO = {
        Minor: {
            icon: "\u2705",
            explanation:
                "<strong>Minor Damage Detected.</strong> Surface-level cosmetic issues such as light scratches, small paint chips, or shallow dents. <br><br><em>Typical action:</em> Paintless dent repair or touch-up. Estimated repair cost is generally low. The vehicle remains fully safe and operational.",
        },
        Moderate: {
            icon: "\u26A0\uFE0F",
            explanation:
                "<strong>Moderate Damage Detected.</strong> Visible structural impact \u2014 cracked bumper covers, bent panels, or component misalignment. <br><br><em>Typical action:</em> Body shop inspection recommended. Parts may need replacement (bumper, fender, trim). Check for hidden damage to mounts or sensors before driving long distances.",
        },
        Severe: {
            icon: "\u274C",
            explanation:
                "<strong>Severe Damage Detected.</strong> Major structural compromise \u2014 crushed body panels, shattered glass, airbag deployment, or visible frame deformation. <br><br><em>Typical action:</em> Do <strong>not</strong> drive the vehicle. Tow to a certified collision center. File an insurance claim for professional assessment. Frame damage may render the vehicle a total loss.",
        },
    };

    const MODE_INFO = {
        multimodal: {
            icon: "\uD83D\uDD17",  // 🔗
            label: "Multimodal Fusion",
            desc: "Using both image + text with fusion model",
            color: "var(--indigo-400)",
        },
        image_only: {
            icon: "\uD83D\uDDBC\uFE0F",  // 🖼️
            label: "Image Only",
            desc: "Using EfficientNet-B0 image classifier",
            color: "var(--minor-color)",
        },
        text_only: {
            icon: "\uD83D\uDCDD",  // 📝
            label: "Text Only",
            desc: "Using DistilBERT text classifier",
            color: "var(--moderate-color)",
        },
    };

    // ── DOM references ──────────────────────────────────────
    const $  = (sel) => document.querySelector(sel);
    const statusDot        = $("#statusDot");
    const statusText       = $("#statusText");
    const uploadZone       = $("#uploadZone");
    const imageInput       = $("#imageInput");
    const uploadPlaceholder = $("#uploadPlaceholder");
    const uploadPreview    = $("#uploadPreview");
    const previewImage     = $("#previewImage");
    const removeImageBtn   = $("#removeImage");
    const textInput        = $("#textInput");
    const charCount        = $("#charCount");
    const predictBtn       = $("#predictBtn");
    const predictionForm   = $("#predictionForm");
    const resultsSection   = $("#resultsSection");
    const verdictContainer = $("#verdictContainer");
    const verdictValue     = $("#verdictValue");
    const verdictConfidence = $("#verdictConfidence");
    const probMinorVal     = $("#probMinorVal");
    const probModerateVal  = $("#probModerateVal");
    const probSevereVal    = $("#probSevereVal");
    const probMinorFill    = $("#probMinorFill");
    const probModerateFill = $("#probModerateFill");
    const probSevereFill   = $("#probSevereFill");
    const explanationIcon  = $("#explanationIcon");
    const explanationText  = $("#explanationText");
    const errorToast       = $("#errorToast");
    const errorMessage     = $("#errorMessage");
    const closeToastBtn    = $("#closeToast");
    const modeIndicator    = $("#modeIndicator");
    const modeIcon         = $("#modeIcon");
    const modeText         = $("#modeText");
    const modelSelector    = $("#modelSelector");
    const resultModeBadge  = $("#resultModeBadge");
    const resultModeIcon   = $("#resultModeIcon");
    const resultModeText   = $("#resultModeText");

    // ── State ───────────────────────────────────────────────
    let selectedFile = null;

    // ── Health check ────────────────────────────────────────
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();
            if (data.status === "ok" && data.models_loaded && data.models_loaded.length > 0) {
                statusDot.classList.add("status-dot--ok");
                statusDot.classList.remove("status-dot--error");
                const count = data.models_loaded.length;
                statusText.textContent = `${count} models ready (${data.device})`;
            } else {
                throw new Error("Models not loaded");
            }
        } catch {
            statusDot.classList.add("status-dot--error");
            statusDot.classList.remove("status-dot--ok");
            statusText.textContent = "Model offline";
        }
    }

    // ── Determine current mode ──────────────────────────────
    function getCurrentMode() {
        const hasImage = selectedFile !== null;
        const hasText  = textInput.value.trim().length > 0;
        if (hasImage && hasText) return "multimodal";
        if (hasImage)           return "image_only";
        if (hasText)            return "text_only";
        return null;
    }

    // ── Update mode indicator ───────────────────────────────
    function updateModeIndicator() {
        const mode = getCurrentMode();

        if (!mode) {
            modeIndicator.className = "mode-indicator";
            modeIcon.textContent = "\u2139\uFE0F";
            modeText.textContent = "Provide at least an image or text description to start.";
            modelSelector.classList.remove("model-selector--active");
        } else {
            const info = MODE_INFO[mode];
            modeIndicator.className = `mode-indicator mode-indicator--${mode}`;
            modeIcon.textContent = info.icon;
            modeText.textContent = info.desc;

            // Show/dim model selector based on whether fusion is active
            if (mode === "multimodal") {
                modelSelector.classList.add("model-selector--active");
            } else {
                modelSelector.classList.remove("model-selector--active");
            }
        }
    }

    // ── Form validation ─────────────────────────────────────
    function validateForm() {
        const mode = getCurrentMode();
        predictBtn.disabled = mode === null;
        updateModeIndicator();
    }

    // ── Image upload handling ───────────────────────────────
    function handleFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            showError("Please select a valid image file.");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            showError("Image must be under 10 MB.");
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadPlaceholder.hidden = true;
            uploadPreview.hidden = false;
        };
        reader.readAsDataURL(file);
        validateForm();
    }

    function clearImage() {
        selectedFile = null;
        imageInput.value = "";
        previewImage.src = "";
        uploadPlaceholder.hidden = false;
        uploadPreview.hidden = true;
        validateForm();
    }

    // Click to upload
    uploadZone.addEventListener("click", (e) => {
        if (e.target === removeImageBtn || removeImageBtn.contains(e.target)) return;
        imageInput.click();
    });

    uploadZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            imageInput.click();
        }
    });

    imageInput.addEventListener("change", () => {
        if (imageInput.files.length > 0) handleFile(imageInput.files[0]);
    });

    removeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearImage();
    });

    // Drag and drop
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("upload-zone--drag");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("upload-zone--drag");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("upload-zone--drag");
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // ── Text input handling ─────────────────────────────────
    textInput.addEventListener("input", () => {
        charCount.textContent = textInput.value.length;
        validateForm();
    });

    // ── Error toast ─────────────────────────────────────────
    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.hidden = false;
        setTimeout(() => { errorToast.hidden = true; }, 6000);
    }

    closeToastBtn.addEventListener("click", () => { errorToast.hidden = true; });

    // ── Display results ─────────────────────────────────────
    function displayResults(data) {
        const { prediction, confidence, probabilities, mode, model_used } = data;

        // Show section
        resultsSection.hidden = false;

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);

        // Mode badge
        const mInfo = MODE_INFO[mode] || MODE_INFO.multimodal;
        resultModeIcon.textContent = mInfo.icon;
        resultModeText.textContent = `${mInfo.label}${mode === "multimodal" ? ` (${model_used.replace("_", "-")})` : ""}`;
        resultModeBadge.className = `results__mode results__mode--${mode}`;

        // Verdict
        const severity = prediction;
        const cls = severity.toLowerCase();
        verdictContainer.className = `results__verdict verdict--${cls}`;
        verdictValue.textContent = severity;
        verdictConfidence.textContent = `${confidence}% confidence`;

        // Probability bars (animate after a tiny delay)
        setTimeout(() => {
            const pMinor = probabilities.Minor || 0;
            const pMod   = probabilities.Moderate || 0;
            const pSev   = probabilities.Severe || 0;

            probMinorVal.textContent    = `${pMinor}%`;
            probModerateVal.textContent = `${pMod}%`;
            probSevereVal.textContent   = `${pSev}%`;

            probMinorFill.style.width    = `${pMinor}%`;
            probModerateFill.style.width = `${pMod}%`;
            probSevereFill.style.width   = `${pSev}%`;
        }, 200);

        // Explanation — prefer LLM-generated, fall back to static
        const fallback = SEVERITY_INFO[severity];
        if (data.explanation) {
            explanationIcon.textContent = fallback ? fallback.icon : "\uD83D\uDD0D";
            explanationText.textContent = data.explanation;
        } else if (fallback) {
            explanationIcon.textContent = fallback.icon;
            explanationText.innerHTML   = fallback.explanation;
        }
    }

    // ── Form submission ─────────────────────────────────────
    predictionForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (predictBtn.disabled) return;

        // Loading state
        predictBtn.classList.add("btn--loading");
        predictBtn.disabled = true;
        resultsSection.hidden = true;

        // Reset probability bar widths for re-animation
        probMinorFill.style.width    = "0%";
        probModerateFill.style.width = "0%";
        probSevereFill.style.width   = "0%";

        // Build form data
        const formData = new FormData();

        if (selectedFile) {
            formData.append("image", selectedFile);
        }

        const text = textInput.value.trim();
        if (text) {
            formData.append("text", text);
        }

        const modelType = document.querySelector('input[name="model_type"]:checked').value;
        formData.append("model_type", modelType);

        try {
            const res = await fetch(`${API_BASE}/api/predict`, {
                method: "POST",
                body: formData,
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || `Server returned ${res.status}`);
            }

            displayResults(data);
        } catch (err) {
            showError(err.message || "Failed to get prediction. Is the server running?");
        } finally {
            predictBtn.classList.remove("btn--loading");
            validateForm();
        }
    });

    // ── Init ────────────────────────────────────────────────
    checkHealth();
    setInterval(checkHealth, 15000);
    validateForm();
})();

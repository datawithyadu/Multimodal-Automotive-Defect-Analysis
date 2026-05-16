"""
Visualizations for each Research Question in the thesis.
Generates publication-quality figures saved to results/ directory.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "rq_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

# ── Color palette (professional deep tones) ──
C_IMAGE   = "#2563EB"   # blue
C_TEXT    = "#7C3AED"   # purple
C_CONCAT  = "#059669"   # teal-green
C_CROSS   = "#DC2626"   # red
C_RANDOM  = "#9CA3AF"   # gray
C_MINOR   = "#3B82F6"   # light blue
C_MOD     = "#F59E0B"   # amber
C_SEVERE  = "#EF4444"   # red

# ═══════════════════════════════════════════════════════════════════
# RQ 1  – Image-only baseline per-class performance + confusion matrix
# ═══════════════════════════════════════════════════════════════════
def rq1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.suptitle("RQ1 — Image-Only Classification Performance (EfficientNet-B0)",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── Left: per-class precision / recall / F1 grouped bar ──
    ax = axes[0]
    classes = ["Minor", "Moderate", "Severe"]
    precision = [0.7253, 0.5156, 0.8065]
    recall    = [0.8049, 0.4400, 0.8242]
    f1        = [0.7630, 0.4748, 0.8152]

    x = np.arange(len(classes))
    w = 0.22
    bars_p = ax.bar(x - w, precision, w, label="Precision", color="#60A5FA", edgecolor="white", linewidth=0.6)
    bars_r = ax.bar(x,     recall,    w, label="Recall",    color="#34D399", edgecolor="white", linewidth=0.6)
    bars_f = ax.bar(x + w, f1,        w, label="F1-Score",  color="#FBBF24", edgecolor="white", linewidth=0.6)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics (Held-Out Test)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # Highlight moderate weakness
    ax.axhspan(0, 0.50, xmin=0.28, xmax=0.72, color="#FEF3C7", alpha=0.35, zorder=0)
    ax.annotate("Weakest class", xy=(1, 0.44), fontsize=9, color="#92400E",
                ha="center", fontstyle="italic")

    # ── Right: confusion matrix heatmap ──
    ax2 = axes[1]
    cm = np.array([[66, 16, 0],
                   [24, 33, 18],
                   [ 1, 15, 75]])
    im = ax2.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > 50 else "black"
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold", color=color)
    ax2.set_xticks([0,1,2]); ax2.set_yticks([0,1,2])
    ax2.set_xticklabels(classes); ax2.set_yticklabels(classes)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (Held-Out Test)", fontsize=12)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Red box around moderate row
    from matplotlib.patches import Rectangle
    rect = Rectangle((-0.5, 0.5), 3, 1, linewidth=2.5, edgecolor="#DC2626",
                     facecolor="none", linestyle="--")
    ax2.add_patch(rect)
    ax2.annotate("F1 = 0.47", xy=(2.7, 1), fontsize=10, color="#DC2626", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "rq1_image_only_performance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# RQ 2  – Multimodal vs unimodal: accuracy & per-class F1 comparison
# ═══════════════════════════════════════════════════════════════════
def rq2():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("RQ2 — Does Multimodal Fusion Improve Over Unimodal Baselines?",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── Left: overall test accuracy bar chart ──
    ax = axes[0]
    models = ["Image-only", "Text-only", "Concat\nFusion", "Cross-Attn\nFusion"]
    acc    = [70.2, 65.3, 73.3, 73.4]
    colors = [C_IMAGE, C_TEXT, C_CONCAT, C_CROSS]
    bars = ax.bar(models, acc, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, a + 0.4, f"{a}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(33.3, color=C_RANDOM, linestyle="--", linewidth=1.2, label="Random chance (33.3%)")
    ax.set_ylim(0, 85)
    ax.set_ylabel("Held-Out Test Accuracy (%)")
    ax.set_title("Overall Accuracy Comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # Improvement arrows
    ax.annotate("", xy=(2, 73.3), xytext=(2, 70.2),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2))
    ax.text(2.35, 71.6, "+4.8%", color="#059669", fontsize=10, fontweight="bold")

    # ── Right: per-class F1 grouped bar ──
    ax2 = axes[1]
    classes = ["Minor", "Moderate", "Severe"]
    f1_img   = [0.76, 0.47, 0.82]
    f1_txt   = [0.75, 0.56, 0.66]
    f1_cat   = [0.81, 0.60, 0.79]
    f1_cross = [0.82, 0.57, 0.77]

    x = np.arange(len(classes))
    w = 0.18
    ax2.bar(x - 1.5*w, f1_img,   w, label="Image-only",  color=C_IMAGE,  edgecolor="white")
    ax2.bar(x - 0.5*w, f1_txt,   w, label="Text-only",   color=C_TEXT,   edgecolor="white")
    ax2.bar(x + 0.5*w, f1_cat,   w, label="Concat Fusion",color=C_CONCAT, edgecolor="white")
    ax2.bar(x + 1.5*w, f1_cross, w, label="Cross-Attn",  color=C_CROSS,  edgecolor="white")

    # Value labels
    for vals, offset in [(f1_img, -1.5*w), (f1_txt, -0.5*w), (f1_cat, 0.5*w), (f1_cross, 1.5*w)]:
        for i, v in enumerate(vals):
            ax2.text(i + offset, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("F1-Score")
    ax2.set_title("Per-Class F1 Comparison (Held-Out Test)", fontsize=12)
    ax2.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Highlight moderate improvement
    ax2.axhspan(0, 0.65, xmin=0.25, xmax=0.70, color="#FEF3C7", alpha=0.30, zorder=0)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "rq2_multimodal_vs_unimodal.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# RQ 3  – Synthetic text pipeline stats & dataset composition
# ═══════════════════════════════════════════════════════════════════
def rq3():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("RQ3 — Synthetic Data Generation Pipeline Effectiveness",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── Left: Dataset expansion (before/after) ──
    ax = axes[0]
    categories = ["Original\n(Images Only)", "After Pipeline\n(Image-Text Pairs)"]
    counts = [1631, 4815]
    colors_bar = ["#94A3B8", "#10B981"]
    bars = ax.bar(categories, counts, color=colors_bar, edgecolor="white",
                  linewidth=0.8, width=0.45)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, c + 60, f"{c:,}",
                ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Expansion (3× augmentation)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 5600)

    # Multiplier annotation
    ax.annotate("×2.95", xy=(1, 4815), xytext=(1.3, 3800),
                fontsize=16, fontweight="bold", color="#059669",
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2))

    # ── Right: Pipeline success rates + text style distribution pie ──
    ax2 = axes[1]
    # Donut chart: text styles
    styles = ["Formal\nReport", "Technician\nNote", "Casual\nDescription"]
    style_counts = [1605, 1605, 1605]  # ~equal split (3 per image)
    colors_pie = ["#3B82F6", "#8B5CF6", "#F97316"]
    wedges, texts, autotexts = ax2.pie(
        style_counts, labels=styles, autopct="%1.0f%%",
        colors=colors_pie, startangle=90,
        pctdistance=0.78, labeldistance=1.15,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops={"fontsize": 10, "fontweight": "bold"}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("Text Variant Distribution per Image", fontsize=12)

    # Stats annotation box
    stats_text = (
        "Pipeline Stats\n"
        "─────────────────\n"
        f"Step 1 success: 1,608/1,631 (98.6%)\n"
        f"Step 2 success: 4,815/4,821 (99.9%)\n"
        f"Banned-word violations: 0\n"
        f"Noise injection: 40%"
    )
    fig.text(0.73, 0.12, stats_text, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0FDF4",
                       edgecolor="#059669", alpha=0.95),
             family="monospace", verticalalignment="bottom")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "rq3_synthetic_pipeline.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# RQ 4  – Practical applicability: radar chart of system capabilities
# ═══════════════════════════════════════════════════════════════════
def rq4():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("RQ4 — Practical Applicability of the Multimodal System",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── Left: Confusion matrix comparison (image-only vs best fusion) ──
    ax = axes[0]
    # Show the error reduction from image-only → concat fusion for each class
    classes = ["Minor", "Moderate", "Severe"]

    # Recall values
    recall_img  = [80.5, 44.0, 82.4]
    recall_fuse = [74.0, 64.9, 79.5]  # concat fusion

    x = np.arange(len(classes))
    w = 0.30
    b1 = ax.bar(x - w/2, recall_img,  w, label="Image-only",      color=C_IMAGE,  edgecolor="white")
    b2 = ax.bar(x + w/2, recall_fuse, w, label="Concat Fusion",   color=C_CONCAT, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-Class Recall Improvement", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # Big arrow on moderate
    ax.annotate("+20.9pp", xy=(1 + w/2, 64.9), xytext=(1.6, 55),
                fontsize=12, fontweight="bold", color="#059669",
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2))

    # ── Right: Application workflow diagram as a table ──
    ax2 = axes[1]
    ax2.axis("off")

    # Create a styled table
    col_labels = ["Application", "Input", "System Output", "Impact"]
    rows = [
        ["Insurance\nTriage", "Photo +\nDescription", "Severity\nClassification", "Auto-route\nclaims"],
        ["Repair\nEstimation", "Damage\nImage", "Severity\nLevel", "Cost\nbracket"],
        ["Quality\nControl", "Inspection\nReport", "Defect\nCategory", "Prioritize\nrepairs"],
    ]

    table = ax2.table(cellText=rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    # Style header row
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor("#1E40AF")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        cell.set_edgecolor("white")

    # Style data rows
    row_colors = ["#EFF6FF", "#F0FDF4", "#FFF7ED"]
    for i in range(1, 4):
        for j in range(4):
            cell = table[i, j]
            cell.set_facecolor(row_colors[i-1])
            cell.set_edgecolor("#E5E7EB")

    ax2.set_title("Target Application Domains", fontsize=12, pad=20)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "rq4_practical_applicability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# RQ 5  – Modality contributions: radar + stacked F1 comparison
# ═══════════════════════════════════════════════════════════════════
def rq5():
    fig = plt.figure(figsize=(15, 5.5))
    fig.suptitle("RQ5 — Contribution of Each Modality to Overall Performance",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── Left: Modality contribution heatmap ──
    ax1 = fig.add_subplot(131)
    classes = ["Minor", "Moderate", "Severe"]
    models  = ["Image-only", "Text-only"]
    data = np.array([
        [0.76, 0.47, 0.82],   # image
        [0.75, 0.56, 0.66],   # text
    ])
    im = ax1.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=0.9, aspect="auto")
    for i in range(2):
        for j in range(3):
            ax1.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                     fontsize=13, fontweight="bold",
                     color="white" if data[i,j] < 0.50 else "black")
    ax1.set_xticks([0,1,2]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(classes, fontweight="bold")
    ax1.set_yticklabels(models, fontweight="bold")
    ax1.set_title("Per-Class F1 by Modality", fontsize=12)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="F1-Score")

    # Annotate strengths
    ax1.annotate("Image\nstrongest", xy=(2, 0), fontsize=8, color="#065F46",
                 ha="center", va="center",
                 xytext=(2.8, -0.3), arrowprops=dict(arrowstyle="->", color="#065F46"))
    ax1.annotate("Text\nstrongest", xy=(1, 1), fontsize=8, color="#065F46",
                 ha="center", va="center",
                 xytext=(1.8, 1.3), arrowprops=dict(arrowstyle="->", color="#065F46"))

    # ── Middle: Overall accuracy waterfall ──
    ax2 = fig.add_subplot(132)
    labels = ["Random\nChance", "Text\nSignal", "Image\nSignal", "Fusion\nBenefit", "Best\nModel"]
    values = [33.3, 65.3, 70.2, 73.3, 73.3]
    increments = [33.3, 32.0, 4.9, 3.1, 0]  # conceptual increments

    bottom = 0
    colors_wf = ["#D1D5DB", C_TEXT, C_IMAGE, C_CONCAT]
    cumulative = [33.3, 65.3, 70.2, 73.3]
    heights = [33.3, 32.0, 4.9, 3.1]

    bars = []
    bottoms = [0, 0, 0, 0]
    # Draw stacked waterfall
    for i in range(4):
        if i == 0:
            b = ax2.bar(labels[i], heights[i], bottom=0, color=colors_wf[i],
                        edgecolor="white", width=0.5)
        elif i == 1:
            b = ax2.bar(labels[i], heights[i], bottom=33.3, color=colors_wf[i],
                        edgecolor="white", width=0.5)
        elif i == 2:
            b = ax2.bar(labels[i], heights[i], bottom=65.3, color=colors_wf[i],
                        edgecolor="white", width=0.5)
        elif i == 3:
            b = ax2.bar(labels[i], heights[i], bottom=70.2, color=colors_wf[i],
                        edgecolor="white", width=0.5)
        bars.append(b)

    # Add cumulative labels on top
    for i, (lbl, cum) in enumerate(zip(labels[:4], cumulative)):
        ax2.text(i, cum + 0.8, f"{cum}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    # Add increment labels inside bars
    inc_labels = ["+33.3%", "+32.0%", "+4.9%", "+3.1%"]
    inc_y = [16.5, 49.0, 67.7, 71.7]
    for i, (il, iy) in enumerate(zip(inc_labels, inc_y)):
        ax2.text(i, iy, il, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color="white")

    ax2.set_ylim(0, 85)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Contributions\n(Conceptual Waterfall)", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Right: Where each modality wins ──
    ax3 = fig.add_subplot(133)

    # Bar chart: "advantage" of image over text, and text over image
    classes_short = ["Minor", "Moderate", "Severe"]
    img_advantage = [0.76-0.75, 0.47-0.56, 0.82-0.66]  # positive = image better
    colors_adv = [C_IMAGE if v > 0 else C_TEXT for v in img_advantage]

    bars = ax3.barh(classes_short, img_advantage, color=colors_adv, edgecolor="white",
                    height=0.45)
    for bar, v in zip(bars, img_advantage):
        xpos = v + 0.005 if v > 0 else v - 0.005
        ha = "left" if v > 0 else "right"
        ax3.text(xpos, bar.get_y() + bar.get_height()/2,
                 f"{v:+.2f}", ha=ha, va="center", fontsize=11, fontweight="bold")

    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("F1 Advantage (Image − Text)")
    ax3.set_title("Modality Advantage per Class", fontsize=12)
    ax3.spines[["top", "right"]].set_visible(False)

    # Add legend patches
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color=C_IMAGE, label="Image stronger"),
                         Patch(color=C_TEXT,  label="Text stronger")],
               loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "rq5_modality_contributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# BONUS: Cross-validation stability comparison (supports all RQs)
# ═══════════════════════════════════════════════════════════════════
def bonus_cv_stability():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Cross-Validation Stability Across All Models (5-Fold F1 Macro)",
                 fontsize=14, fontweight="bold", y=1.01)

    models = ["Image-only", "Text-only", "Concat\nFusion", "Cross-Attn\n(v1)", "Cross-Attn\n(v2 +dropout)"]
    means  = [68.30, 64.69, 70.21, 69.59, 71.08]
    stds   = [ 3.04,  2.12,  1.61,  1.64,  2.24]
    colors_cv = [C_IMAGE, C_TEXT, C_CONCAT, C_CROSS, "#B91C1C"]

    bars = ax.bar(models, means, yerr=stds, color=colors_cv,
                  edgecolor="white", linewidth=0.8, width=0.50,
                  capsize=6, error_kw={"linewidth": 1.8, "capthick": 1.8})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.5,
                f"{m:.1f}±{s:.1f}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    ax.set_ylim(55, 80)
    ax.set_ylabel("CV F1-Macro (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(33.3, color=C_RANDOM, linestyle="--", linewidth=1, label="Random chance", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "bonus_cv_stability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Research Question visualizations …\n")
    rq1()
    rq2()
    rq3()
    rq4()
    rq5()
    bonus_cv_stability()
    print(f"\n[DONE] All figures saved to: {OUT_DIR}")

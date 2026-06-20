"""
Statistical significance testing for cross-validation F1 scores.

Compares all four models using:
  - Paired t-test (parametric)
  - Wilcoxon signed-rank test (non-parametric)
  - Cohen's d effect size (paired)

IMPORTANT CAVEAT: With n=5 folds, the Wilcoxon test cannot produce
p < 0.0625 (two-tailed) and the t-test has very low power (~20-30%
for small effects). Results should be interpreted as directional
evidence, not definitive proof. This is a known limitation of
small-dataset CV evaluation and is disclosed in the thesis.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# ── Per-fold macro-F1 scores (from JSON results files) ──────────────

IMAGE_ONLY_F1   = [0.6626, 0.7264, 0.6377, 0.6926, 0.6955]
TEXT_ONLY_F1    = [0.6879, 0.6355, 0.6376, 0.6451, 0.6284]
CONCAT_F1       = [0.7164, 0.7008, 0.6742, 0.7195, 0.6994]
CROSS_ATTN_F1   = [0.7329, 0.7012, 0.6870, 0.7420, 0.6911]  # v2 (modality dropout + L2 norm)

MODELS = {
    "image_only":     np.array(IMAGE_ONLY_F1),
    "text_only":      np.array(TEXT_ONLY_F1),
    "concat_fusion":  np.array(CONCAT_F1),
    "cross_attention": np.array(CROSS_ATTN_F1),
}


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (difference scores)."""
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def compare(name_a: str, name_b: str, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Run both tests and compute effect size for a vs b."""
    diff = scores_a - scores_b
    mean_diff = diff.mean()

    # Paired t-test (two-tailed — we don't assume direction during testing)
    t_stat, t_p = stats.ttest_rel(scores_a, scores_b)

    # Wilcoxon signed-rank test (zero_method='wilcox' drops zero diffs)
    # With n=5, minimum two-tailed p is 2/32 = 0.0625
    try:
        w_stat, w_p = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox",
                                      alternative="two-sided")
    except ValueError:
        # All differences are zero
        w_stat, w_p = 0.0, 1.0

    d = cohens_d_paired(scores_a, scores_b)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_diff": round(float(mean_diff), 4),
        "direction": f"{name_a} {'>' if mean_diff > 0 else '<'} {name_b}",
        "t_statistic": round(float(t_stat), 4),
        "t_p_value": round(float(t_p), 4),
        "t_significant_p05": bool(t_p < 0.05),
        "wilcoxon_statistic": round(float(w_stat), 4),
        "wilcoxon_p_value": round(float(w_p), 4),
        "wilcoxon_significant_p05": bool(w_p < 0.05),
        "cohens_d": round(d, 4),
        "effect_size": effect_label(d),
    }


def print_result(r: dict) -> None:
    sig_t = "* p<0.05" if r["t_significant_p05"] else "  n.s.  "
    sig_w = "* p<0.05" if r["wilcoxon_significant_p05"] else "  n.s.  "
    print(f"\n  {r['comparison']}")
    print(f"    Mean F1 diff : {r['mean_diff']:+.4f}  ({r['direction']})")
    print(f"    Paired t-test: t={r['t_statistic']:+.3f}, p={r['t_p_value']:.4f}  [{sig_t}]")
    print(f"    Wilcoxon     : W={r['wilcoxon_statistic']:.1f},  p={r['wilcoxon_p_value']:.4f}  [{sig_w}]")
    print(f"    Effect size  : Cohen's d={r['cohens_d']:+.3f}  ({r['effect_size']})")


def main():
    print("=" * 65)
    print("  Statistical Significance Tests — 5-Fold CV Macro-F1 Scores")
    print("=" * 65)

    print("\n  Per-fold F1 scores:")
    for name, scores in MODELS.items():
        print(f"    {name:<20s}: {[round(s, 4) for s in scores]}")
        print(f"    {'':20s}  mean={scores.mean():.4f}  std={scores.std(ddof=1):.4f}")

    print("\n" + "-" * 65)
    print("  KEY COMPARISONS")
    print("-" * 65)

    comparisons = [
        ("concat_fusion",   "image_only"),
        ("cross_attention",  "image_only"),
        ("concat_fusion",   "text_only"),
        ("cross_attention",  "text_only"),
        ("concat_fusion",   "cross_attention"),
    ]

    results = []
    for a, b in comparisons:
        r = compare(a, b, MODELS[a], MODELS[b])
        print_result(r)
        results.append(r)

    print("\n" + "-" * 65)
    print("  INTERPRETATION NOTES")
    print("-" * 65)
    print("""
  n=5 folds is a known limitation for significance testing:
    - Paired t-test: very low power (~20-30% for small effects)
    - Wilcoxon: cannot achieve p < 0.0625 (two-tailed) with n=5
      The minimum achievable p is 2 * (1/2^5) = 0.0625

  Cohen's d thresholds: |d|<0.2 negligible, 0.2-0.5 small,
                        0.5-0.8 medium, >0.8 large

  Recommendation: Report mean ± std differences + Cohen's d.
  The direction and effect size are the meaningful claims here;
  p-values are underpowered but included for completeness.
""")

    # Save results
    output = {
        "caveat": (
            "n=5 folds limits statistical power. Wilcoxon cannot achieve p<0.0625 "
            "two-tailed. Use mean difference and Cohen's d as primary evidence."
        ),
        "fold_scores": {k: v.tolist() for k, v in MODELS.items()},
        "comparisons": results,
    }
    out_path = RESULTS_DIR / "significance_tests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()

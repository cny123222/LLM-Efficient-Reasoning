#!/usr/bin/env python3
"""
Parameter sensitivity visualization for DynaTree (WikiText-2).

Data source:
  results/adaptive/sensitivity/comprehensive_sensitivity_1500tokens.json

We visualize a comprehensive sweep over:
  - confidence thresholds (tau_h, tau_l)
  - branch bounds (Bmin, Bmax)
  - depth ranges (D0, Dmax)
  - a small set of cross-combinations

All numbers are real and traceable to the JSON above. Speedup in the figure is computed
relative to the AR throughput measured in the same sweep run.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Academic paper style settings (consistent with other plotting scripts)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.edgecolor"] = "#999999"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["font.size"] = 10


def main() -> None:
    src = Path("results/adaptive/sensitivity/comprehensive_sensitivity_1500tokens.json")
    data = json.loads(src.read_text())

    rows = data.get("all_results", [])
    ar = next(r for r in rows if r.get("category") == "baseline" and r.get("config_name") == "Baseline (AR)")
    ar_thr = float(ar["throughput_tps"])

    def sp(r: dict) -> float:
        return float(r.get("throughput_tps", 0.0)) / ar_thr if ar_thr > 0 else 0.0

    # Split by category
    cats = ["threshold", "branch", "depth", "cross"]
    by_cat = {c: [r for r in rows if r.get("category") == c] for c in cats}
    for c in cats:
        by_cat[c] = sorted(by_cat[c], key=lambda r: float(r.get("throughput_tps", 0.0)))

    # 2x2 horizontal bar charts, one per category (sorted by throughput)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    axes = axes.reshape(2, 2)

    cat_titles = {
        "threshold": r"(a) Threshold sweep $(\tau_h,\tau_\ell)$",
        "branch": r"(b) Branch sweep $(B_{\min},B_{\max})$",
        "depth": r"(c) Depth sweep $(D_0,D_{\max})$",
        "cross": r"(d) Cross combinations",
    }

    base_color = "#8BACC6"
    best_color = "#D97757"

    def short_name(r: dict) -> str:
        name = str(r.get("config_name", ""))
        # Keep the important part, remove prefixes to reduce clutter.
        name = name.replace("thresh_", "h").replace("_l", " l")
        name = name.replace("branch_", "B ").replace("depth_", "D ").replace("cross_", "")
        return name

    for ax, cat in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], cats):
        data_cat = by_cat[cat]
        if not data_cat:
            ax.axis("off")
            continue
        y = np.arange(len(data_cat))
        thr = np.array([float(r.get("throughput_tps", 0.0)) for r in data_cat])
        best_idx = int(np.argmax(thr))

        colors = [base_color for _ in data_cat]
        colors[best_idx] = best_color

        ax.barh(y, thr, color=colors, edgecolor="#333333", linewidth=0.4, alpha=0.92)
        ax.axvline(ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.6)

        ax.set_yticks(y)
        ax.set_yticklabels([short_name(r) for r in data_cat], fontsize=8)
        ax.set_title(cat_titles[cat], fontsize=11, pad=6)
        ax.set_xlabel("Throughput (tokens/s)", fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.35, linewidth=0.6)

        # annotate speedup on the best bar only (avoid clutter)
        r_best = data_cat[best_idx]
        ax.text(
            thr[best_idx] + (thr.max() * 0.01),
            best_idx,
            f"{sp(r_best):.2f}×",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    plt.tight_layout()

    out_pdf = Path("figures/parameter_sensitivity.pdf")
    out_png = Path("figures/parameter_sensitivity.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print(f"AR throughput (for speedup): {ar_thr:.2f}")
    for c in cats:
        print(f"{c}: {len(by_cat[c])} configs")


if __name__ == "__main__":
    main()



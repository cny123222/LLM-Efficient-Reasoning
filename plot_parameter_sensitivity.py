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

    fixed = next(
        r for r in rows if r.get("category") == "baseline" and str(r.get("config_name", "")).startswith("Fixed Tree")
    )

    # Split by category (non-baseline)
    cats = ["threshold", "branch", "depth", "cross"]
    by_cat = {c: [r for r in rows if r.get("category") == c] for c in cats}

    colors = {
        "threshold": "#4A708B",  # steel blue
        "branch": "#6FAF8A",     # green
        "depth": "#8BACC6",      # sky blue
        "cross": "#D97757",      # terra cotta
    }
    markers = {"threshold": "o", "branch": "s", "depth": "D", "cross": "^"}

    # Find best non-baseline config by throughput
    all_non_base = [r for r in rows if r.get("category") in cats]
    best = max(all_non_base, key=lambda r: float(r.get("throughput_tps", 0.0)))

    # --- Figure: 1x2 panels (distribution + tradeoff) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # (a) Throughput distribution by category (jittered scatter)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cats):
        data_cat = by_cat[c]
        if not data_cat:
            continue
        thr = np.array([float(r.get("throughput_tps", 0.0)) for r in data_cat])
        std = np.array([float(r.get("throughput_std", 0.0)) for r in data_cat])
        x = i + rng.uniform(-0.14, 0.14, size=len(data_cat))
        ax1.scatter(
            x,
            thr,
            s=42,
            marker=markers[c],
            color=colors[c],
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.9,
            label=c,
        )
        # thin error bars (std), to show variability without clutter
        ax1.errorbar(x, thr, yerr=std, fmt="none", ecolor="#666666", elinewidth=0.5, alpha=0.25, capsize=0)

    # Baselines as reference lines/markers
    ax1.axhline(ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7, label="AR (sweep)")
    ax1.axhline(float(fixed.get("throughput_tps", 0.0)), color="#777777", linestyle=":", linewidth=0.9, alpha=0.7, label="Fixed Tree")

    # Highlight best
    ax1.scatter(
        [cats.index(best["category"])],
        [float(best["throughput_tps"])],
        s=120,
        marker="*",
        color="#D97757",
        edgecolor="#333333",
        linewidth=0.5,
        zorder=5,
    )
    ax1.text(
        cats.index(best["category"]) + 0.05,
        float(best["throughput_tps"]) + 2.0,
        f"best {sp(best):.2f}×",
        fontsize=9,
        color="#333333",
    )

    ax1.set_title(r"(a) Throughput across sweep categories", fontsize=11, pad=8)
    ax1.set_xticks(np.arange(len(cats)))
    ax1.set_xticklabels(["thresh", "branch", "depth", "cross"], fontsize=9)
    ax1.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.35, linewidth=0.6)

    # (b) Tradeoff: throughput vs acceptance
    for c in cats:
        data_cat = by_cat[c]
        if not data_cat:
            continue
        thr = np.array([float(r.get("throughput_tps", 0.0)) for r in data_cat])
        acc = np.array([float(r.get("acceptance_rate", 0.0)) * 100.0 for r in data_cat])
        ax2.scatter(
            acc,
            thr,
            s=46,
            marker=markers[c],
            color=colors[c],
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.9,
            label=c,
        )

    # Baseline points
    ax2.scatter([0.0], [ar_thr], s=70, marker="x", color="#333333", linewidth=1.6, label="AR (sweep)")
    ax2.scatter(
        [float(fixed.get("acceptance_rate", 0.0)) * 100.0],
        [float(fixed.get("throughput_tps", 0.0))],
        s=80,
        marker="P",
        color="#777777",
        edgecolor="#333333",
        linewidth=0.35,
        label="Fixed Tree",
    )

    ax2.scatter(
        [float(best.get("acceptance_rate", 0.0)) * 100.0],
        [float(best.get("throughput_tps", 0.0))],
        s=140,
        marker="*",
        color="#D97757",
        edgecolor="#333333",
        linewidth=0.5,
        zorder=5,
    )

    ax2.set_title(r"(b) Throughput vs acceptance", fontsize=11, pad=8)
    ax2.set_xlabel("Accept. (%)", fontsize=11)
    ax2.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax2.set_xlim(0, 110)

    # Keep a compact legend
    handles, labels_ = ax2.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels_):
        if l not in uniq:
            uniq[l] = h
    ax2.legend(list(uniq.values()), list(uniq.keys()), loc="lower right", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

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



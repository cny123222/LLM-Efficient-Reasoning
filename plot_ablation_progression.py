#!/usr/bin/env python3
"""
Plot progressive ablation (Fixed Tree -> + Dynamic Breadth & Depth -> + History Adaptation)
using the paper's main benchmark JSON (WikiText-2, T=1500).

Goal: a compact, multi-metric figure (SpecInfer-style) that explains *why* throughput improves.
We render a 1x2 layout (two panels side-by-side), with **two metrics per panel**:
  - Left: Throughput (bars) + #Iter (line, separate y-axis)
  - Right: tokens/iter (line) + Accept.(%) (line, separate y-axis)
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
    src = Path("results/adaptive/main/paper_benchmark_main_1500tokens_2.json")
    data = json.loads(src.read_text())
    m = {r["method"]: r for r in data["all_results"]}

    ar = m["Baseline (AR)"]
    fixed = m["Fixed Tree (D=5, B=2)"]
    phase2 = m["Phase 2: + Dynamic Depth"]  # corresponds to "+ Dynamic Breadth & Depth"
    phase3 = m["Phase 3: + History Adjust"]  # corresponds to "+ History Adaptation" (DynaTree)

    labels = ["Fixed Tree", "+ Breadth&Depth", "+ History"]
    x = np.arange(len(labels))
    throughput = [
        fixed["throughput_tps"],
        phase2["throughput_tps"],
        phase3["throughput_tps"],
    ]
    speedup = [t / ar["throughput_tps"] for t in throughput]
    rounds = [
        fixed["total_rounds"],
        phase2["total_rounds"],
        phase3["total_rounds"],
    ]

    tokens_per_iter = [
        fixed.get("tokens_per_round", 0.0),
        phase2.get("tokens_per_round", 0.0),
        phase3.get("tokens_per_round", 0.0),
    ]
    avg_path_len = [
        fixed.get("avg_path_length", 0.0),
        phase2.get("avg_path_length", 0.0),
        phase3.get("avg_path_length", 0.0),
    ]
    accept = [
        fixed.get("acceptance_rate", 0.0),
        phase2.get("acceptance_rate", 0.0),
        phase3.get("acceptance_rate", 0.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))

    colors = ["#6FAF8A", "#8BACC6", "#D97757"]  # fixed, phase2, phase3 (history highlighted)
    edge = "#333333"

    # (a) Throughput (bars) + #Iter (line, twin y-axis)
    ax = axes[0]
    bars = ax.bar(x, throughput, color=colors, edgecolor=edge, linewidth=0.6, alpha=0.92, label="Throughput")
    ax.set_title("(a) Throughput vs. iterations", fontsize=11, pad=8)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.6)
    ax.set_ylim(0, max(throughput) * 1.22)

    # Keep minimal bar labels (throughput + speedup); no line value labels per request.
    for b, tps, sp in zip(bars, throughput, speedup):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(throughput) * 0.01,
            f"{tps:.1f} ({sp:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_it = ax.twinx()
    ax_it.plot(
        x,
        rounds,
        marker="o",
        linewidth=2.0,
        markersize=7,
        color="#4A708B",
        label="#Iter.",
        alpha=0.95,
    )
    ax_it.set_ylabel("#Iter.", fontsize=11, color="#4A708B")
    ax_it.tick_params(axis="y", labelcolor="#4A708B")
    rmin, rmax = min(rounds), max(rounds)
    pad = max(3, int((rmax - rmin) * 0.2))
    ax_it.set_ylim(rmin - pad, rmax + pad)

    # Combined legend (no duplicate boxes)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_it.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    # (b) Per-iteration progress: tokens/iter (left y) + Accept.(%) (right y)
    ax = axes[1]
    ax.set_title(r"(b) Progress per iteration", fontsize=11, pad=8)
    ax.plot(
        x,
        tokens_per_iter,
        marker="D",
        linewidth=2.2,
        markersize=7,
        color="#D97757",
        label=r"$\bar{L}$ (tokens/iter)",
        alpha=0.95,
    )
    ax.set_ylabel(r"$\bar{L}$ (tokens/iter)", fontsize=11, color="#D97757")
    ax.tick_params(axis="y", labelcolor="#D97757")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    lmin, lmax = min(tokens_per_iter), max(tokens_per_iter)
    ax.set_ylim(lmin - 0.3, lmax + 0.5)

    ax_acc = ax.twinx()
    ax_acc.plot(
        x,
        [a * 100 for a in accept],
        marker="s",
        linewidth=2.0,
        markersize=7,
        color="#4A708B",
        linestyle="--",
        label="Accept. (%)",
        alpha=0.95,
    )
    ax_acc.set_ylabel("Accept. (%)", fontsize=11, color="#4A708B")
    ax_acc.tick_params(axis="y", labelcolor="#4A708B")
    amin, amax = min([a * 100 for a in accept]), max([a * 100 for a in accept])
    # Tight range so acceptance is visible (do NOT share scale with length).
    ax_acc.set_ylim(max(0, amin - 8), min(110, amax + 8))

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_acc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    plt.tight_layout()

    out_pdf = Path("figures/ablation_progression.pdf")
    out_png = Path("figures/ablation_progression.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print("Values:")
    for name, thr, sp, r, tpi, ell, a in zip(labels, throughput, speedup, rounds, tokens_per_iter, avg_path_len, accept):
        print(
            f"  {name:16s} thr={thr:.2f}  sp={sp:.3f}  iters={int(r)}  "
            f"tokens/iter={tpi:.2f}  path={ell:.2f}  accept={a*100:.1f}%"
        )


if __name__ == "__main__":
    main()



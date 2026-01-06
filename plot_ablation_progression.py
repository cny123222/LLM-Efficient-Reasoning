#!/usr/bin/env python3
"""
Plot progressive ablation (Fixed Tree -> + Dynamic Breadth & Depth -> + History Adaptation)
using the paper's main benchmark JSON (WikiText-2, T=1500).

Goal: a compact, multi-metric figure (SpecInfer-style) that explains *why* throughput improves.
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

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    axes = axes.flatten()

    colors = ["#6FAF8A", "#8BACC6", "#D97757"]  # fixed, phase2, phase3 (history highlighted)
    edge = "#333333"

    # (a) Throughput + speedup (annotation)
    ax = axes[0]
    bars = ax.bar(x, throughput, color=colors, edgecolor=edge, linewidth=0.6, alpha=0.92)
    ax.set_title("(a) Throughput", fontsize=11, pad=8)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.6)
    ymax = max(throughput) * 1.18
    ax.set_ylim(0, ymax)
    for b, tps, sp in zip(bars, throughput, speedup):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ymax * 0.02,
            f"{tps:.1f}\n({sp:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # (b) Verification iterations (#Iter)
    ax = axes[1]
    bars = ax.bar(x, rounds, color=colors, edgecolor=edge, linewidth=0.6, alpha=0.92)
    ax.set_title("(b) Verification iterations", fontsize=11, pad=8)
    ax.set_ylabel("#Iter.", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.6)
    ymax = max(rounds) * 1.15
    ax.set_ylim(0, ymax)
    for b, v in zip(bars, rounds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ymax * 0.02,
            f"{int(v)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # (c) Per-iteration progress (tokens/iter and avg path length)
    ax = axes[2]
    ax.set_title(r"(c) Per-iteration progress", fontsize=11, pad=8)
    ax.plot(x, tokens_per_iter, marker="D", linewidth=2.0, markersize=7, color="#D97757", label=r"$\bar{L}$ (tokens/iter)", alpha=0.95)
    ax.plot(x, avg_path_len, marker="o", linewidth=2.0, markersize=7, color="#4A708B", label=r"$\bar{\ell}$ (avg path length)", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)
    for xi, lbar, ell in zip(x, tokens_per_iter, avg_path_len):
        ax.text(xi, lbar + 0.05, f"{lbar:.2f}", ha="center", va="bottom", fontsize=9, color="#D97757")
        ax.text(xi, ell - 0.12, f"{ell:.2f}", ha="center", va="top", fontsize=9, color="#4A708B")

    # (d) Acceptance rate
    ax = axes[3]
    ax.set_title(r"(d) Acceptance rate", fontsize=11, pad=8)
    ax.plot(x, [a * 100 for a in accept], marker="s", linewidth=2.0, markersize=7, color="#6FAF8A", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accept. (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.set_ylim(0, max([a * 100 for a in accept]) * 1.10)
    for xi, a in zip(x, accept):
        ax.text(xi, a * 100 + 0.8, f"{a * 100:.1f}%", ha="center", va="bottom", fontsize=9)

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



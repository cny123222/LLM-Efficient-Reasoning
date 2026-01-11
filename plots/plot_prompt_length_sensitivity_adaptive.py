#!/usr/bin/env python3
"""
Prompt length sensitivity (max prompt length L_max) for the paper.

Uses the new results under:
  results/adaptive/prompts_length/*/results.json

Each subdirectory name is the prompt cap (e.g., 100, 200, ..., 1000).
All runs use T=1500 and report AR / Linear(K=5) / Fixed Tree(D=5,B=2) / DynaTree.
"""

import json
import re
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
    base = Path("results/adaptive/prompts_length")
    files = sorted(
        [p for p in base.glob("*/results.json") if re.fullmatch(r"\d+", p.parent.name)],
        key=lambda p: int(p.parent.name),
    )
    if not files:
        raise FileNotFoundError("No prompt-length results found under results/adaptive/prompts_length/*/results.json")

    Ls = [int(p.parent.name) for p in files]

    methods = [
        ("Baseline (AR)", "AR", "#4A708B", "o"),
        ("Linear Spec (K=5)", "Linear", "#8BACC6", "^"),
        ("Fixed Tree (D=5, B=2)", "Fixed Tree", "#6FAF8A", "s"),
        ("Phase 3: + History Adjust", "DynaTree", "#D97757", "D"),
    ]

    by_L = {}
    for p in files:
        L = int(p.parent.name)
        data = json.loads(p.read_text())
        by_L[L] = {r.get("method"): r for r in data.get("all_results", [])}

    # Extract series
    thr = {k: [] for k, *_ in methods}
    sp = {k: [] for k, *_ in methods}
    acc = {k: [] for k, *_ in methods}

    for L in Ls:
        ar_thr = by_L[L]["Baseline (AR)"]["throughput_tps"]
        for key, _, _, _ in methods:
            r = by_L[L][key]
            t = float(r["throughput_tps"])
            thr[key].append(t)
            sp[key].append(t / ar_thr if ar_thr else 0.0)
            acc[key].append(float(r.get("acceptance_rate", 0.0)) * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

    # (a) Throughput vs prompt cap
    ax = axes[0]
    for key, label, color, marker in methods:
        ax.plot(
            Ls,
            thr[key],
            marker=marker,
            linewidth=2.2 if "History" in key else 1.8,
            markersize=6,
            color=color,
            alpha=0.95,
            label=label,
        )
    ax.set_title("(a) Throughput vs. prompt length", fontsize=11, pad=8)
    ax.set_xlabel(r"Max prompt length $L_{\max}$ (tokens)", fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls], fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    # (b) Speedup vs prompt cap (relative to AR at the same L_max)
    ax = axes[1]
    for key, label, color, marker in methods:
        if key == "Baseline (AR)":
            continue
        ax.plot(
            Ls,
            sp[key],
            marker=marker,
            linewidth=2.2 if "History" in key else 1.8,
            markersize=6,
            color=color,
            alpha=0.95,
            label=label,
        )
    ax.axhline(y=1.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title("(b) Speedup vs. prompt length", fontsize=11, pad=8)
    ax.set_xlabel(r"Max prompt length $L_{\max}$ (tokens)", fontsize=11)
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls], fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    plt.tight_layout()

    out_pdf = Path("figures/prompt_length_sensitivity.pdf")
    out_png = Path("figures/prompt_length_sensitivity.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print("Prompt caps:", Ls)
    for key, label, *_ in methods:
        print(f"{label:10s} throughput:", ", ".join(f"{v:.1f}" for v in thr[key]))
    print("Note: speedup computed as throughput / AR_throughput at the same L_max.")


if __name__ == "__main__":
    main()



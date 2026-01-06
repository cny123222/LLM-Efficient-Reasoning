#!/usr/bin/env python3
"""
Plot 2x2 scaling figure on WikiText-2: throughput + efficiency metrics vs generation length T.

Panels:
  (a) Throughput (tokens/s)
  (b) Tokens per iteration (Lbar)
  (c) Avg path length (ellbar)
  (d) Acceptance rate (%)

Data source (preferred, traceable):
  results/adaptive/scalablity/*/results.json

Output:
  figures/length_scaling_fourpanel.pdf
  figures/length_scaling_fourpanel.png
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


def _load_by_T() -> tuple[list[int], dict[int, dict[str, dict]]]:
    scal_dir = Path("results/adaptive/scalablity")
    per_t = sorted(
        [p for p in scal_dir.glob("*/results.json") if re.fullmatch(r"\d+", p.parent.name)],
        key=lambda p: int(p.parent.name),
    )
    by_T: dict[int, dict[str, dict]] = {}
    if not per_t:
        raise FileNotFoundError("No per-length results found under results/adaptive/scalablity/*/results.json")
    for p in per_t:
        T = int(p.parent.name)
        data = json.loads(p.read_text())
        for r in data.get("all_results", []):
            by_T.setdefault(T, {})[r.get("method")] = r
    lengths = sorted(by_T.keys())
    return lengths, by_T


def main() -> None:
    lengths, by_T = _load_by_T()

    methods = [
        ("Baseline (AR)", "AR", "#4A708B", "o"),
        ("Linear Spec (K=5)", "Linear", "#8BACC6", "^"),
        ("Fixed Tree (D=5, B=2)", "Fixed Tree", "#6FAF8A", "s"),
        ("Phase 3: + History Adjust", "DynaTree", "#D97757", "D"),
    ]

    def series(field: str, method_key: str) -> list[float]:
        out = []
        for T in lengths:
            r = by_T[T].get(method_key) or {}
            out.append(float(r.get(field, 0.0)))
        return out

    throughput = {k: series("throughput_tps", k) for k, _, _, _ in methods}
    lbar = {k: series("tokens_per_round", k) for k, _, _, _ in methods}
    ellbar = {k: series("avg_path_length", k) for k, _, _, _ in methods}
    acc = {k: [a * 100 for a in series("acceptance_rate", k)] for k, _, _, _ in methods}

    # Mask non-meaningful AR efficiency series for plotting clarity.
    for k, _, _, _ in methods:
        if k == "Baseline (AR)":
            ellbar[k] = [np.nan for _ in lengths]
            acc[k] = [np.nan for _ in lengths]

    def fmt_T(t: int) -> str:
        if t >= 1000:
            v = t / 1000.0
            return f"{v:g}k"
        return str(t)

    # Avoid crowded labels in the 1k+ region: keep ticks but hide some labels.
    show_labels = {100, 200, 300, 500, 1000, 1800}
    xticklabels = [fmt_T(t) if t in show_labels else "" for t in lengths]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))

    def common_x(ax):
        ax.set_xscale("log")
        ax.set_xticks(lengths)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)

    # (a) Throughput
    ax = axes[0, 0]
    for key, label, color, marker in methods:
        ax.plot(
            lengths,
            throughput[key],
            marker=marker,
            linewidth=2.0 if "History" in key else 1.8,
            markersize=6,
            color=color,
            alpha=0.95,
            label=label,
        )
    ax.set_title(r"(a) Throughput", fontsize=11, pad=6)
    ax.set_ylabel("Tokens/s", fontsize=11)
    ax.set_xlabel(r"Generation length $T$ (tokens)", fontsize=11)
    common_x(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)
    ymax = max(max(v) for v in throughput.values()) * 1.10
    ymin = min(min(v) for v in throughput.values()) * 0.92
    ax.set_ylim(ymin, ymax)

    # (b) Tokens/iter
    ax = axes[0, 1]
    for key, label, color, marker in methods:
        ax.plot(
            lengths,
            lbar[key],
            marker=marker,
            linewidth=2.0 if "History" in key else 1.8,
            markersize=6,
            color=color,
            alpha=0.95,
        )
    ax.set_title(r"(b) Tokens per iteration ($\bar{L}$)", fontsize=11, pad=6)
    ax.set_ylabel(r"$\bar{L}$ (tokens/iter)", fontsize=11)
    ax.set_xlabel(r"Generation length $T$ (tokens)", fontsize=11)
    common_x(ax)
    ax.set_ylim(0, max(max(v) for v in lbar.values()) * 1.15)

    # (c) Avg path length
    ax = axes[1, 0]
    for key, label, color, marker in methods:
        ax.plot(
            lengths,
            ellbar[key],
            marker=marker,
            linewidth=2.0 if "History" in key else 1.8,
            markersize=6,
            color=color,
            alpha=0.95,
        )
    ax.set_title(r"(c) Avg path length ($\bar{\ell}$)", fontsize=11, pad=6)
    ax.set_ylabel(r"$\bar{\ell}$ (tokens)", fontsize=11)
    ax.set_xlabel(r"Generation length $T$ (tokens)", fontsize=11)
    common_x(ax)
    ymax = np.nanmax([np.nanmax(v) for k, v in ellbar.items() if k != "Baseline (AR)"]) * 1.15
    ax.set_ylim(0, ymax)

    # (d) Acceptance
    ax = axes[1, 1]
    for key, label, color, marker in methods:
        ax.plot(
            lengths,
            acc[key],
            marker=marker,
            linewidth=2.0 if "History" in key else 1.8,
            markersize=6,
            color=color,
            alpha=0.95,
        )
    ax.set_title(r"(d) Acceptance rate", fontsize=11, pad=6)
    ax.set_ylabel("Accept. (%)", fontsize=11)
    ax.set_xlabel(r"Generation length $T$ (tokens)", fontsize=11)
    common_x(ax)
    ax.set_ylim(0, min(110, np.nanmax([np.nanmax(v) for k, v in acc.items() if k != "Baseline (AR)"]) * 1.10))

    plt.tight_layout()

    out_pdf = Path("figures/length_scaling_fourpanel.pdf")
    out_png = Path("figures/length_scaling_fourpanel.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"lengths: {lengths}")
    print("source: results/adaptive/scalablity/*/results.json")


if __name__ == "__main__":
    main()



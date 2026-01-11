#!/usr/bin/env python3
"""
Parameter sensitivity visualization for DynaTree (WikiText-2).

Data source:
  results/adaptive/sensitivity/comprehensive_sensitivity_1500tokens.json

We visualize robustness to three key parameter groups (each as its own panel):
  - confidence thresholds (tau_h, tau_l)
  - branch bounds (Bmin, Bmax)
  - depth ranges (D0, Dmax)

All numbers are real and traceable to the JSON above. Speedup (when shown) is computed
relative to the AR throughput measured in the same sweep run.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

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

    # Baseline for reference
    fixed = next(r for r in rows if r.get("category") == "baseline" and str(r.get("config_name", "")).startswith("Fixed Tree"))
    fixed_thr = float(fixed.get("throughput_tps", 0.0))

    # Build three groups
    thr_rows = [r for r in rows if r.get("category") == "threshold"]
    br_rows = [r for r in rows if r.get("category") == "branch"]
    dp_rows = [r for r in rows if r.get("category") == "depth"]

    # Because the sweep is not a full Cartesian grid, a scatter plot is more faithful
    # for (tau_h, tau_l) than a dense heatmap with many empty cells.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    # (a) threshold scatter: color = throughput
    ax0 = axes[0]
    x = np.array([float(r["config_params"]["high_conf_threshold"]) for r in thr_rows])
    y = np.array([float(r["config_params"]["low_conf_threshold"]) for r in thr_rows])
    c = np.array([float(r.get("throughput_tps", 0.0)) for r in thr_rows])
    # Widen color normalization to reduce perceived contrast between nearby points
    # (keeps mapping stable and avoids "too different" colors for small deltas).
    cmin = float(np.min(c)) if len(c) else 0.0
    cmax = float(np.max(c)) if len(c) else 1.0
    span = max(1e-6, cmax - cmin)
    norm = Normalize(vmin=max(0.0, cmin - 0.35 * span), vmax=cmax + 0.35 * span)
    sc = ax0.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        norm=norm,
        s=75,
        edgecolors="#333333",
        linewidths=0.4,
        alpha=0.95,
    )
    ax0.set_title(r"(a) Thresholds $(\tau_h,\tau_\ell)$", fontsize=11, pad=8)
    ax0.set_xlabel(r"$\tau_h$", fontsize=10)
    ax0.set_ylabel(r"$\tau_\ell$", fontsize=10)
    ax0.grid(True, linestyle=":", alpha=0.55, linewidth=0.8, color="#777777")
    ax0.set_xticks(sorted(set(x.tolist())))
    ax0.set_yticks(sorted(set(y.tolist())))

    # Make the colorbar visually longer/taller (more like a "full" bar).
    cbar0 = fig.colorbar(sc, ax=ax0, fraction=0.055, pad=0.03, shrink=1.05, aspect=28)
    cbar0.set_label("Throughput (tokens/s)", fontsize=10)

    ax0.text(
        0.02,
        0.98,
        f"AR: {ar_thr:.0f} tps\nFixed: {fixed_thr:.0f} tps",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )

    # (b,c) pair-wise errorbars (mean±std), sorted by pair for readability
    def plot_pairs(ax, rows_g, k1, k2, title, xlabel):
        rows_s = sorted(rows_g, key=lambda r: (r["config_params"][k1], r["config_params"][k2]))
        labels = [f"({r['config_params'][k1]},{r['config_params'][k2]})" for r in rows_s]
        yv = np.array([float(r.get("throughput_tps", 0.0)) for r in rows_s])
        ye = np.array([float(r.get("throughput_std", 0.0)) for r in rows_s])
        xx = np.arange(len(rows_s))
        ax.errorbar(
            xx,
            yv,
            yerr=ye,
            fmt="o-",
            color="#4A708B",
            ecolor="#666666",
            elinewidth=0.8,
            capsize=2,
            markersize=4.5,
            linewidth=1.5,
            alpha=0.95,
        )
        ax.axhline(ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7, label="AR (sweep)")
        ax.axhline(fixed_thr, color="#777777", linestyle=":", linewidth=0.9, alpha=0.7, label="Fixed Tree")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Throughput (tokens/s)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35, linewidth=0.6)

        ymin = min(ar_thr, fixed_thr, float(np.min(yv - ye))) * 0.85
        ymax = max(ar_thr, fixed_thr, float(np.max(yv + ye))) * 1.15
        ax.set_ylim(ymin, ymax)

    plot_pairs(axes[1], br_rows, "min_branch", "max_branch", r"(b) Breadth $(B_{\min},B_{\max})$", "Branch bounds")
    plot_pairs(axes[2], dp_rows, "base_depth", "max_depth", r"(c) Depth $(D_0,D_{\max})$", "Depth bounds")

    # Legend once
    handles, labels_ = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=8)

    out_pdf = Path("figures/parameter_sensitivity.pdf")
    out_png = Path("figures/parameter_sensitivity.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print(f"AR throughput (for speedup): {ar_thr:.2f}")
    print(f"threshold configs: {len(thr_rows)}")
    print(f"branch configs: {len(br_rows)}")
    print(f"depth configs: {len(dp_rows)}")


if __name__ == "__main__":
    main()



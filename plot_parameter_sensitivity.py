#!/usr/bin/env python3
"""
Parameter sensitivity visualization for DynaTree (WikiText-2).

Data source:
  results/adaptive/sensitivity/comprehensive_sensitivity_1500tokens.json

We visualize robustness to three key parameter groups (each as its own panel)
as heatmaps of throughput (tokens/s):
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

    # Global color scale for fair visual comparison across panels
    all_thr = [float(r.get("throughput_tps", 0.0)) for r in (thr_rows + br_rows + dp_rows)]
    vmin = min(all_thr) if all_thr else 0.0
    vmax = max(all_thr) if all_thr else 1.0
    # widen slightly for aesthetics
    pad = (vmax - vmin) * 0.06
    vmin = max(0.0, vmin - pad)
    vmax = vmax + pad

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    def heatmap_panel(ax, rows_g, x_key, y_key, title, x_label, y_label, is_float=False):
        # Collect unique coordinates
        xs = sorted({r["config_params"][x_key] for r in rows_g})
        ys = sorted({r["config_params"][y_key] for r in rows_g})

        # Grid (y rows, x cols)
        grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
        for r in rows_g:
            xv = r["config_params"][x_key]
            yv = r["config_params"][y_key]
            xi = xs.index(xv)
            yi = ys.index(yv)
            grid[yi, xi] = float(r.get("throughput_tps", np.nan))

        m = np.ma.masked_invalid(grid)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#F3F3F3")  # untested cells

        im = ax.imshow(
            m,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )

        # Ticks & labels
        ax.set_xticks(np.arange(len(xs)))
        ax.set_yticks(np.arange(len(ys)))
        if is_float:
            ax.set_xticklabels([f"{x:.2g}" for x in xs], fontsize=8)
            ax.set_yticklabels([f"{y:.2g}" for y in ys], fontsize=8)
        else:
            ax.set_xticklabels([str(x) for x in xs], fontsize=8)
            ax.set_yticklabels([str(y) for y in ys], fontsize=8)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(axis="x", rotation=0)

        # Reference throughput annotation (keeps traceability without overlay lines)
        ax.text(
            0.02,
            0.98,
            f"AR: {ar_thr:.0f} tps\nFixed: {fixed_thr:.0f} tps",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
        )

        return im

    im0 = heatmap_panel(
        axes[0],
        thr_rows,
        x_key="high_conf_threshold",
        y_key="low_conf_threshold",
        title=r"(a) Thresholds $(\tau_h,\tau_\ell)$",
        x_label=r"$\tau_h$",
        y_label=r"$\tau_\ell$",
        is_float=True,
    )
    im1 = heatmap_panel(
        axes[1],
        br_rows,
        x_key="max_branch",
        y_key="min_branch",
        title=r"(b) Breadth $(B_{\min},B_{\max})$",
        x_label=r"$B_{\max}$",
        y_label=r"$B_{\min}$",
        is_float=False,
    )
    im2 = heatmap_panel(
        axes[2],
        dp_rows,
        x_key="max_depth",
        y_key="base_depth",
        title=r"(c) Depth $(D_0,D_{\max})$",
        x_label=r"$D_{\max}$",
        y_label=r"$D_0$",
        is_float=False,
    )

    # One shared colorbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Throughput (tokens/s)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.98, 1])

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



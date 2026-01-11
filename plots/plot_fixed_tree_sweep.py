#!/usr/bin/env python3
"""
Fixed Tree hyperparameter sweep visualization (paper protocol).

Data source (current protocol sweep):
  results/adaptive/fixed_tree_sweep/fixed_tree_sweep_*.json

Outputs:
  figures/fixed_tree_sweep.pdf
  figures/fixed_tree_sweep.png
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Academic paper style settings
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
    paths = sorted(glob.glob("results/adaptive/fixed_tree_sweep/fixed_tree_sweep_*.json"))
    if not paths:
        raise FileNotFoundError("No sweep JSON found under results/adaptive/fixed_tree_sweep/")
    src = paths[-1]
    data = json.loads(Path(src).read_text())

    baseline_thr = float(data.get("baseline", {}).get("throughput", 0.0))
    rows = [r for r in data.get("results", []) if r.get("success", True)]
    if not rows:
        raise ValueError("Sweep JSON has no results")

    depths = sorted({int(r["depth"]) for r in rows})
    branches = sorted({int(r["branch"]) for r in rows})
    taus = sorted({float(r["threshold"]) for r in rows})

    def sel(D: int, B: int, tau: float):
        for r in rows:
            if int(r["depth"]) == D and int(r["branch"]) == B and float(r["threshold"]) == float(tau):
                return r
        return None

    def speedup_of(r: dict) -> float:
        bt = float(r.get("baseline_throughput", baseline_thr))
        bt = bt if bt > 0 else baseline_thr
        return float(r["throughput"]) / bt if bt > 0 else float(r.get("speedup", 0.0))

    # Build best-over-tau summaries
    best_speedup = {}  # (D,B) -> (speedup, tau, row)
    for D in depths:
        for B in branches:
            best = None
            for tau in taus:
                r = sel(D, B, tau)
                if r is None:
                    continue
                s = speedup_of(r)
                if (best is None) or (s > best[0]):
                    best = (s, tau, r)
            if best is not None:
                best_speedup[(D, B)] = best

    best_overall = max(best_speedup.values(), key=lambda x: x[0])
    best_tau = float(best_overall[1])
    best_row = best_overall[2]
    best_D = int(best_row["depth"])
    best_B = int(best_row["branch"])

    # Panels (2x3) similar to the reference figure
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2))
    # No star markers (paper robustness presentation); avoid drawing attention to a single point.

    # (a) Best speedup vs depth (fix B=best_B)
    ax = axes[0, 0]
    x = np.array(depths)
    y = np.array([best_speedup[(D, best_B)][0] for D in depths])
    ax.plot(x, y, marker="o", linewidth=2.0, color="#4A708B", alpha=0.9)
    ax.set_title("(a) Speedup vs Depth", fontsize=11, pad=8)
    ax.set_xlabel(r"Tree depth $D$ (fix $B$)", fontsize=10)
    ax.set_ylabel("Speedup", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    # Wider y-range: keep the first point visible while conveying stability.
    ax.set_ylim(min(0.8, float(np.min(y)) * 0.95), max(3.0, float(np.max(y)) * 1.25))

    # (b) Best speedup vs branching factor (fix D=best_D)
    ax = axes[0, 1]
    x = np.array(branches)
    y = np.array([best_speedup[(best_D, B)][0] for B in branches])
    ax.plot(x, y, marker="s", linewidth=2.0, color="#8BACC6", alpha=0.9)
    ax.set_title("(b) Speedup vs Branching Factor", fontsize=11, pad=8)
    ax.set_xlabel(r"Branch factor $B$ (fix $D$)", fontsize=10)
    ax.set_ylabel("Speedup", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.set_ylim(min(0.8, float(np.min(y)) * 0.95), max(3.0, float(np.max(y)) * 1.25))

    # (c) Speedup vs pruning threshold (fix D=best_D,B=best_B)
    ax = axes[0, 2]
    y = np.array([speedup_of(sel(best_D, best_B, tau)) for tau in taus])
    ax.plot(taus, y, marker="^", linewidth=2.0, color="#6495B8", alpha=0.9)
    ax.set_title("(c) Speedup vs Threshold", fontsize=11, pad=8)
    ax.set_xlabel(r"Pruning threshold $\tau$ (log)", fontsize=10)
    ax.set_ylabel("Speedup", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.set_ylim(min(0.8, float(np.min(y)) * 0.95), max(3.0, float(np.max(y)) * 1.25))

    # (d) Heatmap: best speedup over tau for each (D,B)
    ax = axes[1, 0]
    mat = np.zeros((len(depths), len(branches)))
    for i, D in enumerate(depths):
        for j, B in enumerate(branches):
            mat[i, j] = best_speedup[(D, B)][0]
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=np.min(mat), vmax=np.max(mat))
    ax.set_xticks(np.arange(len(branches)))
    ax.set_xticklabels([str(b) for b in branches], fontsize=9)
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(d) for d in depths], fontsize=9)
    ax.set_xlabel(r"$B$", fontsize=10)
    ax.set_ylabel(r"$D$", fontsize=10)
    ax.set_title("(d) Best speedup over $\\tau$", fontsize=11, pad=8)
    # No explicit best-cell highlight (paper presentation focuses on robustness).
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speedup", fontsize=9)

    # (e) Heatmap: avg path length at the best tau (for each D,B)
    ax = axes[1, 1]
    mat2 = np.zeros((len(depths), len(branches)))
    for i, D in enumerate(depths):
        for j, B in enumerate(branches):
            r = best_speedup[(D, B)][2]
            mat2[i, j] = float(r.get("avg_path_length", 0.0))
    im2 = ax.imshow(mat2, cmap="Blues", aspect="auto", vmin=np.min(mat2), vmax=np.max(mat2))
    ax.set_xticks(np.arange(len(branches)))
    ax.set_xticklabels([str(b) for b in branches], fontsize=9)
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(d) for d in depths], fontsize=9)
    ax.set_xlabel(r"$B$", fontsize=10)
    ax.set_ylabel(r"$D$", fontsize=10)
    ax.set_title("(e) Avg path length at best $\\tau$", fontsize=11, pad=8)
    # No explicit best-cell highlight.
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$\bar{\ell}$", fontsize=9)

    # (f) Acceptance rate vs threshold (fix D=best_D,B=best_B)
    ax = axes[1, 2]
    acc = np.array([float(sel(best_D, best_B, tau).get("acceptance_rate", 0.0)) for tau in taus]) * 100.0
    ax.plot(taus, acc, marker="D", linewidth=2.0, color="#4A708B", alpha=0.9)
    ax.set_title("(f) Acceptance vs Threshold", fontsize=11, pad=8)
    ax.set_xlabel(r"Pruning threshold $\tau$ (log)", fontsize=10)
    ax.set_ylabel("Accept. (%)", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.set_ylim(0, 110)

    suptitle = (
        r"Fixed Tree hyperparameter sweep (WikiText-2, $T=1500$, $L_{\max}=800$): "
        + rf"best at $D={best_D}, B={best_B}, \tau={best_tau}$"
    )
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()

    out_pdf = Path("figures/fixed_tree_sweep.pdf")
    out_png = Path("figures/fixed_tree_sweep.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print(f"Best: D={best_D} B={best_B} tau={best_tau} speedup={best_overall[0]:.3f}")


if __name__ == "__main__":
    main()



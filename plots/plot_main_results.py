#!/usr/bin/env python3
"""
Generate grouped bar chart for main results across two datasets (WikiText-2 vs PG-19).
Data sources (source of truth):
  - results/adaptive/main_D8B3/1500_2/results.json        (WikiText-2)
  - results/adaptive/pg19/pg19_benchmark_D8B3.json        (PG-19)

Each subplot shows grouped bars per method with two bars (WikiText-2, PG-19).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Academic paper style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#999999'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

wt_path = Path("results/adaptive/main_D8B3/1500_2/results.json")
pg_path = Path("results/adaptive/pg19/pg19_benchmark_D8B3.json")
wt = json.loads(wt_path.read_text())
pg = json.loads(pg_path.read_text())

wt_rows = wt.get("all_results", [])
pg_rows = pg.get("all_results", [])


def pick(rows: list[dict], *, startswith: str | None = None, equals: str | None = None) -> dict:
    if equals is not None:
        for r in rows:
            if r.get("method") == equals:
                return r
    if startswith is not None:
        for r in rows:
            m = str(r.get("method", ""))
            if m.startswith(startswith):
                return r
    raise KeyError(f"Could not find method (equals={equals}, startswith={startswith}) in rows")


# We keep PG-19 as-is for now (per user request), while WikiText-2 is updated to D8B3.
wt_ar = pick(wt_rows, equals="Baseline (AR)")
wt_linear = pick(wt_rows, startswith="Linear Spec")
wt_fixed = pick(wt_rows, startswith="Fixed Tree")
# D8B3 JSON reports phases; Phase 3 corresponds to the full DynaTree.
wt_dyn = pick(wt_rows, equals="Phase 3: + History Adjust")

pg_ar = pick(pg_rows, equals="Baseline (AR)")
pg_linear = pick(pg_rows, equals="Linear Spec (K=5)")
pg_fixed = pick(pg_rows, startswith="Tree Spec Decode (D=8, B=3")
pg_dyn = pick(pg_rows, equals="Adaptive Tree (Phase 3)")

plot_methods = [
    ("AR", wt_ar, pg_ar),
    ("Linear", wt_linear, pg_linear),
    ("Fixed\nTree", wt_fixed, pg_fixed),
    ("DynaTree", wt_dyn, pg_dyn),
]

wt_throughput = []
pg_throughput = []
wt_speedup = []
pg_speedup = []
labels = []

wt_ar_thr = wt_ar["throughput_tps"]
pg_ar_thr = pg_ar["throughput_tps"]

for label, wt_r, pg_r in plot_methods:
    labels.append(label)
    wt_throughput.append(wt_r["throughput_tps"])
    pg_throughput.append(pg_r["throughput_tps"])
    # Some result files contain a placeholder speedup field; compute from throughputs for consistency.
    wt_speedup.append(wt_r["throughput_tps"] / wt_ar_thr)
    pg_speedup.append(pg_r["throughput_tps"] / pg_ar_thr)

# Set up the figure - grouped bars (two datasets)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

wt_color = "#D97757"   # terra cotta
pg_color = "#6495B8"   # steel blue

# --- Subplot 1: Throughput ---
ax1 = axes[0]
x = np.arange(len(labels))
width = 0.35
bars1 = ax1.bar(x - width / 2, wt_throughput, width, label="WikiText-2", color=wt_color, edgecolor="#333333", linewidth=0.6, alpha=0.9)
bars2 = ax1.bar(x + width / 2, pg_throughput, width, label="PG-19", color=pg_color, edgecolor="#333333", linewidth=0.6, alpha=0.9)
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax1.set_xlabel('Method', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, max(max(wt_throughput), max(pg_throughput)) * 1.18)
ax1.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.5)
# Add value labels
for bar, val in zip(bars1, wt_throughput):
    ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, pg_throughput):
    ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
ax1.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

# --- Subplot 2: Speedup ---
ax2 = axes[1]
bars3 = ax2.bar(x - width / 2, wt_speedup, width, label="WikiText-2", color=wt_color, edgecolor="#333333", linewidth=0.6, alpha=0.9)
bars4 = ax2.bar(x + width / 2, pg_speedup, width, label="PG-19", color=pg_color, edgecolor="#333333", linewidth=0.6, alpha=0.9)
ax2.set_ylabel('Speedup', fontsize=11)
ax2.set_xlabel('Method', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylim(0, max(max(wt_speedup), max(pg_speedup)) * 1.18)
ax2.axhline(y=1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.5)
# Add value labels
for bar, val in zip(bars3, wt_speedup):
    ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{val:.2f}×", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars4, pg_speedup):
    ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

# Save figure
output_path = 'figures/main_results_bars.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {output_path}")

# Also save as PDF for LaTeX
output_pdf = 'figures/main_results_bars.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✓ PDF saved to: {output_pdf}")

plt.close(fig)


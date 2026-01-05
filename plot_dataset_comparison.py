#!/usr/bin/env python3
"""
Create cross-dataset performance comparison (PG-19 vs WikiText-2).
Data sources (source of truth):
  - results/adaptive/main/paper_benchmark_main_with_linear.json
  - results/adaptive/pg19/pg19_benchmark_with_linear.json
Demonstrates robustness across different text domains with identical greedy decoding.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Academic paper style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#999999'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['font.size'] = 10

print("=" * 70)
print("Creating cross-dataset comparison figure (adaptive JSONs)...")
print("=" * 70)

wt_path = Path("results/adaptive/main/paper_benchmark_main_with_linear.json")
pg_path = Path("results/adaptive/pg19/pg19_benchmark_with_linear.json")
wt = json.loads(wt_path.read_text())
pg = json.loads(pg_path.read_text())

def to_map(payload, rename=None):
    rename = rename or {}
    m = {}
    for r in payload["all_results"]:
        m[rename.get(r["method"], r["method"])] = r
    return m

wt_map = to_map(wt, rename={"Phase 3: + History Adjust": "DynaTree"})
pg_map = to_map(pg, rename={"Adaptive Tree (Phase 3)": "DynaTree", "Tree Spec Decode (D=5, B=2)": "Fixed Tree (D=5, B=2)"})

# Methods to compare (display label)
methods = [
    ("Baseline (AR)", "AR"),
    ("Linear Spec (K=5)", "Linear\n(K=5)"),
    ("Fixed Tree (D=5, B=2)", "Fixed\nTree"),
    ("DynaTree", "DynaTree"),
]

# Extract data
pg19_throughput = []
wikitext_throughput = []
pg19_speedup = []
wikitext_speedup = []
labels = []

for method_full, method_short in methods:
    labels.append(method_short)

    pg_r = pg_map.get(method_full)
    wt_r = wt_map.get(method_full)
    if pg_r is None or wt_r is None:
        raise RuntimeError(f"Missing method '{method_full}' in PG or WT JSONs")

    pg19_throughput.append(pg_r["throughput_tps"])
    pg19_speedup.append(pg_r["speedup"])
    wikitext_throughput.append(wt_r["throughput_tps"])
    wikitext_speedup.append(wt_r["speedup"])

print("\nExtracted data (t/s):")
for i, label in enumerate(labels):
    print(f"  {label.replace(chr(10), ' '):<18}: PG-19={pg19_throughput[i]:.1f}, WikiText-2={wikitext_throughput[i]:.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

x = np.arange(len(labels))
width = 0.35

# Color scheme
pg19_color = '#6495B8'     # Light steel blue (long-context)
wikitext_color = '#D97757'  # Terra cotta (standard benchmark)

# Subplot 1: Throughput comparison
bars1 = ax1.bar(x - width/2, pg19_throughput, width, label='PG-19 (long-context)', 
                color=pg19_color, edgecolor='#666666', linewidth=0.5, alpha=0.9)
bars2 = ax1.bar(x + width/2, wikitext_throughput, width, label='WikiText-2 (standard)',
                color=wikitext_color, edgecolor='#666666', linewidth=0.5, alpha=0.9)

ax1.set_xlabel('Method', fontsize=11)
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax1.set_title('(a) Absolute Throughput', fontsize=11, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5, linewidth=0.5, axis='y')
ax1.set_ylim(bottom=0, top=max(max(pg19_throughput), max(wikitext_throughput)) * 1.18)

# Subplot 2: Speedup comparison
bars3 = ax2.bar(x - width/2, pg19_speedup, width, label='PG-19 (long-context)',
                color=pg19_color, edgecolor='#666666', linewidth=0.5, alpha=0.9)
bars4 = ax2.bar(x + width/2, wikitext_speedup, width, label='WikiText-2 (standard)',
                color=wikitext_color, edgecolor='#666666', linewidth=0.5, alpha=0.9)

ax2.set_xlabel('Method', fontsize=11)
ax2.set_ylabel('Speedup (relative to AR)', fontsize=11)
ax2.set_title('(b) Speedup vs. Baseline', fontsize=11, pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1, alpha=0.7)
ax2.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5, linewidth=0.5, axis='y')
ax2.set_ylim(bottom=0.9, top=max(max(pg19_speedup), max(wikitext_speedup)) * 1.12)

plt.tight_layout()

# Save
output_pdf = 'figures/dataset_comparison.pdf'
output_png = 'figures/dataset_comparison.png'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"\n✓ Figure saved to: {output_pdf}")
print(f"✓ Figure saved to: {output_png}")

# Print summary table
plt.close(fig)


#!/usr/bin/env python3
"""
Plot performance across different generation lengths
ALL DATA IS REAL - extracted from teammate's experiments
Data source: results/不同生成token长度性能对比/*.json
"""

import matplotlib.pyplot as plt
import json
import re
from pathlib import Path

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
print("Loading REAL length-scaling data from adaptive experiment logs...")
print("=" * 70)

# Prefer the per-length results directory (newest, most traceable).
SCAL_DIR = Path("results/adaptive/scalablity")
PER_T_FILES = sorted(
    [p for p in SCAL_DIR.glob("*/results.json") if re.fullmatch(r"\d+", p.parent.name)],
    key=lambda p: int(p.parent.name),
)

by_T = {}
if PER_T_FILES:
    for p in PER_T_FILES:
        T = int(p.parent.name)
        data = json.loads(p.read_text())
        for r in data.get("all_results", []):
            by_T.setdefault(T, {})[r.get("method")] = r
    DATA_PATH = "results/adaptive/scalablity/*/results.json"
else:
    # Fallback to the aggregate JSON if per-length files are missing.
    DATA_PATH = "results/adaptive/scalablity/paper_benchmark_scalability_v2.json"
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    for r in data.get("all_results", []):
        cfg = r.get("config") or {}
        T = cfg.get("max_new_tokens")
        if T is None:
            continue
        by_T.setdefault(int(T), {})[r.get("method")] = r

lengths = sorted(by_T.keys())

def _get_thr(method: str):
    return [by_T[T][method]["throughput_tps"] for T in lengths]

baseline_throughput = _get_thr("Baseline (AR)")
linear_throughput = _get_thr("Linear Spec (K=5)")
fixed_tree_throughput = _get_thr("Fixed Tree (D=5, B=2)")
dynatree_throughput = _get_thr("Phase 3: + History Adjust")

print("\nReal data extracted:")
print(f"  lengths: {lengths}")
print("  methods: AR, Linear (K=5), Fixed Tree (D=5,B=2), DynaTree")
print(f"  data source: {DATA_PATH}")

# Create figure - academic paper style with reduced width
fig, ax = plt.subplots(figsize=(7, 4))

# Academic color palette - matching main_results figure
colors = {
    'AR': '#4A708B',          # Steel blue (baseline)
    'Linear': '#8BACC6',      # Sky blue
    'FixedTree': '#6FAF8A',   # Desaturated green
    'DynaTree': '#D97757'     # Terra cotta (highlight)
}

# Plot lines with markers - academic style (all SOLID lines - all real data)
ax.plot(lengths, baseline_throughput, 
        marker='o', linewidth=1.8, markersize=6,
        color=colors['AR'], label='AR (target-only)', linestyle='-', alpha=0.95)

ax.plot(lengths, linear_throughput, 
        marker='^', linewidth=1.8, markersize=6,
        color=colors['Linear'], label='Linear Spec. (K=5)', linestyle='-', alpha=0.95)

ax.plot(lengths, fixed_tree_throughput,
        marker='s', linewidth=1.8, markersize=6,
        color=colors['FixedTree'], label='Fixed Tree (D=5, B=2)', linestyle='-', alpha=0.95)

ax.plot(lengths, dynatree_throughput, 
        marker='D', linewidth=2.2, markersize=7,
        color=colors['DynaTree'], label='DynaTree', linestyle='-', alpha=0.98)

# Formatting
ax.set_xlabel('Generation Length (tokens)', fontsize=11, fontweight='normal')
ax.set_ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='normal')
ax.set_xscale('log')
ax.set_xticks(lengths)
ax.set_xticklabels([str(x) for x in lengths])
ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)

# Set y-axis range (avoid overly compressed curves by not forcing a zero baseline)
ymax = max(max(baseline_throughput), max(linear_throughput), max(fixed_tree_throughput), max(dynatree_throughput))
ymin = min(min(baseline_throughput), min(linear_throughput), min(fixed_tree_throughput), min(dynatree_throughput))
ax.set_ylim(ymin * 0.92, ymax * 1.05)

plt.tight_layout()

# Save
plt.savefig('figures/length_scaling.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/length_scaling.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figures/length_scaling.pdf")
print("✓ Saved: figures/length_scaling.png")

# Display summary
print("\n" + "="*80)
print("Performance Summary Across Generation Lengths (ALL REAL DATA)")
print("="*80)
print(f"{'Length':<10} {'AR':<12} {'Linear K=5':<15} {'Fixed Tree':<15} {'DynaTree':<15} {'Speedup':<8}")
print("-"*80)
for i, length in enumerate(lengths):
    speedup = dynatree_throughput[i] / baseline_throughput[i]
    print(
        f"{length:<10} {baseline_throughput[i]:<12.2f} "
        f"{linear_throughput[i]:<15.2f} {fixed_tree_throughput[i]:<15.2f} "
        f"{dynatree_throughput[i]:<15.2f} {speedup:<8.2f}x"
    )
print("="*80)
print("\n✅ ALL DATA IS REAL - No estimation used")
print(f"Data source: {DATA_PATH}\n")


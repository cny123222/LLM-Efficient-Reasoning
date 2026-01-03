#!/usr/bin/env python3
"""
Plot performance across different generation lengths
"""

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
plt.rcParams['font.size'] = 10

# Data from length_scaling_extracted.json and main results table
lengths = [100, 200, 300, 500, 1000]

# Baseline (AR) throughput at each length (from length_scaling_extracted.json)
baseline_throughput = [105.34, 125.55, 124.22, 123.87, 124.50]

# DynaTree throughput at each length (from length_scaling_extracted.json)
dynatree_throughput = [150.74, 193.21, 199.02, 221.40, 212.30]

# For HF Assisted and Linear, we only have data at 500 tokens
# From main results table (500 tokens):
# - AR: 119.4 tok/s
# - HF Assisted: 161.9 tok/s (1.36x speedup)
# - Linear K=6: 133.1 tok/s (1.11x speedup)
# - DynaTree: 193.4 tok/s (1.62x speedup)

# We'll estimate HF and Linear at other lengths by applying their fixed speedup
# ratios to the baseline throughput at each length
hf_speedup_ratio = 161.9 / 119.4  # ≈ 1.36
linear_speedup_ratio = 133.1 / 119.4  # ≈ 1.11

hf_throughput = [b * hf_speedup_ratio for b in baseline_throughput]
linear_throughput = [b * linear_speedup_ratio for b in baseline_throughput]

# Create figure - academic paper style with reduced width
fig, ax = plt.subplots(figsize=(7, 4))

# Academic color palette - matching main_results figure
colors = {
    'AR': '#4A708B',          # Steel blue (baseline)
    'HF': '#6495B8',          # Light steel blue
    'Linear': '#8BACC6',      # Sky blue
    'DynaTree': '#D97757'     # Terra cotta (highlight)
}

# Plot lines with markers - academic style
ax.plot(lengths, baseline_throughput, 
        marker='o', linewidth=1.8, markersize=6,
        color=colors['AR'], label='AR (target-only)', linestyle='-', alpha=0.85)

ax.plot(lengths, hf_throughput, 
        marker='s', linewidth=1.8, markersize=6,
        color=colors['HF'], label='HF Assisted (est.)', linestyle='--', alpha=0.7)

ax.plot(lengths, linear_throughput, 
        marker='^', linewidth=1.8, markersize=6,
        color=colors['Linear'], label='Linear Spec. (K=6, est.)', linestyle='--', alpha=0.7)

ax.plot(lengths, dynatree_throughput, 
        marker='D', linewidth=2.2, markersize=7,
        color=colors['DynaTree'], label='DynaTree (Ours)', linestyle='-', alpha=0.95)

# Formatting
ax.set_xlabel('Generation Length (tokens)', fontsize=11, fontweight='normal')
ax.set_ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='normal')
ax.set_xscale('log')
ax.set_xticks(lengths)
ax.set_xticklabels(['100', '200', '300', '500', '1000'])
ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)

# Set y-axis range
ax.set_ylim(95, 235)

plt.tight_layout()

# Save
plt.savefig('figures/length_scaling.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/length_scaling.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/length_scaling.pdf")
print("✓ Saved: figures/length_scaling.png")

# Display summary
print("\n" + "="*60)
print("Performance Summary Across Generation Lengths")
print("="*60)
print(f"{'Length':<10} {'AR':<12} {'HF (est.)':<14} {'Linear (est.)':<16} {'DynaTree':<12} {'Speedup':<8}")
print("-"*60)
for i, length in enumerate(lengths):
    speedup = dynatree_throughput[i] / baseline_throughput[i]
    print(f"{length:<10} {baseline_throughput[i]:<12.1f} {hf_throughput[i]:<14.1f} "
          f"{linear_throughput[i]:<16.1f} {dynatree_throughput[i]:<12.1f} {speedup:<8.2f}x")
print("="*60)

plt.show()


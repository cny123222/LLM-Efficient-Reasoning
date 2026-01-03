#!/usr/bin/env python3
"""
Plot performance across different generation lengths
"""

import matplotlib.pyplot as plt
import numpy as np

# Set academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
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

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Color palette (muted academic colors)
colors = {
    'AR': '#5B7C99',          # Blue-grey (baseline)
    'HF': '#7B8D9E',          # Light blue-grey
    'Linear': '#8FA09B',      # Green-grey
    'DynaTree': '#C85A54'     # Muted red (highlight)
}

# Plot lines with markers
ax.plot(lengths, baseline_throughput, 
        marker='o', linewidth=2, markersize=7,
        color=colors['AR'], label='AR (target-only)', linestyle='-')

ax.plot(lengths, hf_throughput, 
        marker='s', linewidth=2, markersize=7,
        color=colors['HF'], label='HF Assisted (est.)', linestyle='--', alpha=0.7)

ax.plot(lengths, linear_throughput, 
        marker='^', linewidth=2, markersize=7,
        color=colors['Linear'], label='Linear Spec. (K=6, est.)', linestyle='--', alpha=0.7)

ax.plot(lengths, dynatree_throughput, 
        marker='D', linewidth=2.5, markersize=8,
        color=colors['DynaTree'], label='DynaTree (Ours)', linestyle='-')

# Formatting
ax.set_xlabel('Generation Length (tokens)', fontsize=12, fontweight='normal')
ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='normal')
ax.set_xscale('log')
ax.set_xticks(lengths)
ax.set_xticklabels(['100', '200', '300', '500', '1000'])
ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#cccccc')

# Set y-axis range
ax.set_ylim(95, 235)

# Add note about estimated data
ax.text(0.98, 0.02, 
        'Note: HF Assisted and Linear throughputs estimated from 500-token measurements\nusing fixed speedup ratios.',
        transform=ax.transAxes,
        fontsize=8, style='italic', color='#666666',
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.8))

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


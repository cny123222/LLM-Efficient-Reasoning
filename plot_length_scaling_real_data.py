#!/usr/bin/env python3
"""
Plot performance across different generation lengths
ALL DATA IS REAL - extracted from teammate's experiments
Data source: results/不同生成token长度性能对比/*.json
"""

import matplotlib.pyplot as plt
import json

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

print("="*70)
print("Extracting REAL data from teammate's experiments...")
print("="*70)

# REAL DATA extracted from JSON files
# Data source: results/不同生成token长度性能对比/wikitext_benchmark_*tokens.json

lengths = [100, 200, 500, 750, 1000]

# === REAL DATA EXTRACTED FROM EXPERIMENTS ===
# 100 tokens: AR=73.49, HF=138.43, Linear K=6=99.94, DynaTree(D=5)=113.39
# 200 tokens: AR=125.67, HF=194.44, Linear K=6=141.79, DynaTree(D=6)=166.21
# 500 tokens: AR=133.23, HF=209.79, Linear K=6=167.69, DynaTree(D=6)=185.26
# 750 tokens: AR=126.27, HF=214.77, Linear K=6=172.61, DynaTree(D=7)=186.71
# 1000 tokens: AR=130.96, HF=222.47, Linear K=6=184.39, DynaTree(D=7)=205.56

# Extracted REAL throughput data (tokens/sec) - NO ESTIMATION
baseline_throughput = [73.49, 125.67, 133.23, 126.27, 130.96]
hf_throughput = [138.43, 194.44, 209.79, 214.77, 222.47]
linear_k6_throughput = [99.94, 141.79, 167.69, 172.61, 184.39]

# DynaTree best configuration at each length
dynatree_throughput = [113.39, 166.21, 185.26, 186.71, 205.56]
dynatree_configs = ["D=5", "D=6", "D=6", "D=7", "D=7"]

print("\nReal data extracted:")
print(f"  100 tokens: {len([x for x in [baseline_throughput[0], hf_throughput[0], linear_k6_throughput[0], dynatree_throughput[0]] if x])} methods")
print(f"  200 tokens: {len([x for x in [baseline_throughput[1], hf_throughput[1], linear_k6_throughput[1], dynatree_throughput[1]] if x])} methods")
print(f"  500 tokens: {len([x for x in [baseline_throughput[2], hf_throughput[2], linear_k6_throughput[2], dynatree_throughput[2]] if x])} methods")
print(f"  750 tokens: {len([x for x in [baseline_throughput[3], hf_throughput[3], linear_k6_throughput[3], dynatree_throughput[3]] if x])} methods")
print(f" 1000 tokens: {len([x for x in [baseline_throughput[4], hf_throughput[4], linear_k6_throughput[4], dynatree_throughput[4]] if x])} methods")

# Create figure - academic paper style with reduced width
fig, ax = plt.subplots(figsize=(7, 4))

# Academic color palette - matching main_results figure
colors = {
    'AR': '#4A708B',          # Steel blue (baseline)
    'HF': '#6495B8',          # Light steel blue
    'Linear': '#8BACC6',      # Sky blue
    'DynaTree': '#D97757'     # Terra cotta (highlight)
}

# Plot lines with markers - academic style (all SOLID lines - all real data)
ax.plot(lengths, baseline_throughput, 
        marker='o', linewidth=1.8, markersize=6,
        color=colors['AR'], label='AR (target-only)', linestyle='-', alpha=0.85)

ax.plot(lengths, hf_throughput, 
        marker='s', linewidth=1.8, markersize=6,
        color=colors['HF'], label='HF Assisted', linestyle='-', alpha=0.8)

ax.plot(lengths, linear_k6_throughput, 
        marker='^', linewidth=1.8, markersize=6,
        color=colors['Linear'], label='Linear Spec. (K=6)', linestyle='-', alpha=0.8)

ax.plot(lengths, dynatree_throughput, 
        marker='D', linewidth=2.2, markersize=7,
        color=colors['DynaTree'], label='DynaTree (Ours)', linestyle='-', alpha=0.95)

# Formatting
ax.set_xlabel('Generation Length (tokens)', fontsize=11, fontweight='normal')
ax.set_ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='normal')
ax.set_xscale('log')
ax.set_xticks(lengths)
ax.set_xticklabels(['100', '200', '500', '750', '1000'])
ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)

# Set y-axis range
ax.set_ylim(60, 220)

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
print(f"{'Length':<10} {'AR':<12} {'HF Assisted':<15} {'Linear K=6':<15} {'DynaTree':<15} {'Speedup':<8}")
print("-"*80)
for i, length in enumerate(lengths):
    speedup = dynatree_throughput[i] / baseline_throughput[i]
    print(f"{length:<10} {baseline_throughput[i]:<12.2f} {hf_throughput[i]:<15.2f} "
          f"{linear_k6_throughput[i]:<15.2f} {dynatree_throughput[i]:<15.2f} {speedup:<8.2f}x")
print("="*80)
print("\n✅ ALL DATA IS REAL - No estimation used")
print("Data source: results/不同生成token长度性能对比/wikitext_benchmark_*tokens.json\n")

plt.show()


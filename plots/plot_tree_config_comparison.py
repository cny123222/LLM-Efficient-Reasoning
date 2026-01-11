#!/usr/bin/env python3
"""
Create a multi-line comparison plot showing impact of tree configurations
Similar style to SpecInfer Figure 10
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Load data
print("📊 Loading parameter sweep data...")
with open('results/tree_param_search_wikitext_20260103_155215.json', 'r') as f:
    data = json.load(f)

results = data['results']
df = pd.DataFrame(results)
print(f"✓ Loaded {len(df)} configurations")

# Create figure with 1x3 subplots (similar to SpecInfer style)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Academic color palette for different lines
colors = {
    2: '#4A708B',   # steel blue
    3: '#8BACC6',   # sky blue  
    4: '#6495B8',   # light steel blue
    5: '#A8C8D8',   # powder blue
    6: '#7B9FB8',   # medium blue
    7: '#5B8AA8',   # darker blue
    8: '#D97757',   # terra cotta (highlight)
}

# Marker styles
markers = {2: 'o', 3: '^', 4: 's', 5: 'D', 6: 'v', 7: '<', 8: '>'}

# =============================================================================
# (a) Throughput vs Branch Factor for different Depths
# =============================================================================
ax1 = axes[0]

# Fix τ=0.03, tokens=500, vary D and B
subset = df[(df['threshold'] == 0.03) & (df['tokens'] == 500)]

depths_to_plot = [4, 5, 6, 7]  # Only these depths have data
for depth in depths_to_plot:
    depth_data = subset[subset['depth'] == depth].sort_values('branch')
    if len(depth_data) > 0:
        color = colors.get(depth, '#666666')
        marker = markers.get(depth, 'o')
        label = f'Depth = {depth}'
        ax1.plot(depth_data['branch'], depth_data['throughput'], 
                marker=marker, linewidth=2, markersize=8, 
                color=color, label=label, alpha=0.85)

ax1.set_xlabel('Branching Factor (B)', fontsize=11)
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax1.set_title('(a) Impact of Branch Factor', fontsize=11, pad=10)
ax1.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
ax1.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)
ax1.set_xticks([2, 3])  # Only 2 and 3 have data

# =============================================================================
# (b) Throughput vs Tree Depth for different Branch Factors
# =============================================================================
ax2 = axes[1]

# Fix τ=0.03, tokens=500, vary D and B
branches_to_plot = [2, 3]  # Only these branches have data
branch_colors = {2: '#4A708B', 3: '#D97757'}

for branch in branches_to_plot:
    branch_data = subset[subset['branch'] == branch].sort_values('depth')
    if len(branch_data) > 0:
        color = branch_colors.get(branch, '#666666')
        marker = markers.get(branch, 'o')
        label = f'Branch = {branch}'
        ax2.plot(branch_data['depth'], branch_data['throughput'], 
                marker=marker, linewidth=2, markersize=8, 
                color=color, label=label, alpha=0.85)

ax2.set_xlabel('Tree Depth (D)', fontsize=11)
ax2.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax2.set_title('(b) Impact of Tree Depth', fontsize=11, pad=10)
ax2.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
ax2.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)
ax2.set_xticks([4, 5, 6, 7])  # Only these depths have data

# =============================================================================
# (c) Throughput vs Pruning Threshold for different Depths
# =============================================================================
ax3 = axes[2]

# Fix B=3, tokens=500, vary D and τ
subset_b3 = df[(df['branch'] == 3) & (df['tokens'] == 500)]

# depths_to_plot already defined above as [4, 5, 6, 7]
for depth in depths_to_plot:
    depth_data = subset_b3[subset_b3['depth'] == depth].sort_values('threshold')
    if len(depth_data) > 0:
        color = colors.get(depth, '#666666')
        marker = markers.get(depth, 'o')
        label = f'Depth = {depth}'
        ax3.plot(depth_data['threshold'], depth_data['throughput'], 
                marker=marker, linewidth=2, markersize=8, 
                color=color, label=label, alpha=0.85)

ax3.set_xlabel('Pruning Threshold (τ)', fontsize=11)
ax3.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax3.set_title('(c) Impact of Pruning Threshold', fontsize=11, pad=10)
ax3.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
ax3.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='#999999', fontsize=9)
ax3.set_xscale('log')

plt.tight_layout()

# Save figure
output_png = 'figures/tree_config_comparison.png'
output_pdf = 'figures/tree_config_comparison.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\n✓ Figure saved to: {output_png}")
print(f"✓ PDF saved to: {output_pdf}")

# Print summary
print("\n" + "="*70)
print("Key Findings from Configuration Analysis")
print("="*70)

# Find optimal configuration
optimal = df[(df['tokens'] == 500)].loc[df[(df['tokens'] == 500)]['throughput'].idxmax()]
print(f"\nOptimal Configuration @ 500 tokens:")
print(f"  Depth (D): {optimal['depth']}")
print(f"  Branch (B): {optimal['branch']}")
print(f"  Threshold (τ): {optimal['threshold']}")
print(f"  Throughput: {optimal['throughput']:.2f} tokens/s")
print(f"  Speedup: {optimal['speedup']:.2f}x")

# Branch factor impact
print(f"\nBranch Factor Impact (D=8, τ=0.03):")
for b in [2, 3, 4]:
    row = df[(df['depth'] == 8) & (df['branch'] == b) & 
             (df['threshold'] == 0.03) & (df['tokens'] == 500)]
    if len(row) > 0:
        print(f"  B={b}: {row.iloc[0]['throughput']:.2f} tokens/s")

# Depth impact
print(f"\nDepth Impact (B=3, τ=0.03):")
for d in [4, 5, 6, 7, 8]:
    row = df[(df['depth'] == d) & (df['branch'] == 3) & 
             (df['threshold'] == 0.03) & (df['tokens'] == 500)]
    if len(row) > 0:
        print(f"  D={d}: {row.iloc[0]['throughput']:.2f} tokens/s")

plt.show()


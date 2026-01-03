#!/usr/bin/env python3
"""
Generate 6-panel parameter sweep visualization for DynaTree paper.
Based on results/tree_param_search_20251231_140952.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
plt.rcParams['font.size'] = 9

# Academic color palette - matching other figures
COLORS = {
    'main': '#4A708B',       # steel blue
    'highlight': '#D97757',  # terra cotta
    'secondary': '#6495B8',  # light steel blue
    'tertiary': '#8BACC6',   # sky blue
}

# Load data
print("📊 Loading parameter sweep data...")
with open('results/tree_param_search_20251231_140952.json', 'r') as f:
    data = json.load(f)

results = data['results']
df = pd.DataFrame(results)

print(f"✓ Loaded {len(df)} configurations")
print(f"  Depths: {sorted(df['depth'].unique())}")
print(f"  Branches: {sorted(df['branch'].unique())}")
print(f"  Thresholds: {sorted(df['threshold'].unique())}")
print(f"  Lengths: {sorted(df['tokens'].unique())}")

# Create figure with 2x3 grid - compact for academic paper
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# =============================================================================
# (a) Speedup vs Depth (fix B=3, τ=0.03, tokens=500)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])
subset = df[(df['branch'] == 3) & (df['threshold'] == 0.03) & (df['tokens'] == 500)]
subset_sorted = subset.sort_values('depth')

ax1.plot(subset_sorted['depth'], subset_sorted['speedup'], 
         marker='o', linewidth=1.8, markersize=6, color=COLORS['main'], alpha=0.85)
ax1.set_xlabel('Tree Depth $D$')
ax1.set_ylabel('Speedup')
ax1.set_title('(a) Speedup vs Depth', fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
ax1.set_xticks(sorted(df['depth'].unique()))
ax1.axhline(y=1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)

# Highlight optimal
if len(subset_sorted) > 0:
    best_idx = subset_sorted['speedup'].idxmax()
    best_row = subset_sorted.loc[best_idx]
    ax1.scatter([best_row['depth']], [best_row['speedup']], 
                color=COLORS['highlight'], s=150, zorder=5, marker='*')

# =============================================================================
# (b) Speedup vs Branch Factor (fix D=8, τ=0.03, tokens=500)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])
subset = df[(df['depth'] == 8) & (df['threshold'] == 0.03) & (df['tokens'] == 500)]
subset_sorted = subset.sort_values('branch')

ax2.plot(subset_sorted['branch'], subset_sorted['speedup'], 
         marker='s', linewidth=1.8, markersize=6, color=COLORS['secondary'], alpha=0.85)
ax2.set_xlabel('Branching Factor $B$')
ax2.set_ylabel('Speedup')
ax2.set_title('(b) Speedup vs Branching Factor', fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
ax2.set_xticks(sorted(df['branch'].unique()))
ax2.axhline(y=1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)

# Highlight optimal
if len(subset_sorted) > 0:
    best_idx = subset_sorted['speedup'].idxmax()
    best_row = subset_sorted.loc[best_idx]
    ax2.scatter([best_row['branch']], [best_row['speedup']], 
                color=COLORS['highlight'], s=150, zorder=5, marker='*')

# =============================================================================
# (c) Speedup vs Threshold (fix D=8, B=3, tokens=500)
# =============================================================================
ax3 = fig.add_subplot(gs[0, 2])
subset = df[(df['depth'] == 8) & (df['branch'] == 3) & (df['tokens'] == 500)]
subset_sorted = subset.sort_values('threshold')

ax3.plot(subset_sorted['threshold'], subset_sorted['speedup'], 
         marker='^', linewidth=1.8, markersize=6, color=COLORS['tertiary'], alpha=0.85)
ax3.set_xlabel('Pruning Threshold $\\tau$')
ax3.set_ylabel('Speedup')
ax3.set_title('(c) Speedup vs Threshold', fontsize=10)
ax3.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
ax3.set_xscale('log')
ax3.axhline(y=1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)

# Highlight optimal
if len(subset_sorted) > 0:
    best_idx = subset_sorted['speedup'].idxmax()
    best_row = subset_sorted.loc[best_idx]
    ax3.scatter([best_row['threshold']], [best_row['speedup']], 
                color=COLORS['highlight'], s=150, zorder=5, marker='*')

# =============================================================================
# (d) Speedup vs Generation Length (best config per length)
# =============================================================================
ax4 = fig.add_subplot(gs[1, 0])
best_per_length = df.loc[df.groupby('tokens')['speedup'].idxmax()]
best_per_length_sorted = best_per_length.sort_values('tokens')

ax4.plot(best_per_length_sorted['tokens'], best_per_length_sorted['speedup'], 
         marker='D', linewidth=1.8, markersize=6, color=COLORS['main'], alpha=0.85)
ax4.set_xlabel('Generation Length (tokens)')
ax4.set_ylabel('Speedup')
ax4.set_title('(d) Speedup vs Generation Length', fontsize=10)
ax4.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
ax4.axhline(y=1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)

# Add configuration annotations
for _, row in best_per_length_sorted.iterrows():
    ax4.annotate(f"D={int(row['depth'])},B={int(row['branch'])}", 
                 xy=(row['tokens'], row['speedup']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=7, alpha=0.7)

# =============================================================================
# (e) Tree Size Heatmap (avg_path_length as proxy, D × B, fix τ=0.03, tokens=500)
# =============================================================================
ax5 = fig.add_subplot(gs[1, 1])
subset = df[(df['threshold'] == 0.03) & (df['tokens'] == 500)]
pivot = subset.pivot_table(values='avg_path_length', index='depth', columns='branch', aggfunc='mean')

im = ax5.imshow(pivot.values, cmap='YlOrRd', aspect='auto', origin='lower', alpha=0.9)
ax5.set_xlabel('Branching Factor $B$')
ax5.set_ylabel('Tree Depth $D$')
ax5.set_title('(e) Avg Path Length (D × B)', fontsize=10)
ax5.set_xticks(range(len(pivot.columns)))
ax5.set_xticklabels(pivot.columns)
ax5.set_yticks(range(len(pivot.index)))
ax5.set_yticklabels(pivot.index)

# Add colorbar
cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Avg Path Length', rotation=270, labelpad=15)

# Add values in cells
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        value = pivot.values[i, j]
        if not np.isnan(value):
            ax5.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color='white' if value > pivot.values.max() * 0.6 else 'black',
                    fontsize=8)

# =============================================================================
# (f) Acceptance Rate Distribution (box plot by threshold, fix D=8, B=3, tokens=500)
# =============================================================================
ax6 = fig.add_subplot(gs[1, 2])
subset = df[(df['depth'] == 8) & (df['branch'] == 3) & (df['tokens'] == 500)]
subset_sorted = subset.sort_values('threshold')

thresholds = sorted(subset['threshold'].unique())
acceptance_data = [subset[subset['threshold'] == t]['acceptance_rate'].values for t in thresholds]

bp = ax6.boxplot(acceptance_data, labels=[f'{t:.2f}' for t in thresholds],
                 patch_artist=True, widths=0.6)

# Style boxplot
for patch in bp['boxes']:
    patch.set_facecolor(COLORS['main'])
    patch.set_alpha(0.6)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=COLORS['main'])

ax6.set_xlabel('Pruning Threshold $\\tau$')
ax6.set_ylabel('Acceptance Rate')
ax6.set_title('(f) Acceptance Rate vs Threshold', fontsize=10)
ax6.grid(True, linestyle=':', alpha=0.3, linewidth=0.5, axis='y')

# Save figure
output_png = 'figures/param_sweep.png'
output_pdf = 'figures/param_sweep.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\n✓ Figure saved to: {output_png}")
print(f"✓ PDF saved to: {output_pdf}")

plt.show()


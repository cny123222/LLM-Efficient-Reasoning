#!/usr/bin/env python3
"""
Create heatmap visualization for tree configuration impact
Shows throughput across different depth and branch factor combinations
"""

import json
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

# Load data
data_file = 'results/tree_param_search_wikitext_20260103_155215.json'
with open(data_file, 'r') as f:
    data = json.load(f)

# Extract 500-token results with tau=0.05 (optimal threshold)
results_500 = [r for r in data['results'] if r['tokens'] == 500 and r['threshold'] == 0.05]

# Get unique depths and branches
depths = sorted(set(r['depth'] for r in results_500))
branches = sorted(set(r['branch'] for r in results_500))

print(f"Depths: {depths}")
print(f"Branches: {branches}")

# Create throughput matrix
throughput_matrix = np.zeros((len(depths), len(branches)))

for r in results_500:
    d_idx = depths.index(r['depth'])
    b_idx = branches.index(r['branch'])
    throughput_matrix[d_idx, b_idx] = r['throughput']

print("\nThroughput Matrix:")
print(throughput_matrix)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Create heatmap
im = ax.imshow(throughput_matrix, cmap='YlOrRd', aspect='auto', 
               vmin=throughput_matrix.min(), vmax=throughput_matrix.max())

# Set ticks and labels
ax.set_xticks(np.arange(len(branches)))
ax.set_yticks(np.arange(len(depths)))
ax.set_xticklabels([f'B={b}' for b in branches])
ax.set_yticklabels([f'D={d}' for d in depths])

# Labels
ax.set_xlabel('Branch Factor', fontsize=12)
ax.set_ylabel('Tree Depth', fontsize=12)
ax.set_title('Throughput (tokens/s) across Tree Configurations\n(500-token generation, τ=0.05)', 
             fontsize=12, pad=15)

# Add text annotations
for i in range(len(depths)):
    for j in range(len(branches)):
        text = ax.text(j, i, f'{throughput_matrix[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Throughput (tokens/s)', rotation=270, labelpad=20, fontsize=11)

# Highlight optimal configuration
optimal_config = max(results_500, key=lambda x: x['throughput'])
optimal_d_idx = depths.index(optimal_config['depth'])
optimal_b_idx = branches.index(optimal_config['branch'])

# Draw rectangle around optimal
from matplotlib.patches import Rectangle
rect = Rectangle((optimal_b_idx - 0.5, optimal_d_idx - 0.5), 1, 1,
                 linewidth=3, edgecolor='#D97757', facecolor='none', linestyle='--')
ax.add_patch(rect)

# Add annotation for optimal
ax.annotate(f'Optimal: D={optimal_config["depth"]}, B={optimal_config["branch"]}\n{optimal_config["throughput"]:.1f} t/s',
            xy=(optimal_b_idx, optimal_d_idx),
            xytext=(optimal_b_idx + 0.7, optimal_d_idx - 1),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#D97757', linewidth=1.5),
            arrowprops=dict(arrowstyle='->', color='#D97757', lw=1.5))

plt.tight_layout()

# Save figure
output_png = 'figures/tree_config_heatmap.png'
output_pdf = 'figures/tree_config_heatmap.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"\n✓ Figure saved to: {output_png}")
print(f"✓ PDF saved to: {output_pdf}")

# Print summary
print("\n" + "="*70)
print("Tree Configuration Heatmap Summary")
print("="*70)
print(f"Optimal configuration: D={optimal_config['depth']}, B={optimal_config['branch']}, τ=0.05")
print(f"Optimal throughput: {optimal_config['throughput']:.1f} tokens/s")
print(f"\nKey observations:")
print(f"- Best depth: D={optimal_config['depth']}")
print(f"- Best branch factor: B={optimal_config['branch']}")
print(f"- Throughput range: {throughput_matrix.min():.1f} - {throughput_matrix.max():.1f} t/s")
print("="*70)

plt.show()


#!/usr/bin/env python3
"""
Create ablation study bar chart visualization
Shows progressive component addition and performance improvement
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

# Data from ablation table (Table 3)
methods = [
    'Linear\nSpeculative\n(K=6)',
    '+ Tree\nStructure\n(D=4,B=3,τ=0.01)',
    '+ Depth &\nPruning Opt.\n(D=8,B=3,τ=0.03)'
]

throughput = [133.1, 176.6, 221.4]  # tokens/sec
speedup = [1.11, 1.43, 1.79]

# Create figure with 2 subplots side by side - compact for paper
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Academic color palette - progressive shading
# From baseline (light) to full system (highlight)
colors = ['#7B9FB8', '#6495B8', '#D97757']  # Light blue → Medium blue → Terra cotta

# =============================================================================
# (a) Throughput comparison
# =============================================================================
ax1 = axes[0]
bars1 = ax1.bar(range(len(methods)), throughput, color=colors, 
                edgecolor='#333333', linewidth=0.8, alpha=0.85, width=0.6)

ax1.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylim(0, max(throughput) * 1.15)
ax1.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.5)
ax1.set_title('(a) Throughput Improvement', fontsize=11, pad=10)

# Add value labels on top of bars and improvement percentage inside
for i, (bar, val) in enumerate(zip(bars1, throughput)):
    height = bar.get_height()
    # Value on top
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='normal')
    # Improvement percentage inside bar (from stage 2 onwards)
    if i > 0:
        improvement = (throughput[i] / throughput[i-1] - 1) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'+{improvement:.1f}%',
                ha='center', va='center', fontsize=9, 
                color='white', fontweight='bold')

# =============================================================================
# (b) Speedup comparison
# =============================================================================
ax2 = axes[1]
bars2 = ax2.bar(range(len(methods)), speedup, color=colors, 
                edgecolor='#333333', linewidth=0.8, alpha=0.85, width=0.6)

ax2.set_ylabel('Speedup (vs AR baseline)', fontsize=11)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylim(0, max(speedup) * 1.15)
ax2.axhline(y=1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.5)
ax2.set_title('(b) Speedup Improvement', fontsize=11, pad=10)

# Add value labels on top of bars and improvement inside
for i, (bar, val) in enumerate(zip(bars2, speedup)):
    height = bar.get_height()
    # Value on top
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
             f'{val:.2f}×',
             ha='center', va='bottom', fontsize=10, fontweight='normal')
    # Improvement inside bar (from stage 2 onwards)
    if i > 0:
        improvement = speedup[i] - speedup[i-1]
        ax2.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'+{improvement:.2f}×',
                ha='center', va='center', fontsize=9, 
                color='white', fontweight='bold')

plt.tight_layout()

# Save figure
output_png = 'figures/ablation_bars.png'
output_pdf = 'figures/ablation_bars.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')

print(f"✓ Figure saved to: {output_png}")
print(f"✓ PDF saved to: {output_pdf}")

# Print summary
print("\n" + "="*70)
print("Ablation Study - Progressive Component Addition")
print("="*70)
print()

for i, (method, thr, spd) in enumerate(zip(methods, throughput, speedup)):
    method_clean = method.replace('\n', ' ')
    print(f"{i+1}. {method_clean}")
    print(f"   Throughput: {thr:.1f} tokens/s")
    print(f"   Speedup: {spd:.2f}×")
    if i > 0:
        thr_gain = (thr / throughput[0] - 1) * 100
        spd_gain = (spd / speedup[0] - 1) * 100
        print(f"   Gain over baseline: +{thr_gain:.1f}% throughput, +{spd_gain:.1f}% speedup")
    print()

print("="*70)
print("Key Findings:")
print("="*70)
print(f"• Tree structure alone: +{(throughput[1]/throughput[0]-1)*100:.1f}% improvement")
print(f"• Full optimization: +{(throughput[2]/throughput[0]-1)*100:.1f}% total improvement")
print(f"• Final speedup: {speedup[2]:.2f}× (vs AR baseline)")
print()

plt.show()


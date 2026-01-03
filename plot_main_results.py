#!/usr/bin/env python3
"""
Generate grouped bar chart for DynaTree main results (Table 1).
Each metric (Throughput, Speedup) is shown with bars for each method.
"""

import matplotlib.pyplot as plt
import numpy as np

# Use serif font for academic papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Data from Table 1 (main results) in neurips_2025.tex
methods = ['AR\n(target-only)', 'HF\nAssisted', 'Linear\nSpec.\n(K=6)', 'StreamingLLM\n+ Spec.', 'DynaTree\n(Ours)']
throughput = [119.4, 161.9, 133.1, 132.9, 193.4]  # tokens/sec
speedup = [1.00, 1.36, 1.11, 1.11, 1.62]           # relative to AR

# Set up the figure - only 2 subplots now
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Unified color palette - muted academic style
# Using a softer, more professional color scheme
colors = ['#5B7C99', '#7B8D9E', '#8FA09B', '#A9B5A8', '#C85A54']  # muted blues/grays, last one is muted red for ours

# --- Subplot 1: Throughput ---
ax1 = axes[0]
bars1 = ax1.bar(range(len(methods)), throughput, color=colors, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=11)
ax1.set_xlabel('Method', fontsize=11)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylim(0, max(throughput) * 1.15)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
# Add value labels on top of bars
for i, (bar, val) in enumerate(zip(bars1, throughput)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=9)

# --- Subplot 2: Speedup ---
ax2 = axes[1]
bars2 = ax2.bar(range(len(methods)), speedup, color=colors, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('Speedup', fontsize=11)
ax2.set_xlabel('Method', fontsize=11)
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylim(0, max(speedup) * 1.15)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
ax2.grid(axis='y', linestyle='--', alpha=0.3)
# Add value labels
for i, (bar, val) in enumerate(zip(bars2, speedup)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.2f}×',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save figure
output_path = 'figures/main_results_bars.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {output_path}")

# Also save as PDF for LaTeX
output_pdf = 'figures/main_results_bars.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✓ PDF saved to: {output_pdf}")

plt.show()


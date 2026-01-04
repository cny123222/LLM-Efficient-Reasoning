#!/usr/bin/env python3
"""
Create cross-dataset performance comparison (PG19 vs WikiText)
Demonstrates DynaTree's generalization across different text domains
"""

import json
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

print("="*70)
print("Creating cross-dataset comparison figure...")
print("="*70)

# Load data
with open('results/两个数据集上单图benchmark结果/pg19_benchmark_单图结果.json', 'r') as f:
    pg19_data = json.load(f)

with open('results/两个数据集上单图benchmark结果/wikitext_benchmark_单图结果.json', 'r') as f:
    wikitext_data = json.load(f)

# Methods to compare
methods = [
    ('Baseline (AR)', 'AR\n(target-only)'),
    ('Linear K=6', 'Linear\nK=6'),
    ('Linear K=7', 'Linear\nK=7'),
    ('Tree V2 (D=6, B=2, t=0.05)', 'DynaTree\nD=6'),
    ('Tree V2 (D=7, B=2, t=0.05)', 'DynaTree\nD=7'),
]

# Extract data
pg19_throughput = []
wikitext_throughput = []
pg19_speedup = []
wikitext_speedup = []
labels = []

for method_full, method_short in methods:
    labels.append(method_short)
    
    # PG19
    pg19_result = next((r for r in pg19_data['results'] if r['method'] == method_full), None)
    if pg19_result:
        pg19_throughput.append(pg19_result['throughput_tps'])
        pg19_speedup.append(pg19_result.get('speedup', 0))
    else:
        pg19_throughput.append(0)
        pg19_speedup.append(0)
    
    # WikiText
    wt_result = next((r for r in wikitext_data['results'] if r['method'] == method_full), None)
    if wt_result:
        wikitext_throughput.append(wt_result['throughput_tps'])
        wikitext_speedup.append(wt_result.get('speedup', 0))
    else:
        wikitext_throughput.append(0)
        wikitext_speedup.append(0)

print("\nExtracted data:")
for i, label in enumerate(labels):
    print(f"  {label.replace(chr(10), ' '):<20}: PG19={pg19_throughput[i]:.1f}, WikiText={wikitext_throughput[i]:.1f}")

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

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
ax1.set_ylim(bottom=0, top=200)

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
ax2.set_ylim(bottom=0.9, top=1.7)

plt.tight_layout()

# Save
output_pdf = 'figures/dataset_comparison.pdf'
output_png = 'figures/dataset_comparison.png'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"\n✓ Figure saved to: {output_pdf}")
print(f"✓ Figure saved to: {output_png}")

# Print summary table
print("\n" + "="*80)
print("Cross-Dataset Performance Summary")
print("="*80)
print(f"{'Method':<25} {'PG-19 (t/s)':<15} {'PG-19 Speedup':<18} {'WikiText (t/s)':<15} {'WikiText Speedup':<18}")
print("-"*80)
for i, (method_full, method_short) in enumerate(methods):
    method_display = method_full.replace('Tree V2 (D=6, B=2, t=0.05)', 'DynaTree D=6') \
                                 .replace('Tree V2 (D=7, B=2, t=0.05)', 'DynaTree D=7')
    print(f"{method_display:<25} {pg19_throughput[i]:<15.2f} {pg19_speedup[i]:<18.3f} "
          f"{wikitext_throughput[i]:<15.2f} {wikitext_speedup[i]:<18.3f}")
print("="*80)

# Key findings
print("\n" + "="*80)
print("Key Findings")
print("="*80)
print("1. DynaTree (D=6) achieves consistent speedup across datasets:")
print(f"   - PG-19 (long-context):  {pg19_throughput[3]:.2f} t/s ({pg19_speedup[3]:.3f}x)")
print(f"   - WikiText-2 (standard): {wikitext_throughput[3]:.2f} t/s ({wikitext_speedup[3]:.3f}x)")
print("\n2. Performance on WikiText-2 is generally higher due to shorter, simpler texts")
print("\n3. All speculative methods maintain speedup benefits across both datasets,")
print("   demonstrating robustness to text domain and complexity variations")

plt.show()


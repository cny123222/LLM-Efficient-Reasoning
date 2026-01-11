#!/usr/bin/env python3
"""
Visualization Script for StreamingLLM Benchmark Results

This script generates comparison charts for StreamingLLM performance analysis.

Usage:
    python scripts/plot_streamingllm_results.py
    
Output:
    - results/streamingllm_comparison.png
    - results/streamingllm_tradeoff.png
    - results/streamingllm_sequence_comparison.png
    - results/streamingllm_vs_others.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set font for better display (fallback for systems without Chinese fonts)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# =============================================================================
# DATA: All benchmark results from streamingllm_result.txt
# =============================================================================

# Short sequence (2024 tokens, 10 samples)
data_2024 = {
    'methods': ['baseline', 'streaming_256', 'streaming_512', 'streaming_1024'],
    'TTFT (s)': [0.0090, 0.0065, 0.0074, 0.0076],
    'TPOT (s)': [0.0059, 0.0074, 0.0071, 0.0074],
    'Throughput': [168.71, 135.37, 140.05, 136.03],
    'PPL': [39.99, 43.06, 41.45, 40.78],
    'Accuracy (%)': [35.49, 34.49, 35.04, 35.31],
}

# Medium sequence (5000 tokens, 5 samples)
data_5000 = {
    'methods': ['baseline', 'streaming_256', 'streaming_512', 'streaming_1024'],
    'TTFT (s)': [0.0163, 0.0131, 0.0138, 0.0140],
    'TPOT (s)': [0.0083, 0.0115, 0.0106, 0.0110],
    'Throughput': [123.30, 90.36, 94.40, 90.58],
    'PPL': [124.28, 168.93, 164.10, 161.96],
    'Accuracy (%)': [26.19, 25.10, 25.46, 25.66],
}

# Long sequence (30000 tokens, 10 samples, 20 warmup)
data_30000 = {
    'methods': ['baseline', 'streaming_256', 'streaming_512', 'streaming_1024'],
    'TTFT (s)': [0.0074, 0.0075, 0.0066, 0.0066],
    'TPOT (s)': [0.0100, 0.0086, 0.0085, 0.0085],
    'Throughput': [96.25, 104.19, 105.72, 106.10],
    'PPL': [1646.10, 2906.07, 2865.64, 2845.13],
    'Accuracy (%)': [10.97, 10.30, 10.43, 10.49],
}

# Color scheme - professional and distinct
colors = {
    'baseline': '#2C3E50',
    'streaming_256': '#E74C3C',
    'streaming_512': '#3498DB',
    'streaming_1024': '#27AE60',
}
color_list = [colors['baseline'], colors['streaming_256'], colors['streaming_512'], colors['streaming_1024']]

# =============================================================================
# Figure 1: Multi-metric comparison bar chart (2024 tokens)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('StreamingLLM Performance Comparison (2024 tokens)\nPythia-70M on PG-19', 
             fontsize=14, fontweight='bold')

methods = data_2024['methods']
metrics = ['TTFT (s)', 'TPOT (s)', 'Throughput', 'PPL', 'Accuracy (%)']
ylabels = ['Time (seconds)', 'Time (seconds)', 'Tokens/sec', 'Perplexity', 'Accuracy (%)']
better_lower = [True, True, False, True, False]  # For annotation

for idx, (metric, ylabel, lower_better) in enumerate(zip(metrics, ylabels, better_lower)):
    ax = axes[idx // 3, idx % 3]
    
    values = data_2024[metric]
    bars = ax.bar(methods, values, color=color_list, edgecolor='black', linewidth=0.5)
    
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', rotation=15, labelsize=8)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}' if val < 0.1 else (f'{val:.3f}' if val < 1 else f'{val:.1f}'),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Highlight best value
    best_idx = np.argmin(values) if lower_better else np.argmax(values)
    bars[best_idx].set_edgecolor('#F39C12')
    bars[best_idx].set_linewidth(2.5)
    
    ax.grid(axis='y', alpha=0.3)

# Remove empty subplot and add legend there
axes[1, 2].axis('off')
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='black') 
                   for c in color_list]
axes[1, 2].legend(legend_elements, methods, loc='center', fontsize=11, 
                  title='Methods', title_fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/streamingllm_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_comparison.png")

# =============================================================================
# Figure 2: Trade-off analysis (Cache Size vs PPL/Accuracy/TTFT) for 2024 tokens
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('StreamingLLM: Cache Size Trade-off Analysis (2024 tokens)', 
             fontsize=13, fontweight='bold')

# Actual cache sizes for streaming methods
cache_sizes = [256, 512, 1024]
stream_ppl = data_2024['PPL'][1:]
stream_acc = data_2024['Accuracy (%)'][1:]
stream_ttft = data_2024['TTFT (s)'][1:]

# Baseline reference
baseline_ppl = data_2024['PPL'][0]
baseline_acc = data_2024['Accuracy (%)'][0]
baseline_ttft = data_2024['TTFT (s)'][0]

# Plot 1: PPL vs Cache Size
ax1 = axes[0]
ax1.plot(cache_sizes, stream_ppl, 'o-', color='#E74C3C', linewidth=2.5, markersize=10, label='StreamingLLM')
ax1.axhline(y=baseline_ppl, color='#2C3E50', linestyle='--', linewidth=2, label=f'Baseline ({baseline_ppl:.1f})')
ax1.fill_between(cache_sizes, baseline_ppl, stream_ppl, alpha=0.15, color='#E74C3C')
ax1.set_xlabel('Cache Size (tokens)', fontsize=11)
ax1.set_ylabel('Perplexity (lower = better)', fontsize=11)
ax1.set_title('Perplexity vs Cache Size', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(cache_sizes)

# Add percentage labels
for x, y in zip(cache_sizes, stream_ppl):
    pct = (y - baseline_ppl) / baseline_ppl * 100
    ax1.annotate(f'+{pct:.1f}%', (x, y), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=9, color='#E74C3C', fontweight='bold')

# Plot 2: Accuracy vs Cache Size
ax2 = axes[1]
ax2.plot(cache_sizes, stream_acc, 'o-', color='#27AE60', linewidth=2.5, markersize=10, label='StreamingLLM')
ax2.axhline(y=baseline_acc, color='#2C3E50', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.1f}%)')
ax2.fill_between(cache_sizes, stream_acc, baseline_acc, alpha=0.15, color='#27AE60')
ax2.set_xlabel('Cache Size (tokens)', fontsize=11)
ax2.set_ylabel('Accuracy % (higher = better)', fontsize=11)
ax2.set_title('Accuracy vs Cache Size', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(cache_sizes)

for x, y in zip(cache_sizes, stream_acc):
    pct = (y - baseline_acc) / baseline_acc * 100
    ax2.annotate(f'{pct:.1f}%', (x, y), textcoords="offset points", 
                 xytext=(0, -15), ha='center', fontsize=9, color='#27AE60', fontweight='bold')

# Plot 3: TTFT vs Cache Size  
ax3 = axes[2]
ax3.plot(cache_sizes, stream_ttft, 'o-', color='#3498DB', linewidth=2.5, markersize=10, label='StreamingLLM')
ax3.axhline(y=baseline_ttft, color='#2C3E50', linestyle='--', linewidth=2, label=f'Baseline ({baseline_ttft:.4f}s)')
ax3.fill_between(cache_sizes, stream_ttft, baseline_ttft, alpha=0.15, color='#3498DB')
ax3.set_xlabel('Cache Size (tokens)', fontsize=11)
ax3.set_ylabel('TTFT (seconds, lower = better)', fontsize=11)
ax3.set_title('TTFT vs Cache Size', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(cache_sizes)

for x, y in zip(cache_sizes, stream_ttft):
    pct = (y - baseline_ttft) / baseline_ttft * 100
    ax3.annotate(f'{pct:.0f}%', (x, y), textcoords="offset points", 
                 xytext=(0, -15), ha='center', fontsize=9, color='#3498DB', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('results/streamingllm_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_tradeoff.png")

# =============================================================================
# Figure 3: Sequence Length Impact Comparison
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('StreamingLLM: Impact of Sequence Length\nComparing 2024, 5000, and 30000 tokens', 
             fontsize=14, fontweight='bold')

seq_lengths = [2024, 5000, 30000]
all_data = [data_2024, data_5000, data_30000]

# Subplot 1: PPL Change (%)
ax1 = axes[0, 0]
ppl_baseline = [d['PPL'][0] for d in all_data]
ppl_256 = [(d['PPL'][1] - d['PPL'][0]) / d['PPL'][0] * 100 for d in all_data]
ppl_512 = [(d['PPL'][2] - d['PPL'][0]) / d['PPL'][0] * 100 for d in all_data]
ppl_1024 = [(d['PPL'][3] - d['PPL'][0]) / d['PPL'][0] * 100 for d in all_data]

x = np.arange(len(seq_lengths))
width = 0.25
bars1 = ax1.bar(x - width, ppl_256, width, label='streaming_256', color=colors['streaming_256'], edgecolor='black')
bars2 = ax1.bar(x, ppl_512, width, label='streaming_512', color=colors['streaming_512'], edgecolor='black')
bars3 = ax1.bar(x + width, ppl_1024, width, label='streaming_1024', color=colors['streaming_1024'], edgecolor='black')

ax1.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax1.set_ylabel('PPL Change from Baseline (%)', fontsize=11)
ax1.set_title('Perplexity Degradation vs Sequence Length', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(seq_lengths)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linewidth=0.8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'+{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

# Subplot 2: Throughput Change (%)
ax2 = axes[0, 1]
thru_256 = [(d['Throughput'][1] - d['Throughput'][0]) / d['Throughput'][0] * 100 for d in all_data]
thru_512 = [(d['Throughput'][2] - d['Throughput'][0]) / d['Throughput'][0] * 100 for d in all_data]
thru_1024 = [(d['Throughput'][3] - d['Throughput'][0]) / d['Throughput'][0] * 100 for d in all_data]

bars1 = ax2.bar(x - width, thru_256, width, label='streaming_256', color=colors['streaming_256'], edgecolor='black')
bars2 = ax2.bar(x, thru_512, width, label='streaming_512', color=colors['streaming_512'], edgecolor='black')
bars3 = ax2.bar(x + width, thru_1024, width, label='streaming_1024', color=colors['streaming_1024'], edgecolor='black')

ax2.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax2.set_ylabel('Throughput Change from Baseline (%)', fontsize=11)
ax2.set_title('Throughput Change vs Sequence Length', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(seq_lengths)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=1.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        label = f'+{height:.1f}%' if height > 0 else f'{height:.1f}%'
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax2.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=7, rotation=45)

# Subplot 3: TTFT Change (%)
ax3 = axes[1, 0]
ttft_256 = [(d['TTFT (s)'][1] - d['TTFT (s)'][0]) / d['TTFT (s)'][0] * 100 for d in all_data]
ttft_512 = [(d['TTFT (s)'][2] - d['TTFT (s)'][0]) / d['TTFT (s)'][0] * 100 for d in all_data]
ttft_1024 = [(d['TTFT (s)'][3] - d['TTFT (s)'][0]) / d['TTFT (s)'][0] * 100 for d in all_data]

bars1 = ax3.bar(x - width, ttft_256, width, label='streaming_256', color=colors['streaming_256'], edgecolor='black')
bars2 = ax3.bar(x, ttft_512, width, label='streaming_512', color=colors['streaming_512'], edgecolor='black')
bars3 = ax3.bar(x + width, ttft_1024, width, label='streaming_1024', color=colors['streaming_1024'], edgecolor='black')

ax3.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax3.set_ylabel('TTFT Change from Baseline (%)', fontsize=11)
ax3.set_title('TTFT Improvement vs Sequence Length', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(seq_lengths)
ax3.legend(loc='lower left', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=0, color='black', linewidth=1.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        label = f'{height:.1f}%'
        va = 'top' if height < 0 else 'bottom'
        offset = -3 if height < 0 else 3
        ax3.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=7, rotation=45)

# Subplot 4: Absolute PPL values (log scale)
ax4 = axes[1, 1]
baseline_ppl = [d['PPL'][0] for d in all_data]
stream_1024_ppl = [d['PPL'][3] for d in all_data]

ax4.semilogy(seq_lengths, baseline_ppl, 'o-', color=colors['baseline'], linewidth=2.5, markersize=10, label='baseline')
ax4.semilogy(seq_lengths, stream_1024_ppl, 's-', color=colors['streaming_1024'], linewidth=2.5, markersize=10, label='streaming_1024')

ax4.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax4.set_ylabel('Perplexity (log scale)', fontsize=11)
ax4.set_title('Absolute PPL: Baseline vs StreamingLLM-1024', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xticks(seq_lengths)
ax4.set_xticklabels(seq_lengths)

# Add annotations
for i, (bl, st) in enumerate(zip(baseline_ppl, stream_1024_ppl)):
    ax4.annotate(f'{bl:.0f}', (seq_lengths[i], bl), textcoords="offset points", 
                 xytext=(-15, 5), ha='center', fontsize=9, color=colors['baseline'])
    ax4.annotate(f'{st:.0f}', (seq_lengths[i], st), textcoords="offset points", 
                 xytext=(15, 5), ha='center', fontsize=9, color=colors['streaming_1024'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/streamingllm_sequence_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_sequence_comparison.png")

# =============================================================================
# Figure 4: Method Comparison (StreamingLLM vs other methods) - 2024 tokens
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Comparison data (from results.txt and current experiments)
comparison_methods = [
    'Baseline',
    'StreamingLLM\n(1024)',
    'StreamingLLM\n(512)',
    'StreamingLLM\n(256)',
    'Fix-Size L2\n(keep_low)',
    'Fix-Size L2\n(random)',
    'Sliding Window\n(recent_only)'
]

ppl_change = [0, 2.0, 3.7, 7.7, 4.3, 7.9, 9.5]  # Percentage change from baseline
acc_change = [0, -0.5, -1.3, -2.8, -1.6, -3.1, -3.6]

x = np.arange(len(comparison_methods))
width = 0.35

# Create gradient colors for bars
ppl_colors = ['#2C3E50', '#27AE60', '#27AE60', '#E67E22', '#F39C12', '#E74C3C', '#C0392B']
acc_colors = ['#2C3E50', '#27AE60', '#27AE60', '#E67E22', '#F39C12', '#E74C3C', '#C0392B']

bars1 = ax.bar(x - width/2, ppl_change, width, label='PPL Change (%)', color='#E74C3C', edgecolor='black', alpha=0.8)
bars2 = ax.bar(x + width/2, acc_change, width, label='Accuracy Change (%)', color='#3498DB', edgecolor='black', alpha=0.8)

ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Change from Baseline (%)', fontsize=12)
ax.set_title('KV Cache Compression Methods Comparison (2024 tokens)\nLower PPL change & Higher Accuracy = Better', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_methods, fontsize=9)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Shade the StreamingLLM region
ax.axvspan(-0.5, 3.5, alpha=0.08, color='green')
ax.text(1.5, ax.get_ylim()[1] * 0.9, 'StreamingLLM', ha='center', fontsize=10, 
        color='#27AE60', fontweight='bold', alpha=0.7)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    label = f'+{height:.1f}%' if height > 0 else f'{height:.1f}%'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    label = f'{height:.1f}%'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -12 if height < 0 else 3), textcoords="offset points",
                ha='center', va='top' if height < 0 else 'bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('results/streamingllm_vs_others.png', dpi=150, bbox_inches='tight')
print("Saved: results/streamingllm_vs_others.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print("\nKey insights from the experiments:")
print("-" * 60)
print("1. Short sequences (2024 tokens):")
print("   - StreamingLLM-1024: PPL +2.0%, Acc -0.5%, TTFT -15.6%")
print("   - Best trade-off among all cache sizes")
print()
print("2. Medium sequences (5000 tokens):")
print("   - PPL degradation increases significantly (+30-36%)")
print("   - TTFT improvement remains (~15-20%)")
print()
print("3. Long sequences (30000 tokens):")
print("   - PPL degrades heavily (+72-77%)")
print("   - BUT throughput IMPROVES by 8-10%!")
print("   - This is StreamingLLM's key advantage for streaming")
print()
print("4. vs Other methods (2024 tokens, cache=1024):")
print("   - StreamingLLM beats L2-based methods")
print("   - Attention sink preservation is effective")
print("="*60)

"""
Paper-Quality Figure Generator for Speculative Decoding Research

This script generates publication-ready figures with:
- Clean, readable labels
- Carefully selected experimental configurations
- Professional styling suitable for academic papers

Usage:
    python plot_paper_figures.py --input benchmark_combined_v2_extended_results.json
"""

import json
import argparse
import numpy as np
from typing import List, Dict

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
except ImportError:
    print("matplotlib required: pip install matplotlib")
    exit(1)


def load_results(json_path: str) -> List[Dict]:
    """Load benchmark results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def filter_key_results(results: List[Dict]) -> Dict:
    """Filter to key experimental configurations for paper."""
    # Group by configuration type
    grouped = {
        'baseline': [],
        'spec_k3': [],
        'spec_k5': [],
        'spec_k7': [],
        'spec_k9': [],
        'stream_128': [],
        'stream_256': [],
        'stream_512': [],
    }
    
    for r in results:
        config = r.get('config', '')
        k_value = r.get('k_value', 0)
        streaming = r.get('streaming', False)
        max_cache = r.get('max_cache_len', 0)
        prompt_type = r.get('prompt_type', 'medium')
        
        # Only use medium prompts for cleaner comparison
        if prompt_type != 'medium':
            continue
            
        if 'Baseline' in config:
            grouped['baseline'].append(r)
        elif streaming:
            if max_cache == 128:
                grouped['stream_128'].append(r)
            elif max_cache == 256:
                grouped['stream_256'].append(r)
            elif max_cache == 512:
                grouped['stream_512'].append(r)
        else:
            if k_value == 3:
                grouped['spec_k3'].append(r)
            elif k_value == 5:
                grouped['spec_k5'].append(r)
            elif k_value == 7:
                grouped['spec_k7'].append(r)
            elif k_value == 9:
                grouped['spec_k9'].append(r)
    
    return grouped


def plot_figure1_throughput_comparison(grouped: Dict, save_path: str):
    """
    Figure 1: Throughput comparison across K values and output lengths.
    Clean bar chart showing speedup vs baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Get unique output lengths
    output_lengths = [100, 300, 500]
    k_values = [3, 5, 7, 9]
    
    # Colors for K values
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    x = np.arange(len(output_lengths))
    width = 0.18
    
    # Get baseline throughputs
    baseline_throughputs = {}
    for r in grouped['baseline']:
        baseline_throughputs[r['max_new_tokens']] = r['throughput_mean']
    
    # Plot bars for each K value
    for i, k in enumerate(k_values):
        key = f'spec_k{k}'
        speedups = []
        for olen in output_lengths:
            matches = [r for r in grouped[key] if r['max_new_tokens'] == olen]
            if matches and olen in baseline_throughputs:
                speedup = matches[0]['throughput_mean'] / baseline_throughputs[olen]
                speedups.append(speedup)
            else:
                speedups.append(0)
        
        bars = ax.bar(x + i * width - 1.5 * width, speedups, width, 
                     label=f'K={k}', color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, speedups):
            if val > 0:
                ax.annotate(f'{val:.2f}x',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Baseline')
    
    ax.set_xlabel('Output Length (tokens)')
    ax.set_ylabel('Speedup (×)')
    ax.set_title('Speculative Decoding Speedup vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(output_lengths)
    ax.legend(loc='upper left', ncol=5, framealpha=0.9)
    ax.set_ylim(0, 3.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 1 saved to: {save_path}")
    plt.close()


def plot_figure2_latency_comparison(grouped: Dict, save_path: str):
    """
    Figure 2: TTFT and TPOT comparison.
    Grouped bar chart for latency metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    k_values = [3, 5, 7, 9]
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # TTFT comparison (left subplot)
    ax1 = axes[0]
    output_len = 500  # Focus on 500 tokens
    
    ttfts = []
    labels = ['Baseline']
    
    # Get baseline TTFT
    baseline_matches = [r for r in grouped['baseline'] if r['max_new_tokens'] == output_len]
    if baseline_matches:
        ttfts.append(baseline_matches[0]['ttft_mean'])
    else:
        ttfts.append(0)
    
    # Get spec decoding TTFTs
    for k in k_values:
        key = f'spec_k{k}'
        matches = [r for r in grouped[key] if r['max_new_tokens'] == output_len]
        if matches:
            ttfts.append(matches[0]['ttft_mean'])
            labels.append(f'K={k}')
        else:
            ttfts.append(0)
            labels.append(f'K={k}')
    
    bar_colors = ['gray'] + colors
    bars1 = ax1.bar(labels, ttfts, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars1, ttfts):
        ax1.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('TTFT (ms)')
    ax1.set_title(f'Time to First Token (Output={output_len} tokens)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # TPOT comparison (right subplot)
    ax2 = axes[1]
    
    tpots = []
    labels = ['Baseline']
    
    # Get baseline TPOT
    if baseline_matches:
        tpots.append(baseline_matches[0]['tpot_mean'])
    else:
        tpots.append(0)
    
    # Get spec decoding TPOTs
    for k in k_values:
        key = f'spec_k{k}'
        matches = [r for r in grouped[key] if r['max_new_tokens'] == output_len]
        if matches:
            tpots.append(matches[0]['tpot_mean'])
            labels.append(f'K={k}')
        else:
            tpots.append(0)
            labels.append(f'K={k}')
    
    bars2 = ax2.bar(labels, tpots, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars2, tpots):
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('TPOT (ms/token)')
    ax2.set_title(f'Time per Output Token (Output={output_len} tokens)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 2 saved to: {save_path}")
    plt.close()


def plot_figure3_acceptance_rate(grouped: Dict, save_path: str):
    """
    Figure 3: Acceptance rate analysis.
    Line plot showing acceptance rate vs K value for different output lengths.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = [3, 5, 7, 9]
    output_lengths = [100, 300, 500]
    markers = ['o', 's', '^']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    for i, olen in enumerate(output_lengths):
        acceptance_rates = []
        for k in k_values:
            key = f'spec_k{k}'
            matches = [r for r in grouped[key] if r['max_new_tokens'] == olen]
            if matches:
                acceptance_rates.append(matches[0]['acceptance_rate'] * 100)
            else:
                acceptance_rates.append(0)
        
        ax.plot(k_values, acceptance_rates, marker=markers[i], markersize=10,
               linewidth=2, label=f'Output={olen}', color=colors[i])
    
    ax.set_xlabel('K Value (Draft Tokens per Round)')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Draft Token Acceptance Rate vs K Value')
    ax.set_xticks(k_values)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3 saved to: {save_path}")
    plt.close()


def plot_figure4_streaming_comparison(grouped: Dict, save_path: str):
    """
    Figure 4: StreamingLLM compression impact.
    Bar chart comparing standard vs streaming with different cache sizes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    output_len = 500
    k_value = 5
    
    # Left: Throughput comparison
    ax1 = axes[0]
    
    configs = ['Standard', 'Stream\n(cache=128)', 'Stream\n(cache=256)', 'Stream\n(cache=512)']
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#2ECC71']
    
    throughputs = []
    compressions = []
    
    # Standard K=5
    matches = [r for r in grouped['spec_k5'] if r['max_new_tokens'] == output_len]
    if matches:
        throughputs.append(matches[0]['throughput_mean'])
        compressions.append(0)
    else:
        throughputs.append(0)
        compressions.append(0)
    
    # Streaming variants
    for cache_size in [128, 256, 512]:
        key = f'stream_{cache_size}'
        matches = [r for r in grouped[key] if r['max_new_tokens'] == output_len and r['k_value'] == k_value]
        if matches:
            throughputs.append(matches[0]['throughput_mean'])
            compressions.append(matches[0].get('compression_count', 0))
        else:
            throughputs.append(0)
            compressions.append(0)
    
    bars1 = ax1.bar(configs, throughputs, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars1, throughputs):
        ax1.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Throughput (tokens/s)')
    ax1.set_title(f'Throughput with StreamingLLM (K={k_value}, Output={output_len})')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Compression events
    ax2 = axes[1]
    
    bars2 = ax2.bar(configs, compressions, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars2, compressions):
        ax2.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Compression Events')
    ax2.set_title(f'KV Cache Compression Events (K={k_value}, Output={output_len})')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 4 saved to: {save_path}")
    plt.close()


def plot_figure5_summary_table(grouped: Dict, save_path: str):
    """
    Figure 5: Summary comparison table as a figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Prepare table data
    output_len = 500
    
    columns = ['Configuration', 'Throughput\n(tokens/s)', 'Speedup', 'TTFT\n(ms)', 
               'TPOT\n(ms/token)', 'Accept\nRate', 'Memory\n(MB)']
    
    data = []
    
    # Baseline
    baseline_matches = [r for r in grouped['baseline'] if r['max_new_tokens'] == output_len]
    if baseline_matches:
        b = baseline_matches[0]
        baseline_tp = b['throughput_mean']
        data.append(['Baseline (FP16)', f"{b['throughput_mean']:.1f}", '1.00×',
                    f"{b['ttft_mean']:.1f}", f"{b['tpot_mean']:.1f}", '-',
                    f"{b['peak_memory_mb']:.0f}"])
    
    # Speculative decoding
    for k in [5, 7, 9]:
        key = f'spec_k{k}'
        matches = [r for r in grouped[key] if r['max_new_tokens'] == output_len]
        if matches:
            r = matches[0]
            speedup = r['throughput_mean'] / baseline_tp if baseline_tp > 0 else 0
            data.append([f'Spec Decode (K={k})', f"{r['throughput_mean']:.1f}",
                        f"{speedup:.2f}×", f"{r['ttft_mean']:.1f}",
                        f"{r['tpot_mean']:.1f}", f"{r['acceptance_rate']*100:.0f}%",
                        f"{r['peak_memory_mb']:.0f}"])
    
    # Streaming K=5
    for cache in [256, 512]:
        key = f'stream_{cache}'
        matches = [r for r in grouped[key] if r['max_new_tokens'] == output_len and r['k_value'] == 5]
        if matches:
            r = matches[0]
            speedup = r['throughput_mean'] / baseline_tp if baseline_tp > 0 else 0
            data.append([f'K=5 + Stream({cache})', f"{r['throughput_mean']:.1f}",
                        f"{speedup:.2f}×", f"{r['ttft_mean']:.1f}",
                        f"{r['tpot_mean']:.1f}", f"{r['acceptance_rate']*100:.0f}%",
                        f"{r['peak_memory_mb']:.0f}"])
    
    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                    cellLoc='center', colColours=['#4ECDC4']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best results
    for i, row in enumerate(data):
        if 'K=9' in row[0]:  # Best throughput
            for j in range(len(columns)):
                table[(i+1, j)].set_facecolor('#E8F8F5')
    
    ax.set_title('Performance Summary (Output=500 tokens, Medium Prompts)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 5 saved to: {save_path}")
    plt.close()


def plot_all_figures(input_json: str, output_dir: str = '.'):
    """Generate all paper figures."""
    results = load_results(input_json)
    grouped = filter_key_results(results)
    
    print(f"\n📊 Generating paper-quality figures...")
    print(f"   Input: {input_json}")
    print(f"   Output directory: {output_dir}")
    
    plot_figure1_throughput_comparison(grouped, f"{output_dir}/paper_fig1_throughput.png")
    plot_figure2_latency_comparison(grouped, f"{output_dir}/paper_fig2_latency.png")
    plot_figure3_acceptance_rate(grouped, f"{output_dir}/paper_fig3_acceptance.png")
    plot_figure4_streaming_comparison(grouped, f"{output_dir}/paper_fig4_streaming.png")
    plot_figure5_summary_table(grouped, f"{output_dir}/paper_fig5_summary.png")
    
    print(f"\n✅ All figures generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality figures")
    parser.add_argument("--input", type=str, default="benchmark_combined_v2_extended_results.json",
                       help="Input JSON file with benchmark results")
    parser.add_argument("--output-dir", type=str, default="papers/figures",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_all_figures(args.input, args.output_dir)


if __name__ == "__main__":
    main()









"""
Detailed Benchmark Visualization for Speculative Decoding

This script generates comprehensive visualizations from benchmark results,
including:
1. Throughput Comparison (Custom vs HF vs Baseline)
2. Speedup vs K value
3. Acceptance Rate vs K value
4. TTFT Comparison
5. TPOT / Latency Comparison
6. Memory Usage Comparison

Usage:
    python plot_detailed_benchmark.py --input benchmark_detailed_results.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_detailed_results(results: dict, output_path: str = "detailed_benchmark.png"):
    """Generate detailed benchmark visualization (2x3 grid)."""
    
    baseline = results["baseline"]
    hf_data = results["huggingface"]
    custom_data = results["custom"]
    
    # Extract K values and metrics
    k_values = sorted(set([d["k_value"] for d in hf_data + custom_data]))
    
    hf_throughput = [next((d["throughput"] for d in hf_data if d["k_value"] == k), 0) for k in k_values]
    custom_throughput = [next((d["throughput"] for d in custom_data if d["k_value"] == k), 0) for k in k_values]
    
    hf_ttft = [next((d["ttft_mean"] for d in hf_data if d["k_value"] == k), 0) for k in k_values]
    custom_ttft = [next((d["ttft_mean"] for d in custom_data if d["k_value"] == k), 0) for k in k_values]
    
    hf_tpot = [next((d["tpot_mean"] for d in hf_data if d["k_value"] == k), 0) for k in k_values]
    custom_tpot = [next((d["tpot_mean"] for d in custom_data if d["k_value"] == k), 0) for k in k_values]
    
    custom_acceptance = [next((d["acceptance_rate"] for d in custom_data if d["k_value"] == k), 0) for k in k_values]
    
    hf_memory = [next((d["peak_memory_mb"] for d in hf_data if d["k_value"] == k), 0) for k in k_values]
    custom_memory = [next((d["peak_memory_mb"] for d in custom_data if d["k_value"] == k), 0) for k in k_values]
    
    baseline_throughput = baseline["throughput"]
    baseline_ttft = baseline["ttft_mean"]
    baseline_tpot = baseline["tpot_mean"]
    baseline_memory = baseline["peak_memory_mb"]
    
    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Color scheme
    colors = {
        'baseline': '#808080',  # Gray
        'hf': '#2196F3',        # Blue
        'custom': '#FF5722'     # Orange
    }
    
    x = np.arange(len(k_values))
    width = 0.35
    
    # ==================== Plot 1: Throughput Comparison ====================
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, hf_throughput, width, label='HuggingFace', color=colors['hf'], alpha=0.8)
    ax1.bar(x + width/2, custom_throughput, width, label='Custom', color=colors['custom'], alpha=0.8)
    ax1.axhline(y=baseline_throughput, color=colors['baseline'], linestyle='--', linewidth=2, 
                label=f'Baseline ({baseline_throughput:.1f})')
    ax1.set_xlabel('K (Draft Tokens)')
    ax1.set_ylabel('Throughput (tokens/s)')
    ax1.set_title('Throughput Comparison\n(Higher is Better)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_values)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (hf_v, custom_v) in enumerate(zip(hf_throughput, custom_throughput)):
        if hf_v > 0:
            ax1.annotate(f'{hf_v:.0f}', (x[i] - width/2, hf_v), ha='center', va='bottom', fontsize=8)
        if custom_v > 0:
            ax1.annotate(f'{custom_v:.0f}', (x[i] + width/2, custom_v), ha='center', va='bottom', fontsize=8)
    
    # ==================== Plot 2: Speedup vs K ====================
    ax2 = axes[0, 1]
    hf_speedup = [t / baseline_throughput for t in hf_throughput]
    custom_speedup = [t / baseline_throughput for t in custom_throughput]
    
    ax2.plot(k_values, hf_speedup, 'o-', color=colors['hf'], linewidth=2, markersize=8, label='HuggingFace')
    ax2.plot(k_values, custom_speedup, 's-', color=colors['custom'], linewidth=2, markersize=8, label='Custom')
    ax2.axhline(y=1.0, color=colors['baseline'], linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax2.set_xlabel('K (Draft Tokens)')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs K Value\n(Higher is Better)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Add value labels
    for i, (hf_v, custom_v) in enumerate(zip(hf_speedup, custom_speedup)):
        ax2.annotate(f'{hf_v:.2f}x', (k_values[i], hf_v), ha='center', va='bottom', fontsize=8)
        ax2.annotate(f'{custom_v:.2f}x', (k_values[i], custom_v), ha='center', va='top', fontsize=8)
    
    # ==================== Plot 3: Acceptance Rate vs K ====================
    ax3 = axes[0, 2]
    bars = ax3.bar(k_values, [r * 100 for r in custom_acceptance], color=colors['custom'], alpha=0.8, edgecolor='black')
    ax3.set_xlabel('K (Draft Tokens)')
    ax3.set_ylabel('Acceptance Rate (%)')
    ax3.set_title('Acceptance Rate vs K Value\n(Custom Implementation)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Add value labels
    for bar, rate in zip(bars, custom_acceptance):
        height = bar.get_height()
        ax3.annotate(f'{rate:.1%}', (bar.get_x() + bar.get_width()/2., height),
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ==================== Plot 4: TTFT Comparison ====================
    ax4 = axes[1, 0]
    ax4.bar(x - width/2, hf_ttft, width, label='HuggingFace', color=colors['hf'], alpha=0.8)
    ax4.bar(x + width/2, custom_ttft, width, label='Custom', color=colors['custom'], alpha=0.8)
    ax4.axhline(y=baseline_ttft, color=colors['baseline'], linestyle='--', linewidth=2, 
                label=f'Baseline ({baseline_ttft:.1f}ms)')
    ax4.set_xlabel('K (Draft Tokens)')
    ax4.set_ylabel('TTFT (ms)')
    ax4.set_title('Time to First Token (TTFT)\n(Lower is Better)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_values)
    ax4.legend(loc='upper right')
    ax4.grid(axis='y', alpha=0.3)
    
    # ==================== Plot 5: TPOT Comparison ====================
    ax5 = axes[1, 1]
    ax5.bar(x - width/2, hf_tpot, width, label='HuggingFace', color=colors['hf'], alpha=0.8)
    ax5.bar(x + width/2, custom_tpot, width, label='Custom', color=colors['custom'], alpha=0.8)
    ax5.axhline(y=baseline_tpot, color=colors['baseline'], linestyle='--', linewidth=2, 
                label=f'Baseline ({baseline_tpot:.1f}ms)')
    ax5.set_xlabel('K (Draft Tokens)')
    ax5.set_ylabel('TPOT (ms)')
    ax5.set_title('Time per Output Token (TPOT)\n(Lower is Better)', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(k_values)
    ax5.legend(loc='upper right')
    ax5.grid(axis='y', alpha=0.3)
    
    # ==================== Plot 6: Memory Usage ====================
    ax6 = axes[1, 2]
    
    # Bar chart for memory by K value
    memory_k = k_values[:3] if len(k_values) > 3 else k_values  # Just show first 3 K values for clarity
    hf_mem_subset = hf_memory[:len(memory_k)]
    custom_mem_subset = custom_memory[:len(memory_k)]
    
    x_mem = np.arange(len(memory_k) + 1)  # +1 for baseline
    width_mem = 0.25
    
    # Create grouped bars
    labels = ['Baseline'] + [f'K={k}' for k in memory_k]
    hf_mem_all = [baseline_memory] + hf_mem_subset
    custom_mem_all = [baseline_memory] + custom_mem_subset
    
    ax6.bar(x_mem - width_mem/2, hf_mem_all, width_mem, label='HuggingFace', color=colors['hf'], alpha=0.8)
    ax6.bar(x_mem + width_mem/2, custom_mem_all, width_mem, label='Custom', color=colors['custom'], alpha=0.8)
    ax6.set_xlabel('Configuration')
    ax6.set_ylabel('Peak VRAM (MB)')
    ax6.set_title('Peak Memory Usage\n(Lower is Better)', fontweight='bold')
    ax6.set_xticks(x_mem)
    ax6.set_xticklabels(labels)
    ax6.legend(loc='upper right')
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (hf_v, custom_v) in enumerate(zip(hf_mem_all, custom_mem_all)):
        ax6.annotate(f'{hf_v:.0f}', (x_mem[i] - width_mem/2, hf_v), ha='center', va='bottom', fontsize=7)
        ax6.annotate(f'{custom_v:.0f}', (x_mem[i] + width_mem/2, custom_v), ha='center', va='bottom', fontsize=7)
    
    # Main title
    plt.suptitle('Speculative Decoding: Detailed Performance Analysis\n(Pythia-2.8B Target + Pythia-70M Draft)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    return fig


def plot_acceptance_trend(results: dict, output_path: str = "acceptance_trend.png"):
    """Plot acceptance rate trend as K increases."""
    custom_data = results["custom"]
    
    k_values = sorted([d["k_value"] for d in custom_data])
    acceptance_rates = [next(d["acceptance_rate"] for d in custom_data if d["k_value"] == k) for k in k_values]
    tokens_per_round = [next(d["tokens_per_round"] for d in custom_data if d["k_value"] == k) for k in k_values]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#FF5722'
    ax1.set_xlabel('K (Draft Tokens)', fontsize=12)
    ax1.set_ylabel('Acceptance Rate (%)', color=color1, fontsize=12)
    line1 = ax1.plot(k_values, [r * 100 for r in acceptance_rates], 'o-', color=color1, 
                     linewidth=2, markersize=10, label='Acceptance Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)
    
    ax2 = ax1.twinx()
    color2 = '#2196F3'
    ax2.set_ylabel('Avg Tokens per Round', color=color2, fontsize=12)
    line2 = ax2.plot(k_values, tokens_per_round, 's--', color=color2, 
                     linewidth=2, markersize=10, label='Tokens/Round')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Acceptance Rate & Tokens per Round vs K Value\n(Custom Implementation)', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Acceptance trend plot saved to: {output_path}")
    
    return fig


def plot_speedup_comparison(results: dict, output_path: str = "speedup_comparison.png"):
    """Plot speedup comparison between HuggingFace and Custom."""
    baseline = results["baseline"]
    hf_data = results["huggingface"]
    custom_data = results["custom"]
    
    k_values = sorted(set([d["k_value"] for d in hf_data + custom_data]))
    
    baseline_throughput = baseline["throughput"]
    hf_speedup = [next((d["throughput"] / baseline_throughput for d in hf_data if d["k_value"] == k), 0) for k in k_values]
    custom_speedup = [next((d["throughput"] / baseline_throughput for d in custom_data if d["k_value"] == k), 0) for k in k_values]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hf_speedup, width, label='HuggingFace', color='#2196F3', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, custom_speedup, width, label='Custom', color='#FF5722', alpha=0.8, edgecolor='black')
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    
    ax.set_xlabel('K (Draft Tokens)', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Speedup Comparison: Custom vs HuggingFace\n(Relative to Baseline)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x', (bar.get_x() + bar.get_width()/2., height),
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x', (bar.get_x() + bar.get_width()/2., height),
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add efficiency ratio text
    ax.text(0.98, 0.02, 
            f'Custom/HF Efficiency: {np.mean([c/h if h > 0 else 0 for c, h in zip(custom_speedup, hf_speedup)]):.1%}',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Speedup comparison plot saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot detailed benchmark results")
    parser.add_argument("--input", type=str, default="benchmark_detailed_results.json",
                        help="Input JSON file with benchmark results")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    print(f"📊 Loading results from: {args.input}")
    results = load_results(args.input)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\n🎨 Generating plots...")
    
    plot_detailed_results(results, str(output_dir / "detailed_benchmark.png"))
    plot_acceptance_trend(results, str(output_dir / "acceptance_trend.png"))
    plot_speedup_comparison(results, str(output_dir / "speedup_comparison.png"))
    
    print("\n✅ All plots generated successfully!")


if __name__ == "__main__":
    main()


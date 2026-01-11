#!/usr/bin/env python3
"""
Visualization script for Head-Aware Attention Mask Ablation Study

Generates charts comparing different compression strategies.

Usage:
    python scripts/plot_ablation_results.py \
        --input results/ablation_study/ablation_final.json \
        --output results/ablation_study/figures/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(json_path):
    """Load results from JSON file or extract JSON from log file."""
    with open(json_path, 'r') as f:
        content = f.read()
    
    # Try to parse as pure JSON first
    try:
        data = json.loads(content)
        return data['results']
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from log file (look for { "config": ... } pattern)
    import re
    json_match = re.search(r'(\{\s*"config"[\s\S]*\})\s*$', content)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data['results']
        except json.JSONDecodeError:
            pass
    
    print(f"Warning: Could not parse {json_path}")
    return []


def plot_ppl_vs_context(results, output_dir):
    """Plot PPL vs Effective Context for all groups."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group colors
    colors = {
        'A_window_only': '#e74c3c',      # Red
        'B_streaming': '#2ecc71',         # Green
        'G_confidence_sweep': '#3498db',  # Blue
        'I_inverse_positional': '#9b59b6', # Purple
        'J_small_gathering': '#f39c12',   # Orange
        'K_uniform_aware': '#1abc9c',     # Teal
    }
    
    # Group labels
    labels = {
        'A_window_only': 'A: Window-only (无sink)',
        'B_streaming': 'B: StreamingLLM (基线)',
        'G_confidence_sweep': 'G: Confidence-based',
        'I_inverse_positional': 'I: Inverse (大positional)',
        'J_small_gathering': 'J: Small gathering',
        'K_uniform_aware': 'K: Uniform-aware',
    }
    
    # Plot each group
    for group in colors.keys():
        group_results = [r for r in results if r.get('group') == group and 'error' not in r]
        if not group_results:
            continue
        
        contexts = [r['effective_context'] for r in group_results]
        ppls = [r['perplexity'] for r in group_results]
        
        ax.scatter(contexts, ppls, c=colors[group], label=labels.get(group, group), 
                   s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend line for B_streaming
        if group == 'B_streaming':
            sorted_data = sorted(zip(contexts, ppls))
            ax.plot([x[0] for x in sorted_data], [x[1] for x in sorted_data], 
                   c=colors[group], linestyle='--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Effective Context (tokens)', fontsize=12)
    ax.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax.set_title('PPL vs Effective Context: 不同压缩策略对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(40, 550)
    ax.set_ylim(8, 40)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ppl_vs_context.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'ppl_vs_context.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: ppl_vs_context.png")


def plot_sink_importance(results, output_dir):
    """Plot showing importance of sink tokens."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get A and B groups
    a_results = {r['effective_context']: r['perplexity'] 
                 for r in results if r.get('group') == 'A_window_only'}
    b_results = {r['effective_context']: r['perplexity'] 
                 for r in results if r.get('group') == 'B_streaming'}
    
    contexts = sorted(set(a_results.keys()) & set(b_results.keys()))
    
    x = np.arange(len(contexts))
    width = 0.35
    
    a_ppls = [a_results[c] for c in contexts]
    b_ppls = [b_results[c] for c in contexts]
    
    bars1 = ax.bar(x - width/2, a_ppls, width, label='无 Sink Tokens', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, b_ppls, width, label='有 Sink Tokens (StreamingLLM)', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Effective Context (tokens)', fontsize=12)
    ax.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax.set_title('Sink Tokens 的重要性', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(c)}' for c in contexts])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (a, b) in enumerate(zip(a_ppls, b_ppls)):
        pct = (a - b) / b * 100
        ax.annotate(f'+{pct:.0f}%', xy=(x[i] - width/2, a), ha='center', va='bottom', fontsize=9, color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sink_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sink_importance.png")


def plot_confidence_threshold(results, output_dir):
    """Plot effect of confidence threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get G group results with base128
    g_base128 = [r for r in results if r.get('group') == 'G_confidence_sweep' 
                 and 'base128' in r['name'] and 'error' not in r]
    g_base64 = [r for r in results if r.get('group') == 'G_confidence_sweep' 
                and 'base64' in r['name'] and 'error' not in r]
    
    # Extract threshold from name
    def get_threshold(name):
        parts = name.split('_')
        for p in parts:
            if p.startswith('conf'):
                return float(p.replace('conf', ''))
        return 0
    
    # Plot base128
    if g_base128:
        thresholds = [get_threshold(r['name']) for r in g_base128]
        ppls = [r['perplexity'] for r in g_base128]
        contexts = [r['effective_context'] for r in g_base128]
        
        # Sort by threshold
        sorted_data = sorted(zip(thresholds, ppls, contexts))
        thresholds = [x[0] for x in sorted_data]
        ppls = [x[1] for x in sorted_data]
        contexts = [x[2] for x in sorted_data]
        
        ax1.plot(thresholds, ppls, 'o-', color='#3498db', linewidth=2, markersize=8, label='PPL')
        ax1.axhline(y=9.52, color='#2ecc71', linestyle='--', label='StreamingLLM-128 (9.52)')
        ax1.set_xlabel('Confidence Threshold', fontsize=12)
        ax1.set_ylabel('Perplexity (PPL)', fontsize=12)
        ax1.set_title('Base Window = 128', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Secondary y-axis for context
        ax1_ctx = ax1.twinx()
        ax1_ctx.plot(thresholds, contexts, 's--', color='#f39c12', alpha=0.7, label='Eff. Context')
        ax1_ctx.set_ylabel('Effective Context', fontsize=12, color='#f39c12')
        ax1_ctx.tick_params(axis='y', labelcolor='#f39c12')
    
    # Plot base64
    if g_base64:
        thresholds = [get_threshold(r['name']) for r in g_base64]
        ppls = [r['perplexity'] for r in g_base64]
        contexts = [r['effective_context'] for r in g_base64]
        
        sorted_data = sorted(zip(thresholds, ppls, contexts))
        thresholds = [x[0] for x in sorted_data]
        ppls = [x[1] for x in sorted_data]
        contexts = [x[2] for x in sorted_data]
        
        ax2.plot(thresholds, ppls, 'o-', color='#3498db', linewidth=2, markersize=8, label='PPL')
        ax2.axhline(y=10.57, color='#2ecc71', linestyle='--', label='StreamingLLM-64 (10.57)')
        ax2.set_xlabel('Confidence Threshold', fontsize=12)
        ax2.set_ylabel('Perplexity (PPL)', fontsize=12)
        ax2.set_title('Base Window = 64', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax2_ctx = ax2.twinx()
        ax2_ctx.plot(thresholds, contexts, 's--', color='#f39c12', alpha=0.7, label='Eff. Context')
        ax2_ctx.set_ylabel('Effective Context', fontsize=12, color='#f39c12')
        ax2_ctx.tick_params(axis='y', labelcolor='#f39c12')
    
    fig.suptitle('Confidence Threshold 对 PPL 的影响', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_threshold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: confidence_threshold.png")


def plot_best_configs(results, output_dir):
    """Plot comparison of best configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Select best configs
    best_configs = [
        ('B_streaming_512', 'StreamingLLM-512'),
        ('B_streaming_256', 'StreamingLLM-256'),
        ('I_pos256_mix128_g128', 'Inverse (最佳HA)'),
        ('B_streaming_128', 'StreamingLLM-128'),
        ('G_conf0.6_base128', 'Conf0.6-base128'),
        ('K_uniform128_aware', 'Uniform128-aware'),
        ('B_streaming_64', 'StreamingLLM-64'),
        ('K_uniform64_aware', 'Uniform64-aware'),
        ('G_conf0.7_base64', 'Conf0.7-base64'),
    ]
    
    results_dict = {r['name']: r for r in results if 'error' not in r}
    
    names = []
    ppls = []
    contexts = []
    colors_list = []
    
    for config_name, display_name in best_configs:
        if config_name in results_dict:
            r = results_dict[config_name]
            names.append(display_name)
            ppls.append(r['perplexity'])
            contexts.append(r['effective_context'])
            
            # Color based on group
            if 'streaming' in config_name.lower():
                colors_list.append('#2ecc71')
            elif config_name.startswith('I_'):
                colors_list.append('#9b59b6')
            elif config_name.startswith('G_'):
                colors_list.append('#3498db')
            elif config_name.startswith('K_'):
                colors_list.append('#1abc9c')
            else:
                colors_list.append('#95a5a6')
    
    # Create bar chart
    x = np.arange(len(names))
    bars = ax.bar(x, ppls, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add context labels on top of bars
    for i, (ppl, ctx) in enumerate(zip(ppls, contexts)):
        ax.annotate(f'ctx={int(ctx)}', xy=(i, ppl), ha='center', va='bottom', 
                   fontsize=8, color='#666')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax.set_title('最佳配置对比 (按PPL排序)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='StreamingLLM'),
        Patch(facecolor='#9b59b6', label='Inverse'),
        Patch(facecolor='#3498db', label='Confidence-based'),
        Patch(facecolor='#1abc9c', label='Uniform-aware'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_configs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: best_configs.png")


def plot_inverse_experiment(results, output_dir):
    """Plot inverse experiment results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get I group and B group for comparison
    i_results = [r for r in results if r.get('group') == 'I_inverse_positional' and 'error' not in r]
    b_results = [r for r in results if r.get('group') == 'B_streaming' and 'error' not in r]
    
    if not i_results:
        return
    
    # Sort by context
    i_results = sorted(i_results, key=lambda x: x['effective_context'])
    b_results = sorted(b_results, key=lambda x: x['effective_context'])
    
    # Plot StreamingLLM as reference line
    b_contexts = [r['effective_context'] for r in b_results]
    b_ppls = [r['perplexity'] for r in b_results]
    ax.plot(b_contexts, b_ppls, 'o-', color='#2ecc71', linewidth=2, markersize=10, 
            label='StreamingLLM (基线)', alpha=0.8)
    
    # Plot Inverse results
    i_contexts = [r['effective_context'] for r in i_results]
    i_ppls = [r['perplexity'] for r in i_results]
    ax.scatter(i_contexts, i_ppls, c='#9b59b6', s=150, label='Inverse (大positional)', 
               alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
    
    # Annotate inverse points
    for r in i_results:
        label = r['name'].replace('I_', '').replace('_', '\n')
        ax.annotate(label, xy=(r['effective_context'], r['perplexity']), 
                   xytext=(10, -10), textcoords='offset points', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Effective Context (tokens)', fontsize=12)
    ax.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax.set_title('反向实验: 给 Positional Heads 更大窗口', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inverse_experiment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: inverse_experiment.png")


def plot_summary_table(results, output_dir):
    """Create a summary table image."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare data
    headers = ['配置', 'PPL', 'Accuracy', 'Eff.Ctx', 'vs StreamingLLM']
    
    # Get key results
    results_dict = {r['name']: r for r in results if 'error' not in r}
    
    rows = []
    comparisons = [
        ('B_streaming_512', 'baseline'),
        ('B_streaming_256', 'baseline'),
        ('I_pos256_mix128_g128', 'I组最佳'),
        ('B_streaming_128', 'baseline'),
        ('G_conf0.8_base128', 'G组最佳'),
        ('K_uniform128_aware', 'K组'),
        ('B_streaming_64', 'baseline'),
        ('K_uniform64_aware', 'K组最佳'),
        ('G_conf0.7_base64', 'G组'),
    ]
    
    streaming_baselines = {
        512: 8.81,
        256: 9.14,
        128: 9.52,
        64: 10.57,
    }
    
    for name, note in comparisons:
        if name not in results_dict:
            continue
        r = results_dict[name]
        ctx = r['effective_context']
        ppl = r['perplexity']
        acc = r['accuracy']
        
        # Find closest streaming baseline
        closest_ctx = min(streaming_baselines.keys(), key=lambda x: abs(x - ctx))
        baseline_ppl = streaming_baselines[closest_ctx]
        
        if 'streaming' in name.lower():
            vs_streaming = 'baseline'
        else:
            diff = (ppl - baseline_ppl) / baseline_ppl * 100
            vs_streaming = f'{diff:+.1f}%' if diff != 0 else '0%'
        
        rows.append([
            name,
            f'{ppl:.2f}',
            f'{acc:.1%}',
            f'{ctx:.0f}',
            vs_streaming
        ])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style rows
    for i in range(1, len(rows) + 1):
        if 'streaming' in rows[i-1][0].lower():
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#e8f8f5')
    
    ax.set_title('实验结果汇总', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--input", type=str, nargs='+', 
                       default=["results/ablation_study/ablation_final.json"],
                       help="Input JSON file(s)")
    parser.add_argument("--output", type=str, 
                       default="results/ablation_study/figures/",
                       help="Output directory for figures")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and merge results from all input files
    all_results = []
    for input_file in args.input:
        if os.path.exists(input_file):
            results = load_results(input_file)
            all_results.extend(results)
            print(f"Loaded {len(results)} results from {input_file}")
    
    if not all_results:
        print("No results found!")
        return
    
    # Remove duplicates (keep last occurrence)
    seen = {}
    for r in all_results:
        seen[r['name']] = r
    all_results = list(seen.values())
    print(f"Total unique results: {len(all_results)}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_ppl_vs_context(all_results, args.output)
    plot_sink_importance(all_results, args.output)
    plot_confidence_threshold(all_results, args.output)
    plot_best_configs(all_results, args.output)
    plot_inverse_experiment(all_results, args.output)
    plot_summary_table(all_results, args.output)
    
    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Validate Head Classification Results

This script validates the refined head classification by:
1. Visualizing attention patterns for representative heads of each type
2. Comparing classification accuracy by examining example heads
3. Generating detailed reports for sink-positional vs true-positional distinction

Usage:
    # Validate using existing analysis results
    python scripts/validate_head_classification.py --results_dir results/attention_analysis_pythia-2.8b
    
    # Re-run classification with updated analyzer and visualize
    python scripts/validate_head_classification.py --model_id EleutherAI/pythia-2.8b --reclassify
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_analysis_results(results_dir: str) -> Tuple[dict, dict]:
    """Load existing analysis results from directory."""
    stats_path = os.path.join(results_dir, "head_statistics.json")
    class_path = os.path.join(results_dir, "head_classifications.json")
    
    with open(stats_path, 'r') as f:
        stats_data = json.load(f)
    
    with open(class_path, 'r') as f:
        class_data = json.load(f)
    
    return stats_data, class_data


def reclassify_heads(stats_data: dict) -> List[dict]:
    """
    Re-classify heads using the updated classification logic.
    
    This uses the refined classification that distinguishes:
    - SINK_POSITIONAL: Low entropy + high sink ratio
    - TRUE_POSITIONAL: Low entropy + low sink + high local
    - SINK_MIXED: Medium entropy + high sink
    - GATHERING: High entropy
    - DEAD: Near-uniform
    - MIXED: Everything else
    """
    # Classification thresholds
    LOW_ENTROPY_THRESHOLD = 1.5
    HIGH_ENTROPY_THRESHOLD = 3.0
    DEAD_UNIFORMITY_THRESHOLD = 0.1
    HIGH_LOCAL_THRESHOLD = 0.6
    HIGH_SINK_THRESHOLD = 0.3
    
    new_classifications = []
    
    for stats in stats_data['head_statistics']:
        layer_idx = stats['layer_idx']
        head_idx = stats['head_idx']
        mean_entropy = stats['mean_entropy']
        sink_ratio = stats['sink_ratio']
        local_ratio = stats['position_preference']['local']
        uniformity = stats['uniformity_score']
        rel_pos_dist = stats['relative_position_dist']
        
        # Default values
        head_type = "mixed"
        confidence = 0.5
        can_prune = False
        can_limit_window = False
        recommended_window = -1
        keep_sinks = False
        sink_size = 4
        use_full_cache = True
        compression_strategy = "none"
        
        # Classification logic
        if uniformity < DEAD_UNIFORMITY_THRESHOLD:
            head_type = "dead"
            confidence = 1.0 - uniformity / DEAD_UNIFORMITY_THRESHOLD
            can_prune = True
            use_full_cache = False
            compression_strategy = "prune"
            
        elif mean_entropy < LOW_ENTROPY_THRESHOLD:
            if sink_ratio > HIGH_SINK_THRESHOLD:
                head_type = "sink_positional"
                confidence = sink_ratio
                can_limit_window = True
                keep_sinks = True
                sink_size = 4
                recommended_window = 8
                use_full_cache = False
                compression_strategy = "sink_window"
                
            elif local_ratio > HIGH_LOCAL_THRESHOLD:
                head_type = "true_positional"
                confidence = (1.0 - mean_entropy / LOW_ENTROPY_THRESHOLD) * \
                            (local_ratio / HIGH_LOCAL_THRESHOLD)
                confidence = min(1.0, confidence)
                can_limit_window = True
                keep_sinks = False
                use_full_cache = False
                compression_strategy = "window_only"
                
                # Calculate window from cumsum
                cumsum = 0
                for i, val in enumerate(rel_pos_dist[:16]):
                    cumsum += val
                    if cumsum > 0.8:
                        recommended_window = max(8, i + 4)
                        break
                if recommended_window == -1:
                    recommended_window = 16
                    
        elif mean_entropy < HIGH_ENTROPY_THRESHOLD and sink_ratio > HIGH_SINK_THRESHOLD:
            head_type = "sink_mixed"
            entropy_ratio = (mean_entropy - LOW_ENTROPY_THRESHOLD) / \
                           (HIGH_ENTROPY_THRESHOLD - LOW_ENTROPY_THRESHOLD)
            confidence = sink_ratio * (1.0 - entropy_ratio)
            confidence = min(1.0, max(0.0, confidence))
            can_limit_window = True
            keep_sinks = True
            sink_size = 4
            recommended_window = int(16 + entropy_ratio * 16)
            use_full_cache = False
            compression_strategy = "sink_window"
            
        elif mean_entropy > HIGH_ENTROPY_THRESHOLD:
            head_type = "gathering"
            confidence = min(1.0, (mean_entropy - HIGH_ENTROPY_THRESHOLD) / 2.0)
            use_full_cache = True
            compression_strategy = "full"
        
        new_classifications.append({
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "head_type": head_type,
            "confidence": confidence,
            "can_prune": can_prune,
            "can_limit_window": can_limit_window,
            "recommended_window": recommended_window,
            "keep_sinks": keep_sinks,
            "sink_size": sink_size,
            "use_full_cache": use_full_cache,
            "compression_strategy": compression_strategy,
        })
    
    return new_classifications


def find_representative_heads(
    stats_data: dict,
    classifications: List[dict],
    num_examples: int = 3
) -> Dict[str, List[dict]]:
    """Find representative heads for each classification type."""
    
    stats_map = {(s['layer_idx'], s['head_idx']): s 
                 for s in stats_data['head_statistics']}
    
    # Group by type
    by_type = defaultdict(list)
    for c in classifications:
        key = (c['layer_idx'], c['head_idx'])
        by_type[c['head_type']].append({
            'classification': c,
            'stats': stats_map[key]
        })
    
    # Select best examples (highest confidence) for each type
    representatives = {}
    for head_type, heads in by_type.items():
        sorted_heads = sorted(heads, key=lambda x: x['classification']['confidence'], reverse=True)
        representatives[head_type] = sorted_heads[:num_examples]
    
    return representatives


def plot_relative_position_distribution(
    heads: Dict[str, List[dict]],
    output_path: str,
    max_distance: int = 32
) -> None:
    """Plot relative position distributions for each head type."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    head_types = ['sink_positional', 'true_positional', 'sink_mixed', 
                  'gathering', 'mixed', 'dead']
    
    colors = {
        'sink_positional': '#e74c3c',  # Red
        'true_positional': '#2ecc71',  # Green
        'sink_mixed': '#f39c12',       # Orange
        'gathering': '#3498db',        # Blue
        'mixed': '#9b59b6',            # Purple
        'dead': '#95a5a6',             # Gray
    }
    
    for idx, head_type in enumerate(head_types):
        ax = axes[idx]
        
        if head_type not in heads or len(heads[head_type]) == 0:
            ax.text(0.5, 0.5, f'No {head_type} heads', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{head_type.replace("_", " ").title()}')
            continue
        
        for i, head in enumerate(heads[head_type]):
            stats = head['stats']
            c = head['classification']
            dist = stats['relative_position_dist'][:max_distance]
            
            label = f'L{stats["layer_idx"]}H{stats["head_idx"]} (conf={c["confidence"]:.2f})'
            alpha = 0.9 - i * 0.2
            ax.bar(range(len(dist)), dist, alpha=alpha, label=label, 
                  color=colors.get(head_type, '#333333'))
        
        ax.set_xlabel('Relative Position (tokens back)')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'{head_type.replace("_", " ").title()}\n'
                    f'({len(heads[head_type])} heads total)')
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, max_distance - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved relative position distribution plot to {output_path}")


def plot_sink_vs_local_scatter(
    stats_data: dict,
    classifications: List[dict],
    output_path: str
) -> None:
    """Create scatter plot of sink ratio vs local ratio, colored by classification."""
    
    stats_map = {(s['layer_idx'], s['head_idx']): s 
                 for s in stats_data['head_statistics']}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {
        'sink_positional': '#e74c3c',
        'true_positional': '#2ecc71',
        'sink_mixed': '#f39c12',
        'gathering': '#3498db',
        'mixed': '#9b59b6',
        'dead': '#95a5a6',
    }
    
    for head_type in colors.keys():
        type_heads = [c for c in classifications if c['head_type'] == head_type]
        if not type_heads:
            continue
            
        sink_ratios = [stats_map[(c['layer_idx'], c['head_idx'])]['sink_ratio'] 
                      for c in type_heads]
        local_ratios = [stats_map[(c['layer_idx'], c['head_idx'])]['position_preference']['local'] 
                       for c in type_heads]
        
        ax.scatter(sink_ratios, local_ratios, 
                  c=colors[head_type], 
                  label=f'{head_type} ({len(type_heads)})',
                  alpha=0.6, s=30)
    
    # Add threshold lines
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Sink threshold (0.3)')
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Local threshold (0.6)')
    
    ax.set_xlabel('Sink Ratio (attention to first 4 tokens)', fontsize=12)
    ax.set_ylabel('Local Ratio (attention within window of 8)', fontsize=12)
    ax.set_title('Head Classification: Sink Ratio vs Local Ratio', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sink vs local scatter plot to {output_path}")


def plot_entropy_distribution(
    stats_data: dict,
    classifications: List[dict],
    output_path: str
) -> None:
    """Plot entropy distribution for each head type."""
    
    stats_map = {(s['layer_idx'], s['head_idx']): s 
                 for s in stats_data['head_statistics']}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    head_types = ['sink_positional', 'true_positional', 'sink_mixed', 
                  'gathering', 'mixed', 'dead']
    
    colors = {
        'sink_positional': '#e74c3c',
        'true_positional': '#2ecc71',
        'sink_mixed': '#f39c12',
        'gathering': '#3498db',
        'mixed': '#9b59b6',
        'dead': '#95a5a6',
    }
    
    for idx, head_type in enumerate(head_types):
        ax = axes[idx]
        
        type_heads = [c for c in classifications if c['head_type'] == head_type]
        if not type_heads:
            ax.text(0.5, 0.5, f'No {head_type} heads', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{head_type.replace("_", " ").title()}')
            continue
        
        entropies = [stats_map[(c['layer_idx'], c['head_idx'])]['mean_entropy'] 
                    for c in type_heads]
        
        ax.hist(entropies, bins=20, color=colors.get(head_type, '#333333'), 
               alpha=0.7, edgecolor='black')
        
        ax.axvline(x=np.mean(entropies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(entropies):.2f}')
        
        ax.set_xlabel('Mean Entropy')
        ax.set_ylabel('Count')
        ax.set_title(f'{head_type.replace("_", " ").title()} ({len(type_heads)} heads)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved entropy distribution plot to {output_path}")


def generate_summary_report(
    stats_data: dict,
    classifications: List[dict],
    output_path: str
) -> None:
    """Generate a detailed summary report."""
    
    stats_map = {(s['layer_idx'], s['head_idx']): s 
                 for s in stats_data['head_statistics']}
    
    # Count by type
    type_counts = defaultdict(int)
    strategy_counts = defaultdict(int)
    
    for c in classifications:
        type_counts[c['head_type']] += 1
        strategy_counts[c['compression_strategy']] += 1
    
    total = len(classifications)
    
    # Compute statistics by type
    type_stats = {}
    for head_type in type_counts.keys():
        type_heads = [c for c in classifications if c['head_type'] == head_type]
        
        sink_ratios = [stats_map[(c['layer_idx'], c['head_idx'])]['sink_ratio'] 
                      for c in type_heads]
        local_ratios = [stats_map[(c['layer_idx'], c['head_idx'])]['position_preference']['local'] 
                       for c in type_heads]
        entropies = [stats_map[(c['layer_idx'], c['head_idx'])]['mean_entropy'] 
                    for c in type_heads]
        
        type_stats[head_type] = {
            'count': len(type_heads),
            'sink_mean': np.mean(sink_ratios),
            'sink_std': np.std(sink_ratios),
            'local_mean': np.mean(local_ratios),
            'local_std': np.std(local_ratios),
            'entropy_mean': np.mean(entropies),
            'entropy_std': np.std(entropies),
        }
    
    # Calculate compression potential
    compressible_heads = sum(1 for c in classifications if not c['use_full_cache'])
    sink_needed = sum(1 for c in classifications if c['keep_sinks'])
    
    report = []
    report.append("=" * 80)
    report.append("HEAD CLASSIFICATION VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("CLASSIFICATION SUMMARY")
    report.append("-" * 40)
    for head_type in ['sink_positional', 'true_positional', 'sink_mixed', 
                      'gathering', 'mixed', 'dead']:
        count = type_counts.get(head_type, 0)
        pct = count / total * 100
        report.append(f"  {head_type:20}: {count:4} heads ({pct:5.1f}%)")
    report.append("")
    
    report.append("COMPRESSION STRATEGY SUMMARY")
    report.append("-" * 40)
    for strategy in ['sink_window', 'window_only', 'full', 'none', 'prune']:
        count = strategy_counts.get(strategy, 0)
        pct = count / total * 100
        report.append(f"  {strategy:20}: {count:4} heads ({pct:5.1f}%)")
    report.append("")
    
    report.append("COMPRESSION POTENTIAL")
    report.append("-" * 40)
    report.append(f"  Total heads: {total}")
    report.append(f"  Compressible (not full cache): {compressible_heads} ({compressible_heads/total*100:.1f}%)")
    report.append(f"  Require sink tokens: {sink_needed} ({sink_needed/total*100:.1f}%)")
    report.append("")
    
    report.append("DETAILED STATISTICS BY TYPE")
    report.append("-" * 40)
    for head_type in ['sink_positional', 'true_positional', 'sink_mixed', 
                      'gathering', 'mixed', 'dead']:
        if head_type not in type_stats:
            continue
        stats = type_stats[head_type]
        report.append(f"\n  {head_type.upper()}")
        report.append(f"    Count: {stats['count']}")
        report.append(f"    Sink ratio: {stats['sink_mean']:.3f} ± {stats['sink_std']:.3f}")
        report.append(f"    Local ratio: {stats['local_mean']:.3f} ± {stats['local_std']:.3f}")
        report.append(f"    Entropy: {stats['entropy_mean']:.3f} ± {stats['entropy_std']:.3f}")
    
    report.append("")
    report.append("KEY INSIGHTS")
    report.append("-" * 40)
    
    # Calculate key insights
    sink_pos_count = type_counts.get('sink_positional', 0)
    true_pos_count = type_counts.get('true_positional', 0)
    
    if sink_pos_count > 0 and true_pos_count > 0:
        report.append(f"  • Sink-positional heads outnumber true-positional by "
                     f"{sink_pos_count/true_pos_count:.1f}x")
    
    if sink_needed / total > 0.5:
        report.append(f"  • CRITICAL: {sink_needed/total*100:.1f}% of heads require sink tokens!")
        report.append(f"    -> StreamingLLM-style (sink + window) is essential")
    
    sink_mixed_count = type_counts.get('sink_mixed', 0)
    if sink_mixed_count > 0:
        report.append(f"  • {sink_mixed_count} sink-mixed heads need larger windows (16-32)")
    
    gathering_count = type_counts.get('gathering', 0)
    if gathering_count > 0:
        report.append(f"  • {gathering_count} gathering heads need full KV cache")
        report.append(f"    -> These cannot be compressed without quality loss")
    
    report.append("")
    report.append("RECOMMENDED COMPRESSION CONFIG")
    report.append("-" * 40)
    report.append("  {")
    report.append(f'    "sink_positional": {{"sink_size": 4, "window_size": 8}},')
    report.append(f'    "true_positional": {{"sink_size": 0, "window_size": 8}},')
    report.append(f'    "sink_mixed": {{"sink_size": 4, "window_size": 24}},')
    report.append(f'    "gathering": {{"use_full_cache": true}},')
    report.append(f'    "mixed": {{"sink_size": 4, "window_size": 32}},')
    report.append(f'    "dead": {{"prune": true}}')
    report.append("  }")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate head classification results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(project_root, "results", "attention_analysis_pythia-2.8b"),
        help="Directory containing analysis results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for validation results (default: {results_dir}/validation)"
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Re-run classification with updated logic"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "validation")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("HEAD CLASSIFICATION VALIDATION")
    print("=" * 70)
    print(f"\nResults directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reclassify: {args.reclassify}")
    
    # Load results
    print("\nLoading analysis results...")
    stats_data, class_data = load_analysis_results(args.results_dir)
    
    # Reclassify if requested
    if args.reclassify:
        print("\nRe-classifying heads with updated logic...")
        classifications = reclassify_heads(stats_data)
        
        # Save new classifications
        new_class_path = os.path.join(args.output_dir, "refined_classifications.json")
        with open(new_class_path, 'w') as f:
            json.dump({"classifications": classifications}, f, indent=2)
        print(f"Saved refined classifications to {new_class_path}")
    else:
        classifications = class_data['classifications']
    
    # Find representative heads
    print("\nFinding representative heads for each type...")
    representatives = find_representative_heads(stats_data, classifications, num_examples=3)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_relative_position_distribution(
        representatives,
        os.path.join(args.output_dir, "relative_position_by_type.png")
    )
    
    plot_sink_vs_local_scatter(
        stats_data,
        classifications,
        os.path.join(args.output_dir, "sink_vs_local_scatter.png")
    )
    
    plot_entropy_distribution(
        stats_data,
        classifications,
        os.path.join(args.output_dir, "entropy_by_type.png")
    )
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(
        stats_data,
        classifications,
        os.path.join(args.output_dir, "validation_report.txt")
    )
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  - relative_position_by_type.png")
    print("  - sink_vs_local_scatter.png")
    print("  - entropy_by_type.png")
    print("  - validation_report.txt")
    if args.reclassify:
        print("  - refined_classifications.json")


if __name__ == "__main__":
    main()


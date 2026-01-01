#!/usr/bin/env python3
"""
Tree-based Speculative Decoding Parameter Search

This script systematically searches for the optimal Tree V2 configuration
by testing various combinations of:
- D (depth): 3, 4, 5, 6, 7, 8
- B (branch factor): 2, 3, 4
- t (threshold): 0.01, 0.02, 0.03, 0.05, 0.1
- tokens: 100, 200, 300, 500, 1000

Usage:
    python tree_param_search.py \
        --target-model /mnt/disk1/models/pythia-2.8b \
        --draft-model /mnt/disk1/models/pythia-70m \
        --output-dir results
"""

import os
import sys
import json
import time
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Optional seaborn for prettier heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

logging.set_verbosity_error()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_decode.core import TreeSpeculativeGeneratorV2


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_DEPTHS = [3, 4, 5, 6, 7, 8]
DEFAULT_BRANCHES = [2, 3, 4]
DEFAULT_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.1]
DEFAULT_TOKEN_LENGTHS = [100, 200, 300, 500, 1000]

TEST_PROMPT = """Write a detailed technical explanation about the development of large language models and their applications in modern AI systems. Cover the architecture, training methods, and future directions. Begin your explanation:

Large language models have become"""


# =============================================================================
# Utility Functions
# =============================================================================
def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# Measurement Functions
# =============================================================================
def measure_baseline(
    target_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    num_runs: int = 2
) -> float:
    """Measure baseline autoregressive throughput."""
    cleanup()
    
    # Disable EOS for forced long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    throughputs = []
    try:
        for _ in range(num_runs):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.inference_mode():
                outputs = target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            generated_tokens = outputs.shape[1] - input_ids.shape[1]
            throughputs.append(generated_tokens / elapsed)
    finally:
        tokenizer.eos_token_id = original_eos
    
    return np.mean(throughputs)


def measure_tree_v2(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    depth: int,
    branch: int,
    threshold: float,
    device: str,
    num_runs: int = 2
) -> Tuple[float, float, float]:
    """
    Measure Tree V2 performance.
    
    Returns:
        throughput: tokens/sec
        avg_path_length: average accepted path length
        acceptance_rate: acceptance rate
    """
    cleanup()
    
    # Disable EOS for forced long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        generator = TreeSpeculativeGeneratorV2(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=depth,
            branch_factor=branch,
            probability_threshold=threshold,
            max_tree_nodes=128,
            device=device,
            use_compile=False
        )
        
        throughputs = []
        path_lengths = []
        acceptance_rates = []
        
        for _ in range(num_runs):
            generator.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            throughputs.append(stats['total_tokens'] / elapsed)
            path_lengths.append(stats.get('avg_accepted_path_length', 0))
            acceptance_rates.append(stats.get('acceptance_rate', 0))
        
        return (
            np.mean(throughputs),
            np.mean(path_lengths),
            np.mean(acceptance_rates)
        )
    except Exception as e:
        print(f"    ERROR: {str(e)[:50]}")
        return 0.0, 0.0, 0.0
    finally:
        tokenizer.eos_token_id = original_eos


# =============================================================================
# Search Functions
# =============================================================================
def search_tree_params(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    depths: List[int] = DEFAULT_DEPTHS,
    branches: List[int] = DEFAULT_BRANCHES,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
    token_lengths: List[int] = DEFAULT_TOKEN_LENGTHS,
    num_runs: int = 2
) -> List[Dict]:
    """
    Systematically search Tree V2 parameters.
    
    Returns:
        List of result dictionaries with all configurations tested.
    """
    total_configs = len(depths) * len(branches) * len(thresholds) * len(token_lengths)
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"Estimated time: {format_time(total_configs * 3)}")  # ~3s per config
    
    results = []
    baselines = {}  # Cache baselines by token length
    config_idx = 0
    start_time = time.time()
    
    for tokens in token_lengths:
        print(f"\n{'='*70}")
        print(f"Testing with {tokens} tokens")
        print(f"{'='*70}")
        
        # Measure baseline for this token length
        if tokens not in baselines:
            print(f"  Measuring baseline...")
            baselines[tokens] = measure_baseline(
                target_model, tokenizer, TEST_PROMPT, tokens, device, num_runs
            )
            print(f"  Baseline: {baselines[tokens]:.1f} t/s")
        
        baseline_tp = baselines[tokens]
        
        for D in depths:
            for B in branches:
                for t in thresholds:
                    config_idx += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / config_idx) * (total_configs - config_idx)
                    
                    print(f"  [{config_idx}/{total_configs}] D={D} B={B} t={t:.2f}", end=" ")
                    print(f"(ETA: {format_time(eta)})", end=" ")
                    
                    tp, path_len, acc_rate = measure_tree_v2(
                        target_model, draft_model, tokenizer,
                        TEST_PROMPT, tokens, D, B, t, device, num_runs
                    )
                    
                    speedup = tp / baseline_tp if baseline_tp > 0 else 0
                    
                    result = {
                        'tokens': tokens,
                        'depth': D,
                        'branch': B,
                        'threshold': t,
                        'throughput': tp,
                        'speedup': speedup,
                        'avg_path_length': path_len,
                        'acceptance_rate': acc_rate,
                        'baseline_throughput': baseline_tp
                    }
                    results.append(result)
                    
                    if tp > 0:
                        print(f"-> {tp:.1f} t/s ({speedup:.2f}x)")
                    else:
                        print(f"-> FAILED")
    
    return results


def find_best_config(results: List[Dict]) -> Dict:
    """Find the best configuration from search results."""
    valid_results = [r for r in results if r['throughput'] > 0]
    if not valid_results:
        return {}
    
    # Find best by speedup
    best = max(valid_results, key=lambda x: x['speedup'])
    return best


def find_best_per_token_length(results: List[Dict]) -> Dict[int, Dict]:
    """Find best configuration for each token length."""
    best_per_length = {}
    
    for result in results:
        tokens = result['tokens']
        if tokens not in best_per_length or result['speedup'] > best_per_length[tokens]['speedup']:
            best_per_length[tokens] = result
    
    return best_per_length


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_heatmaps(results: List[Dict], output_path: str):
    """Generate heatmap visualizations of parameter search results."""
    # Group results by token length
    token_lengths = sorted(set(r['tokens'] for r in results))
    
    fig, axes = plt.subplots(1, len(token_lengths), figsize=(5*len(token_lengths), 5))
    if len(token_lengths) == 1:
        axes = [axes]
    
    for idx, tokens in enumerate(token_lengths):
        ax = axes[idx]
        
        # Filter results for this token length and B=2 (best branch factor)
        filtered = [r for r in results if r['tokens'] == tokens and r['branch'] == 2]
        
        if not filtered:
            continue
        
        # Create matrix for heatmap (D x threshold)
        depths = sorted(set(r['depth'] for r in filtered))
        thresholds = sorted(set(r['threshold'] for r in filtered))
        
        matrix = np.zeros((len(depths), len(thresholds)))
        for r in filtered:
            i = depths.index(r['depth'])
            j = thresholds.index(r['threshold'])
            matrix[i, j] = r['speedup']
        
        # Plot heatmap
        if HAS_SEABORN:
            sns.heatmap(
                matrix, ax=ax, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'{t:.2f}' for t in thresholds],
                yticklabels=depths,
                cbar_kws={'label': 'Speedup'}
            )
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels(depths)
            
            # Add text annotations
            for i in range(len(depths)):
                for j in range(len(thresholds)):
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Speedup')
        
        ax.set_xlabel('Threshold (t)')
        ax.set_ylabel('Depth (D)')
        ax.set_title(f'{tokens} tokens (B=2)')
    
    plt.suptitle('Tree V2 Parameter Search: Speedup vs D and threshold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap saved to: {output_path}")


def plot_speedup_by_tokens(results: List[Dict], output_path: str):
    """Plot best speedup for each token length."""
    best_per_length = find_best_per_token_length(results)
    
    token_lengths = sorted(best_per_length.keys())
    speedups = [best_per_length[t]['speedup'] for t in token_lengths]
    configs = [f"D={best_per_length[t]['depth']} B={best_per_length[t]['branch']}" 
               for t in token_lengths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(token_lengths)), speedups, color='steelblue')
    ax.set_xticks(range(len(token_lengths)))
    ax.set_xticklabels([str(t) for t in token_lengths])
    ax.set_xlabel('Token Length')
    ax.set_ylabel('Best Speedup')
    ax.set_title('Best Tree V2 Speedup by Token Length')
    
    # Add config labels on bars
    for i, (bar, config) in enumerate(zip(bars, configs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                config, ha='center', va='bottom', fontsize=9)
    
    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Speedup chart saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Tree V2 Parameter Search")
    parser.add_argument("--target-model", type=str, required=True,
                       help="Path to target model")
    parser.add_argument("--draft-model", type=str, required=True,
                       help="Path to draft model")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--num-runs", type=int, default=2,
                       help="Number of runs per configuration")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer configurations")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("papers/figures", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("Tree V2 Parameter Search")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target Model: {args.target_model}")
    print(f"  Draft Model: {args.draft_model}")
    print(f"  Device: {args.device}")
    print(f"  Num Runs: {args.num_runs}")
    
    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    target_model.eval()
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    draft_model.eval()
    
    print(f"  Memory usage: {get_memory_mb():.0f} MB")
    
    # Configure search parameters
    if args.quick:
        depths = [4, 5, 6]
        branches = [2, 3]
        thresholds = [0.02, 0.05]
        token_lengths = [200, 500]
    else:
        depths = DEFAULT_DEPTHS
        branches = DEFAULT_BRANCHES
        thresholds = DEFAULT_THRESHOLDS
        token_lengths = DEFAULT_TOKEN_LENGTHS
    
    # Run search
    print("\nStarting parameter search...")
    results = search_tree_params(
        target_model, draft_model, tokenizer, args.device,
        depths, branches, thresholds, token_lengths,
        args.num_runs
    )
    
    # Find best configuration
    best = find_best_config(results)
    best_per_length = find_best_per_token_length(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SEARCH RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nBest configuration per token length:")
    for tokens in sorted(best_per_length.keys()):
        r = best_per_length[tokens]
        print(f"  {tokens:4d} tokens: D={r['depth']} B={r['branch']} t={r['threshold']:.2f} "
              f"-> {r['throughput']:.1f} t/s ({r['speedup']:.2f}x)")
    
    print(f"\nOverall best configuration:")
    print(f"  Tokens: {best['tokens']}")
    print(f"  Depth (D): {best['depth']}")
    print(f"  Branch (B): {best['branch']}")
    print(f"  Threshold (t): {best['threshold']}")
    print(f"  Throughput: {best['throughput']:.1f} t/s")
    print(f"  Speedup: {best['speedup']:.2f}x")
    print(f"  Avg Path Length: {best['avg_path_length']:.1f}")
    
    # Save results
    output_data = {
        'config': {
            'target_model': args.target_model,
            'draft_model': args.draft_model,
            'depths': depths,
            'branches': branches,
            'thresholds': thresholds,
            'token_lengths': token_lengths,
            'num_runs': args.num_runs,
            'timestamp': timestamp
        },
        'results': results,
        'best_overall': best,
        'best_per_token_length': {str(k): v for k, v in best_per_length.items()}
    }
    
    json_path = os.path.join(args.output_dir, f"tree_param_search_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate visualizations
    heatmap_path = f"papers/figures/tree_param_heatmap_{timestamp}.png"
    plot_heatmaps(results, heatmap_path)
    
    speedup_path = f"papers/figures/tree_param_speedup_{timestamp}.png"
    plot_speedup_by_tokens(results, speedup_path)
    
    print("\n" + "=" * 70)
    print("Parameter search complete!")
    print("=" * 70)
    print(f"\nTo run benchmark with best config, use:")
    print(f"  python papers/benchmark_all_methods.py \\")
    print(f"      --target-model {args.target_model} \\")
    print(f"      --draft-model {args.draft_model} \\")
    print(f"      --tree-config {json_path}")


if __name__ == "__main__":
    main()


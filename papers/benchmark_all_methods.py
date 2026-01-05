#!/usr/bin/env python3
"""
Comprehensive Benchmark: All Speculative Decoding Methods

This script benchmarks all speculative decoding methods using the optimal
Tree V2 configuration found by tree_param_search.py.

Methods tested:
1. Baseline (autoregressive)
2. HuggingFace Assisted Generation
3. Linear Speculative Decoding (K=4, 5, 6, 7, 8)
4. Tree V2 (optimal configuration from search)
5. StreamingLLM + Spec Decode (various configs)

Usage:
    python benchmark_all_methods.py \
        --target-model /mnt/disk1/models/pythia-2.8b \
        --draft-model /mnt/disk1/models/pythia-70m \
        --tree-config results/tree_param_search_*.json
"""

import os
import sys
import json
import time
import argparse
import gc
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_decode.core import (
    SpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
    StreamingSpeculativeGenerator,
)


# =============================================================================
# Configuration
# =============================================================================
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


def load_tree_config(json_path: str) -> Dict:
    """Load best tree configuration from search results."""
    # Handle glob patterns
    if '*' in json_path:
        matches = sorted(glob.glob(json_path))
        if not matches:
            raise FileNotFoundError(f"No files matching: {json_path}")
        json_path = matches[-1]  # Use most recent
        print(f"Using tree config: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data.get('best_overall', {})


# =============================================================================
# Benchmark Functions
# =============================================================================
def benchmark_baseline(
    target_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    num_runs: int = 3
) -> Dict:
    """Benchmark baseline autoregressive generation."""
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    results = {
        'method': 'Baseline (AR)',
        'throughputs': [],
        'latencies': [],
    }
    
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
            
            generated = outputs.shape[1] - input_ids.shape[1]
            results['throughputs'].append(generated / elapsed)
            results['latencies'].append(elapsed)
    finally:
        tokenizer.eos_token_id = original_eos
    
    results['avg_throughput'] = np.mean(results['throughputs'])
    results['std_throughput'] = np.std(results['throughputs'])
    results['avg_latency'] = np.mean(results['latencies'])
    
    return results


def benchmark_hf_assisted(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    num_runs: int = 3
) -> Dict:
    """Benchmark HuggingFace assisted generation."""
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    results = {
        'method': 'HF Assisted',
        'throughputs': [],
        'latencies': [],
    }
    
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
                    assistant_model=draft_model,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            generated = outputs.shape[1] - input_ids.shape[1]
            results['throughputs'].append(generated / elapsed)
            results['latencies'].append(elapsed)
    finally:
        tokenizer.eos_token_id = original_eos
    
    results['avg_throughput'] = np.mean(results['throughputs'])
    results['std_throughput'] = np.std(results['throughputs'])
    results['avg_latency'] = np.mean(results['latencies'])
    
    return results


def benchmark_linear(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    K: int,
    device: str,
    num_runs: int = 3
) -> Dict:
    """Benchmark linear speculative decoding."""
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=K,
        max_len=8192,
        device=device,
        use_compile=False
    )
    
    results = {
        'method': f'Linear K={K}',
        'K': K,
        'throughputs': [],
        'latencies': [],
        'acceptance_rates': [],
    }
    
    try:
        for _ in range(num_runs):
            generator.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            results['throughputs'].append(stats['total_tokens'] / elapsed)
            results['latencies'].append(elapsed)
            results['acceptance_rates'].append(stats.get('acceptance_rate', 0))
    finally:
        tokenizer.eos_token_id = original_eos
    
    results['avg_throughput'] = np.mean(results['throughputs'])
    results['std_throughput'] = np.std(results['throughputs'])
    results['avg_latency'] = np.mean(results['latencies'])
    results['avg_acceptance_rate'] = np.mean(results['acceptance_rates'])
    
    return results


def benchmark_tree_v2(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    depth: int,
    branch: int,
    threshold: float,
    device: str,
    num_runs: int = 3
) -> Dict:
    """Benchmark Tree V2 speculative decoding."""
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
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
    
    results = {
        'method': f'Tree V2 D={depth} B={branch} t={threshold}',
        'depth': depth,
        'branch': branch,
        'threshold': threshold,
        'throughputs': [],
        'latencies': [],
        'path_lengths': [],
    }
    
    try:
        for _ in range(num_runs):
            generator.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            results['throughputs'].append(stats['total_tokens'] / elapsed)
            results['latencies'].append(elapsed)
            results['path_lengths'].append(stats.get('avg_accepted_path_length', 0))
    finally:
        tokenizer.eos_token_id = original_eos
    
    results['avg_throughput'] = np.mean(results['throughputs'])
    results['std_throughput'] = np.std(results['throughputs'])
    results['avg_latency'] = np.mean(results['latencies'])
    results['avg_path_length'] = np.mean(results['path_lengths'])
    
    return results


def benchmark_streaming(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    K: int,
    cache_len: int,
    device: str,
    num_runs: int = 3
) -> Dict:
    """Benchmark streaming speculative decoding."""
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    generator = StreamingSpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=K,
        max_len=8192,
        max_cache_len=cache_len,
        start_size=4,
        recent_size=cache_len - 4,
        device=device,
        use_compile=False
    )
    
    results = {
        'method': f'Streaming K={K} cache={cache_len}',
        'K': K,
        'cache_len': cache_len,
        'throughputs': [],
        'latencies': [],
        'acceptance_rates': [],
    }
    
    try:
        for _ in range(num_runs):
            generator.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            results['throughputs'].append(stats['total_tokens'] / elapsed)
            results['latencies'].append(elapsed)
            results['acceptance_rates'].append(stats.get('acceptance_rate', 0))
    finally:
        tokenizer.eos_token_id = original_eos
    
    results['avg_throughput'] = np.mean(results['throughputs'])
    results['std_throughput'] = np.std(results['throughputs'])
    results['avg_latency'] = np.mean(results['latencies'])
    results['avg_acceptance_rate'] = np.mean(results['acceptance_rates'])
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_comparison(all_results: List[Dict], baseline_tp: float, output_path: str):
    """Generate comparison bar chart."""
    # Sort by speedup
    sorted_results = sorted(
        all_results,
        key=lambda x: x['avg_throughput'] / baseline_tp,
        reverse=True
    )
    
    methods = [r['method'] for r in sorted_results]
    throughputs = [r['avg_throughput'] for r in sorted_results]
    speedups = [r['avg_throughput'] / baseline_tp for r in sorted_results]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color scheme
    colors = []
    for m in methods:
        if 'Baseline' in m:
            colors.append('#808080')  # Gray
        elif 'HF' in m:
            colors.append('#2ecc71')  # Green
        elif 'Linear' in m:
            colors.append('#3498db')  # Blue
        elif 'Tree' in m:
            colors.append('#e74c3c')  # Red
        elif 'Streaming' in m:
            colors.append('#9b59b6')  # Purple
        else:
            colors.append('#95a5a6')  # Light gray
    
    # Plot 1: Throughput
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(methods)), throughputs, color=colors)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xlabel("Throughput (tokens/s)")
    ax1.set_title("Throughput Comparison")
    ax1.invert_yaxis()
    
    for bar, val in zip(bars1, throughputs):
        ax1.text(val + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9)
    
    # Plot 2: Speedup
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(methods)), speedups, color=colors)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.set_xlabel("Speedup (vs Baseline)")
    ax2.set_title("Speedup Comparison")
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.invert_yaxis()
    
    for bar, val in zip(bars2, speedups):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}x', va='center', fontsize=9)
    
    plt.suptitle("Speculative Decoding Methods Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison chart saved to: {output_path}")


def plot_detailed(all_results: List[Dict], baseline_tp: float, output_path: str):
    """Generate detailed multi-metric chart."""
    # Group results by method type
    linear_results = [r for r in all_results if 'Linear' in r['method']]
    streaming_results = [r for r in all_results if 'Streaming' in r['method']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Linear K comparison
    ax1 = axes[0, 0]
    if linear_results:
        ks = [r['K'] for r in linear_results]
        tps = [r['avg_throughput'] for r in linear_results]
        accs = [r.get('avg_acceptance_rate', 0) * 100 for r in linear_results]
        
        ax1_twin = ax1.twinx()
        bars = ax1.bar(range(len(ks)), tps, color='steelblue', alpha=0.7, label='Throughput')
        line = ax1_twin.plot(range(len(ks)), accs, 'ro-', label='Acceptance Rate')
        
        ax1.set_xticks(range(len(ks)))
        ax1.set_xticklabels([f'K={k}' for k in ks])
        ax1.set_ylabel('Throughput (t/s)', color='steelblue')
        ax1_twin.set_ylabel('Acceptance Rate (%)', color='red')
        ax1.set_title('Linear Spec Decode: K Value Analysis')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
    
    # Plot 2: Streaming cache size comparison
    ax2 = axes[0, 1]
    if streaming_results:
        # Group by K value
        k_groups = {}
        for r in streaming_results:
            k = r['K']
            if k not in k_groups:
                k_groups[k] = []
            k_groups[k].append(r)
        
        x_pos = 0
        xticks = []
        xtick_labels = []
        colors = plt.cm.Set2(np.linspace(0, 1, len(k_groups)))
        
        for i, (k, results) in enumerate(sorted(k_groups.items())):
            for r in results:
                ax2.bar(x_pos, r['avg_throughput'], color=colors[i], 
                       label=f'K={k}' if x_pos == 0 or k not in [k_groups[list(k_groups.keys())[j]][0]['K'] for j in range(i)] else '')
                xticks.append(x_pos)
                xtick_labels.append(f"c={r['cache_len']}")
                x_pos += 1
            x_pos += 0.5
        
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xtick_labels, rotation=45)
        ax2.set_ylabel('Throughput (t/s)')
        ax2.set_title('Streaming Spec Decode: Cache Size Analysis')
        ax2.legend()
    
    # Plot 3: Method categories comparison
    ax3 = axes[1, 0]
    categories = {
        'Baseline': [r for r in all_results if 'Baseline' in r['method']],
        'HF Assisted': [r for r in all_results if 'HF' in r['method']],
        'Linear (best)': [max(linear_results, key=lambda x: x['avg_throughput'])] if linear_results else [],
        'Tree V2': [r for r in all_results if 'Tree' in r['method']],
        'Streaming (best)': [max(streaming_results, key=lambda x: x['avg_throughput'])] if streaming_results else [],
    }
    
    cat_names = []
    cat_tps = []
    for name, results in categories.items():
        if results:
            cat_names.append(name)
            cat_tps.append(results[0]['avg_throughput'])
    
    colors = ['#808080', '#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(cat_names)]
    bars = ax3.bar(range(len(cat_names)), cat_tps, color=colors)
    ax3.set_xticks(range(len(cat_names)))
    ax3.set_xticklabels(cat_names, rotation=15)
    ax3.set_ylabel('Throughput (t/s)')
    ax3.set_title('Best Performance by Method Category')
    
    for bar, val in zip(bars, cat_tps):
        speedup = val / baseline_tp
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Find best methods
    best_overall = max(all_results, key=lambda x: x['avg_throughput'])
    best_linear = max(linear_results, key=lambda x: x['avg_throughput']) if linear_results else None
    best_tree = [r for r in all_results if 'Tree' in r['method']]
    best_tree = best_tree[0] if best_tree else None
    best_streaming = max(streaming_results, key=lambda x: x['avg_throughput']) if streaming_results else None
    
    summary_text = "BENCHMARK SUMMARY\n" + "="*40 + "\n\n"
    summary_text += f"Baseline: {baseline_tp:.1f} t/s\n\n"
    summary_text += f"Best Overall: {best_overall['method']}\n"
    summary_text += f"  Throughput: {best_overall['avg_throughput']:.1f} t/s\n"
    summary_text += f"  Speedup: {best_overall['avg_throughput']/baseline_tp:.2f}x\n\n"
    
    if best_linear:
        summary_text += f"Best Linear: {best_linear['method']}\n"
        summary_text += f"  Throughput: {best_linear['avg_throughput']:.1f} t/s\n"
        summary_text += f"  Speedup: {best_linear['avg_throughput']/baseline_tp:.2f}x\n\n"
    
    if best_tree:
        summary_text += f"Tree V2: {best_tree['method']}\n"
        summary_text += f"  Throughput: {best_tree['avg_throughput']:.1f} t/s\n"
        summary_text += f"  Speedup: {best_tree['avg_throughput']/baseline_tp:.2f}x\n\n"
    
    if best_streaming:
        summary_text += f"Best Streaming: {best_streaming['method']}\n"
        summary_text += f"  Throughput: {best_streaming['avg_throughput']:.1f} t/s\n"
        summary_text += f"  Speedup: {best_streaming['avg_throughput']/baseline_tp:.2f}x\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed chart saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive Spec Decode Benchmark")
    parser.add_argument("--target-model", type=str, required=True,
                       help="Path to target model")
    parser.add_argument("--draft-model", type=str, required=True,
                       help="Path to draft model")
    parser.add_argument("--tree-config", type=str, default=None,
                       help="Path to tree param search results JSON (supports glob)")
    parser.add_argument("--max-new-tokens", type=int, default=300,
                       help="Number of tokens to generate")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of runs per method")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("papers/figures", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("Comprehensive Speculative Decoding Benchmark")
    print("=" * 70)
    
    # Load tree config if provided
    tree_config = None
    if args.tree_config:
        try:
            tree_config = load_tree_config(args.tree_config)
            print(f"\nLoaded Tree V2 config:")
            print(f"  Depth: {tree_config.get('depth')}")
            print(f"  Branch: {tree_config.get('branch')}")
            print(f"  Threshold: {tree_config.get('threshold')}")
        except Exception as e:
            print(f"\nWarning: Could not load tree config: {e}")
            print("Using default: D=5 B=2 t=0.05")
            tree_config = {'depth': 5, 'branch': 2, 'threshold': 0.05}
    else:
        print("\nNo tree config specified, using default: D=5 B=2 t=0.05")
        tree_config = {'depth': 5, 'branch': 2, 'threshold': 0.05}
    
    print(f"\nConfiguration:")
    print(f"  Target Model: {args.target_model}")
    print(f"  Draft Model: {args.draft_model}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
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
    
    # Warmup
    print("\nWarming up...")
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    for _ in range(2):
        input_ids = tokenizer(TEST_PROMPT, return_tensors="pt").input_ids.to(args.device)
        with torch.inference_mode():
            _ = target_model.generate(input_ids, max_new_tokens=50, do_sample=False,
                                     assistant_model=draft_model, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
    tokenizer.eos_token_id = original_eos
    cleanup()
    
    all_results = []
    
    # 1. Baseline
    print("\n[1/5] Benchmarking Baseline...")
    result = benchmark_baseline(
        target_model, tokenizer, TEST_PROMPT, args.max_new_tokens,
        args.device, args.num_runs
    )
    all_results.append(result)
    baseline_tp = result['avg_throughput']
    print(f"  {baseline_tp:.1f} t/s (1.00x)")
    
    # 2. HuggingFace Assisted
    print("\n[2/5] Benchmarking HuggingFace Assisted...")
    result = benchmark_hf_assisted(
        target_model, draft_model, tokenizer, TEST_PROMPT, args.max_new_tokens,
        args.device, args.num_runs
    )
    result['speedup'] = result['avg_throughput'] / baseline_tp
    all_results.append(result)
    print(f"  {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x)")
    
    # 3. Linear Spec Decode
    print("\n[3/5] Benchmarking Linear Spec Decode...")
    for K in [4, 5, 6, 7, 8]:
        result = benchmark_linear(
            target_model, draft_model, tokenizer, TEST_PROMPT, args.max_new_tokens,
            K, args.device, args.num_runs
        )
        result['speedup'] = result['avg_throughput'] / baseline_tp
        all_results.append(result)
        print(f"  K={K}: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x) "
              f"Acc={result['avg_acceptance_rate']:.0%}")
    
    # 4. Tree V2 (optimal config)
    print("\n[4/5] Benchmarking Tree V2 (optimal config)...")
    result = benchmark_tree_v2(
        target_model, draft_model, tokenizer, TEST_PROMPT, args.max_new_tokens,
        tree_config['depth'], tree_config['branch'], tree_config['threshold'],
        args.device, args.num_runs
    )
    result['speedup'] = result['avg_throughput'] / baseline_tp
    all_results.append(result)
    print(f"  D={tree_config['depth']} B={tree_config['branch']} t={tree_config['threshold']}: "
          f"{result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x) "
          f"path={result['avg_path_length']:.1f}")
    
    # 5. Streaming Spec Decode
    print("\n[5/5] Benchmarking Streaming Spec Decode...")
    streaming_configs = [
        (5, 256),
        (5, 512),
        (6, 512),
        (6, 1024),
    ]
    
    for K, cache in streaming_configs:
        result = benchmark_streaming(
            target_model, draft_model, tokenizer, TEST_PROMPT, args.max_new_tokens,
            K, cache, args.device, args.num_runs
        )
        result['speedup'] = result['avg_throughput'] / baseline_tp
        all_results.append(result)
        print(f"  K={K} cache={cache}: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x) "
              f"Acc={result['avg_acceptance_rate']:.0%}")
    
    # Sort and print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (sorted by speedup)")
    print("=" * 70)
    
    sorted_results = sorted(all_results, key=lambda x: x.get('speedup', x['avg_throughput']/baseline_tp), reverse=True)
    
    print(f"\n{'Method':<35} {'Throughput':>12} {'Speedup':>10}")
    print("-" * 60)
    for r in sorted_results:
        speedup = r.get('speedup', r['avg_throughput'] / baseline_tp)
        print(f"{r['method']:<35} {r['avg_throughput']:>10.1f} t/s {speedup:>8.2f}x")
    
    # Find best methods
    best_overall = sorted_results[0]
    best_linear = max([r for r in all_results if 'Linear' in r['method']], 
                     key=lambda x: x['avg_throughput'])
    best_tree = [r for r in all_results if 'Tree' in r['method']][0]
    best_streaming = max([r for r in all_results if 'Streaming' in r['method']], 
                        key=lambda x: x['avg_throughput'])
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"\n  Best Overall: {best_overall['method']} -> {best_overall['speedup']:.2f}x")
    print(f"  Best Linear:  {best_linear['method']} -> {best_linear['speedup']:.2f}x")
    print(f"  Tree V2:      {best_tree['method']} -> {best_tree['speedup']:.2f}x")
    print(f"  Best Stream:  {best_streaming['method']} -> {best_streaming['speedup']:.2f}x")
    
    # Save results
    output_data = {
        'config': {
            'target_model': args.target_model,
            'draft_model': args.draft_model,
            'max_new_tokens': args.max_new_tokens,
            'num_runs': args.num_runs,
            'tree_config': tree_config,
            'timestamp': timestamp
        },
        'results': all_results,
        'summary': {
            'baseline_throughput': baseline_tp,
            'best_overall': best_overall['method'],
            'best_overall_speedup': best_overall['speedup'],
            'best_linear': best_linear['method'],
            'best_linear_speedup': best_linear['speedup'],
            'tree_v2_speedup': best_tree['speedup'],
            'best_streaming': best_streaming['method'],
            'best_streaming_speedup': best_streaming['speedup'],
        }
    }
    
    json_path = os.path.join(args.output_dir, f"benchmark_all_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n\nResults saved to: {json_path}")
    
    # Generate visualizations
    comparison_path = f"papers/figures/benchmark_comparison_{timestamp}.png"
    plot_comparison(all_results, baseline_tp, comparison_path)
    
    detailed_path = f"papers/figures/benchmark_detailed_{timestamp}.png"
    plot_detailed(all_results, baseline_tp, detailed_path)
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()









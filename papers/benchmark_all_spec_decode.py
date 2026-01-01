#!/usr/bin/env python3
"""
Comprehensive Speculative Decoding Benchmark

This script benchmarks all speculative decoding implementations:
1. Baseline (autoregressive) - FORCED long generation
2. HuggingFace Assisted Generation
3. Linear Speculative Decoding (Custom implementation)
4. Tree-based Speculative Decoding (SpecInfer-style)
5. StreamingLLM variants

IMPORTANT: Uses forced long generation (disabled EOS) to ensure fair comparison.

Usage:
    python benchmark_all_spec_decode.py \
        --target-model /mnt/disk1/models/pythia-2.8b \
        --draft-model /mnt/disk1/models/pythia-70m \
        --max-new-tokens 500 \
        --output-json results.json \
        --output-plot benchmark_comparison.png
"""

import os
import sys
import json
import time
import argparse
import gc
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
    TreeSpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
    StreamingSpeculativeGenerator,
    TreeStreamingSpeculativeGenerator,
)


# =============================================================================
# Test Prompts - Designed for long generation
# =============================================================================
DEFAULT_PROMPTS = [
    """Write a detailed technical explanation about the development of large language models. 
Cover the history, architecture innovations, training techniques, and future directions.
Begin your explanation:

Large language models have become""",
    
    """Explain the theory and practice of machine learning optimization algorithms in detail.
Discuss gradient descent variants, adaptive methods, and recent advances.
Start your detailed analysis:

Optimization in machine learning is""",
]


# =============================================================================
# Benchmark Functions
# =============================================================================
def cleanup():
    """Clean up GPU memory between tests."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def warmup(target_model, draft_model, tokenizer, device, num_rounds=2):
    """Warm up models with short sequences."""
    print("  Warming up models...")
    prompt = "Hello world"
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999  # Disable EOS
    
    for _ in range(num_rounds):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            _ = target_model.generate(
                input_ids, max_new_tokens=50, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
    
    tokenizer.eos_token_id = original_eos
    cleanup()


def benchmark_baseline(
    target_model, tokenizer, prompts, max_new_tokens, device, num_runs=2
) -> Dict:
    """Benchmark baseline autoregressive generation (forced long generation)."""
    cleanup()
    
    results = {
        "method": "Baseline (AR)",
        "throughputs": [],
        "latencies": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    # Disable EOS to force long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        for _ in range(num_runs):
            for prompt in prompts:
                cleanup()
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
                throughput = generated_tokens / elapsed
                
                results["throughputs"].append(throughput)
                results["latencies"].append(elapsed)
                results["total_tokens"] += generated_tokens
                results["total_time"] += elapsed
    finally:
        tokenizer.eos_token_id = original_eos
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["std_throughput"] = np.std(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    
    return results


def benchmark_hf_assisted(
    target_model, draft_model, tokenizer, prompts, max_new_tokens, device, num_runs=2
) -> Dict:
    """Benchmark HuggingFace assisted generation (forced long generation)."""
    cleanup()
    
    results = {
        "method": "HF Assisted",
        "throughputs": [],
        "latencies": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    # Disable EOS to force long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        for _ in range(num_runs):
            for prompt in prompts:
                cleanup()
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
                
                generated_tokens = outputs.shape[1] - input_ids.shape[1]
                throughput = generated_tokens / elapsed
                
                results["throughputs"].append(throughput)
                results["latencies"].append(elapsed)
                results["total_tokens"] += generated_tokens
                results["total_time"] += elapsed
    finally:
        tokenizer.eos_token_id = original_eos
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["std_throughput"] = np.std(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    
    return results


def benchmark_linear_spec_decode(
    target_model, draft_model, tokenizer, prompts, max_new_tokens, K, device, num_runs=2
) -> Dict:
    """Benchmark linear speculative decoding (forced long generation)."""
    cleanup()
    
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
        "method": f"Linear K={K}",
        "K": K,
        "throughputs": [],
        "latencies": [],
        "acceptance_rates": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    # Disable EOS to force long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        for _ in range(num_runs):
            for prompt in prompts:
                generator.reset()
                cleanup()
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                stats = generator.get_stats()
                generated_tokens = stats["total_tokens"]
                throughput = generated_tokens / elapsed if elapsed > 0 else 0
                
                results["throughputs"].append(throughput)
                results["latencies"].append(elapsed)
                results["acceptance_rates"].append(stats.get("acceptance_rate", 0))
                results["total_tokens"] += generated_tokens
                results["total_time"] += elapsed
    finally:
        tokenizer.eos_token_id = original_eos
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["std_throughput"] = np.std(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["avg_acceptance_rate"] = np.mean(results["acceptance_rates"])
    
    return results


def benchmark_tree_spec_decode(
    target_model, draft_model, tokenizer, prompts, max_new_tokens,
    tree_depth, branch_factor, threshold, device, use_v2=True, num_runs=2
) -> Dict:
    """Benchmark tree-based speculative decoding (forced long generation)."""
    cleanup()
    
    GeneratorClass = TreeSpeculativeGeneratorV2 if use_v2 else TreeSpeculativeGenerator
    
    kwargs = {
        "target_model": target_model,
        "draft_model": draft_model,
        "tokenizer": tokenizer,
        "tree_depth": tree_depth,
        "branch_factor": branch_factor,
        "max_tree_nodes": 128,
        "device": device,
        "use_compile": False,
    }
    if use_v2:
        kwargs["probability_threshold"] = threshold
    
    generator = GeneratorClass(**kwargs)
    
    method_name = f"Tree{'V2' if use_v2 else ''} D={tree_depth} B={branch_factor}"
    if use_v2:
        method_name += f" t={threshold}"
    
    results = {
        "method": method_name,
        "tree_depth": tree_depth,
        "branch_factor": branch_factor,
        "threshold": threshold if use_v2 else None,
        "throughputs": [],
        "latencies": [],
        "acceptance_rates": [],
        "avg_path_lengths": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    # Disable EOS to force long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        for _ in range(num_runs):
            for prompt in prompts:
                generator.reset()
                cleanup()
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                stats = generator.get_stats()
                generated_tokens = stats["total_tokens"]
                throughput = generated_tokens / elapsed if elapsed > 0 else 0
                
                results["throughputs"].append(throughput)
                results["latencies"].append(elapsed)
                results["acceptance_rates"].append(stats.get("acceptance_rate", 0))
                results["avg_path_lengths"].append(stats.get("avg_accepted_path_length", 0))
                results["total_tokens"] += generated_tokens
                results["total_time"] += elapsed
    finally:
        tokenizer.eos_token_id = original_eos
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["std_throughput"] = np.std(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["avg_acceptance_rate"] = np.mean(results["acceptance_rates"])
    results["avg_path_length"] = np.mean(results["avg_path_lengths"])
    
    return results


def benchmark_streaming_spec_decode(
    target_model, draft_model, tokenizer, prompts, max_new_tokens,
    K, start_size, recent_size, max_cache_len, device, num_runs=2
) -> Dict:
    """Benchmark streaming speculative decoding (forced long generation)."""
    cleanup()
    
    generator = StreamingSpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=K,
        max_len=8192,
        start_size=start_size,
        recent_size=recent_size,
        max_cache_len=max_cache_len,
        device=device,
        use_compile=False
    )
    
    results = {
        "method": f"Streaming K={K} cache={max_cache_len}",
        "K": K,
        "max_cache_len": max_cache_len,
        "throughputs": [],
        "latencies": [],
        "acceptance_rates": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    # Disable EOS to force long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        for _ in range(num_runs):
            for prompt in prompts:
                generator.reset()
                cleanup()
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                stats = generator.get_stats()
                generated_tokens = stats["total_tokens"]
                throughput = generated_tokens / elapsed if elapsed > 0 else 0
                
                results["throughputs"].append(throughput)
                results["latencies"].append(elapsed)
                results["acceptance_rates"].append(stats.get("acceptance_rate", 0))
                results["total_tokens"] += generated_tokens
                results["total_time"] += elapsed
    finally:
        tokenizer.eos_token_id = original_eos
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["std_throughput"] = np.std(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["avg_acceptance_rate"] = np.mean(results["acceptance_rates"])
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================
def run_comprehensive_benchmark(
    target_model_path: str,
    draft_model_path: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 100,
    num_runs: int = 2,
    device: str = "cuda"
) -> Dict:
    """Run comprehensive benchmark of all speculative decoding methods."""
    
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    
    print("=" * 70)
    print("Comprehensive Speculative Decoding Benchmark")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target Model: {target_model_path}")
    print(f"  Draft Model: {draft_model_path}")
    print(f"  Max New Tokens: {max_new_tokens}")
    print(f"  Num Prompts: {len(prompts)}")
    print(f"  Num Runs: {num_runs}")
    print(f"  Device: {device}")
    
    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    target_model.eval()
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    draft_model.eval()
    
    print(f"  Memory usage: {get_memory_usage():.0f} MB")
    
    # Warmup
    warmup(target_model, draft_model, tokenizer, device)
    
    all_results = {
        "config": {
            "target_model": target_model_path,
            "draft_model": draft_model_path,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "num_runs": num_runs,
            "timestamp": datetime.now().isoformat(),
        },
        "results": []
    }
    
    # 1. Baseline
    print("\n[1/8] Baseline...")
    baseline = benchmark_baseline(
        target_model, tokenizer, prompts, max_new_tokens, device, num_runs
    )
    all_results["results"].append(baseline)
    baseline_tp = baseline["avg_throughput"]
    print(f"  Throughput: {baseline_tp:.1f} t/s (1.00x)")
    
    # 2. HuggingFace Assisted
    print("\n[2/8] HuggingFace Assisted Generation...")
    hf_result = benchmark_hf_assisted(
        target_model, draft_model, tokenizer, prompts, max_new_tokens, device, num_runs
    )
    hf_result["speedup"] = hf_result["avg_throughput"] / baseline_tp
    all_results["results"].append(hf_result)
    print(f"  Throughput: {hf_result['avg_throughput']:.1f} t/s ({hf_result['speedup']:.2f}x)")
    
    # 3. Linear Speculative Decoding - Test multiple K values
    print("\n[3/8] Linear Speculative Decoding...")
    for K in [4, 5, 6, 7, 8]:
        result = benchmark_linear_spec_decode(
            target_model, draft_model, tokenizer, prompts, max_new_tokens, K, device, num_runs
        )
        result["speedup"] = result["avg_throughput"] / baseline_tp
        all_results["results"].append(result)
        print(f"  K={K}: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x), Acc: {result['avg_acceptance_rate']:.1%}")
    
    # 4. Tree V2 - Best configurations
    print("\n[4/8] Tree-based V2 Speculative Decoding...")
    tree_configs = [
        (5, 2, 0.03),   # D=5 B=2 t=0.03
        (6, 2, 0.02),   # D=6 B=2 t=0.02
    ]
    
    for D, B, t in tree_configs:
        result = benchmark_tree_spec_decode(
            target_model, draft_model, tokenizer, prompts, max_new_tokens,
            D, B, t, device, use_v2=True, num_runs=num_runs
        )
        result["speedup"] = result["avg_throughput"] / baseline_tp
        all_results["results"].append(result)
        print(f"  D={D} B={B} t={t}: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x), Path: {result['avg_path_length']:.1f}")
    
    # 5. Tree V1 for comparison (skip slow configs)
    print("\n[5/8] Tree-based V1 Speculative Decoding...")
    for D, B in [(3, 2)]:
        result = benchmark_tree_spec_decode(
            target_model, draft_model, tokenizer, prompts, max_new_tokens,
            D, B, 0.0, device, use_v2=False, num_runs=num_runs
        )
        result["speedup"] = result["avg_throughput"] / baseline_tp
        all_results["results"].append(result)
        print(f"  D={D} B={B}: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x)")
    
    # 6. Streaming Speculative Decoding - Test multiple configs
    print("\n[6/8] Streaming Speculative Decoding...")
    streaming_configs = [
        (5, 4, 252, 256),   # K=5, cache=256 (memory saving)
        (5, 4, 508, 512),   # K=5, cache=512 (balanced)
        (5, 4, 1020, 1024), # K=5, cache=1024 (performance)
        (6, 4, 1020, 1024), # K=6, cache=1024 (best K)
    ]
    
    for K, start, recent, cache in streaming_configs:
        result = benchmark_streaming_spec_decode(
            target_model, draft_model, tokenizer, prompts, max_new_tokens,
            K, start, recent, cache, device, num_runs
        )
        result["speedup"] = result["avg_throughput"] / baseline_tp
        all_results["results"].append(result)
        print(f"  K={K} cache={cache}: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x)")
    
    # 7. Tree + Streaming combination
    print("\n[7/8] Tree + Streaming Speculative Decoding...")
    cleanup()
    
    # Disable EOS for this test
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        generator = TreeStreamingSpeculativeGenerator(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=5,
            branch_factor=2,
            start_size=4,
            recent_size=508,
            max_cache_len=512,
            device=device,
            use_compile=False
        )
        
        result = {
            "method": "Tree+Streaming D=5 B=2",
            "throughputs": [],
            "latencies": [],
            "total_tokens": 0,
            "total_time": 0.0,
        }
        
        for _ in range(num_runs):
            for prompt in prompts:
                generator.reset()
                cleanup()
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                stats = generator.get_stats()
                generated_tokens = stats["total_tokens"]
                throughput = generated_tokens / elapsed if elapsed > 0 else 0
                
                result["throughputs"].append(throughput)
                result["latencies"].append(elapsed)
                result["total_tokens"] += generated_tokens
                result["total_time"] += elapsed
        
        result["avg_throughput"] = np.mean(result["throughputs"])
        result["speedup"] = result["avg_throughput"] / baseline_tp
        all_results["results"].append(result)
        print(f"  D=5 B=2 cache=512: {result['avg_throughput']:.1f} t/s ({result['speedup']:.2f}x)")
    except Exception as e:
        print(f"  Skipped due to error: {e}")
    finally:
        tokenizer.eos_token_id = original_eos
    
    # Summary
    print("\n[8/8] Generating Summary...")
    
    # Sort by speedup
    sorted_results = sorted(
        all_results["results"],
        key=lambda x: x.get("speedup", x["avg_throughput"] / baseline_tp),
        reverse=True
    )
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (sorted by speedup)")
    print("=" * 70)
    print(f"{'Method':<35} {'Throughput':>12} {'Speedup':>10}")
    print("-" * 57)
    
    for result in sorted_results:
        speedup = result.get("speedup", result["avg_throughput"] / baseline_tp)
        print(f"{result['method']:<35} {result['avg_throughput']:>10.1f} t/s {speedup:>8.2f}x")
    
    # Find best
    best = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST: {best['method']} with {best.get('speedup', best['avg_throughput']/baseline_tp):.2f}x speedup")
    print("=" * 70)
    
    all_results["summary"] = {
        "baseline_throughput": baseline_tp,
        "best_method": best["method"],
        "best_speedup": best.get("speedup", best["avg_throughput"] / baseline_tp),
        "best_throughput": best["avg_throughput"],
    }
    
    return all_results


def plot_results(results: Dict, output_path: str):
    """Generate visualization of benchmark results."""
    
    data = results["results"]
    baseline_tp = results["summary"]["baseline_throughput"]
    
    # Sort by speedup
    sorted_data = sorted(
        data,
        key=lambda x: x.get("speedup", x["avg_throughput"] / baseline_tp),
        reverse=True
    )
    
    methods = [r["method"] for r in sorted_data]
    throughputs = [r["avg_throughput"] for r in sorted_data]
    speedups = [r.get("speedup", r["avg_throughput"] / baseline_tp) for r in sorted_data]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    
    # Plot 1: Throughput
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(methods)), throughputs, color=colors)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xlabel("Throughput (tokens/s)")
    ax1.set_title("Throughput Comparison")
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, throughputs)):
        ax1.text(val + 2, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontsize=9)
    
    # Plot 2: Speedup
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(methods)), speedups, color=colors)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.set_xlabel("Speedup (vs Baseline)")
    ax2.set_title("Speedup Comparison")
    ax2.axvline(x=1.0, color='red', linestyle='--', label='Baseline')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, speedups)):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}x', va='center', fontsize=9)
    
    plt.suptitle(f"Speculative Decoding Benchmark\n"
                f"Target: {results['config']['target_model'].split('/')[-1]}, "
                f"Draft: {results['config']['draft_model'].split('/')[-1]}, "
                f"Max Tokens: {results['config']['max_new_tokens']}",
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    results_serializable = json.loads(json.dumps(results, default=convert))
    
    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Speculative Decoding Benchmark"
    )
    parser.add_argument(
        "--target-model", type=str, required=True,
        help="Path to target model"
    )
    parser.add_argument(
        "--draft-model", type=str, required=True,
        help="Path to draft model"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--num-runs", type=int, default=2,
        help="Number of runs per configuration"
    )
    parser.add_argument(
        "--output-json", type=str, default="benchmark_all_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--output-plot", type=str, default="benchmark_all_comparison.png",
        help="Output plot file path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        device=args.device
    )
    
    # Save results
    save_results(results, args.output_json)
    
    # Generate plot
    plot_results(results, args.output_plot)
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()


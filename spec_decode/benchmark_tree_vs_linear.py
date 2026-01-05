#!/usr/bin/env python3
"""
Benchmark: Tree-based vs Linear Speculative Decoding

This script compares the performance of different speculative decoding strategies:
1. Baseline: Standard autoregressive decoding
2. Linear: Standard speculative decoding (K tokens)
3. Tree: Tree-based speculative decoding (depth D, branch factor B)
4. HuggingFace: HF's built-in assisted generation

Metrics:
- Throughput (tokens/second)
- Latency (TPOT - Time Per Output Token)
- Acceptance Rate
- Memory Usage
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_decode.core import (
    SpeculativeGenerator,
    TreeSpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
)


def get_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def benchmark_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    device: str = "cuda"
) -> Dict:
    """Benchmark baseline autoregressive generation."""
    results = {
        "method": "Baseline",
        "latencies": [],
        "throughputs": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        generated_tokens = outputs.shape[1] - input_ids.shape[1]
        throughput = generated_tokens / elapsed
        
        results["latencies"].append(elapsed)
        results["throughputs"].append(throughput)
        results["total_tokens"] += generated_tokens
        results["total_time"] += elapsed
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["tpot_ms"] = (results["total_time"] / results["total_tokens"]) * 1000
    
    return results


def benchmark_hf_assisted(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    num_assistant_tokens: int = 5,
    device: str = "cuda"
) -> Dict:
    """Benchmark HuggingFace's assisted generation."""
    results = {
        "method": "HuggingFace Assisted",
        "latencies": [],
        "throughputs": [],
        "total_tokens": 0,
        "total_time": 0.0,
        "num_assistant_tokens": num_assistant_tokens,
    }
    
    for prompt in prompts:
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
        
        results["latencies"].append(elapsed)
        results["throughputs"].append(throughput)
        results["total_tokens"] += generated_tokens
        results["total_time"] += elapsed
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["tpot_ms"] = (results["total_time"] / results["total_tokens"]) * 1000
    
    return results


def benchmark_linear_spec_decode(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    K: int = 5,
    device: str = "cuda"
) -> Dict:
    """Benchmark linear speculative decoding."""
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=K,
        device=device,
        use_compile=False  # Disable for fair comparison
    )
    
    results = {
        "method": f"Linear (K={K})",
        "K": K,
        "latencies": [],
        "throughputs": [],
        "acceptance_rates": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    for prompt in prompts:
        generator.reset()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        stats = generator.get_stats()
        generated_tokens = stats["total_tokens"]
        throughput = generated_tokens / elapsed if elapsed > 0 else 0
        
        results["latencies"].append(elapsed)
        results["throughputs"].append(throughput)
        results["acceptance_rates"].append(stats.get("acceptance_rate", 0))
        results["total_tokens"] += generated_tokens
        results["total_time"] += elapsed
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["avg_acceptance_rate"] = np.mean(results["acceptance_rates"])
    results["tpot_ms"] = (results["total_time"] / max(1, results["total_tokens"])) * 1000
    
    return results


def benchmark_tree_spec_decode(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    tree_depth: int = 4,
    branch_factor: int = 2,
    probability_threshold: float = 0.0,
    device: str = "cuda",
    use_v2: bool = False
) -> Dict:
    """Benchmark tree-based speculative decoding."""
    GeneratorClass = TreeSpeculativeGeneratorV2 if use_v2 else TreeSpeculativeGenerator
    
    # Build kwargs based on version
    generator_kwargs = {
        "target_model": target_model,
        "draft_model": draft_model,
        "tokenizer": tokenizer,
        "tree_depth": tree_depth,
        "branch_factor": branch_factor,
        "device": device,
        "use_compile": False
    }
    
    # V2 supports probability threshold
    if use_v2:
        generator_kwargs["probability_threshold"] = probability_threshold
    
    generator = GeneratorClass(**generator_kwargs)
    
    method_name = f"Tree (D={tree_depth}, B={branch_factor})"
    if use_v2:
        method_name += f" V2 (τ={probability_threshold})"
    
    results = {
        "method": method_name,
        "tree_depth": tree_depth,
        "branch_factor": branch_factor,
        "probability_threshold": probability_threshold if use_v2 else None,
        "latencies": [],
        "throughputs": [],
        "acceptance_rates": [],
        "avg_path_lengths": [],
        "total_tokens": 0,
        "total_time": 0.0,
    }
    
    for prompt in prompts:
        generator.reset()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        stats = generator.get_stats()
        generated_tokens = stats["total_tokens"]
        throughput = generated_tokens / elapsed if elapsed > 0 else 0
        
        results["latencies"].append(elapsed)
        results["throughputs"].append(throughput)
        results["acceptance_rates"].append(stats.get("acceptance_rate", 0))
        results["avg_path_lengths"].append(stats.get("avg_accepted_path_length", 0))
        results["total_tokens"] += generated_tokens
        results["total_time"] += elapsed
    
    results["avg_throughput"] = np.mean(results["throughputs"])
    results["avg_latency"] = np.mean(results["latencies"])
    results["avg_acceptance_rate"] = np.mean(results["acceptance_rates"])
    results["avg_path_length"] = np.mean(results["avg_path_lengths"])
    results["tpot_ms"] = (results["total_time"] / max(1, results["total_tokens"])) * 1000
    
    return results


def run_comprehensive_benchmark(
    target_model_name: str = "EleutherAI/pythia-2.8b",
    draft_model_name: str = "EleutherAI/pythia-70m",
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 100,
    num_warmup: int = 2,
    device: str = "cuda"
) -> Dict:
    """Run comprehensive benchmark comparing all methods."""
    
    print("=" * 60)
    print("Tree vs Linear Speculative Decoding Benchmark")
    print("=" * 60)
    
    # Default prompts
    if prompts is None:
        prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The key to understanding quantum mechanics lies in",
            "Once upon a time, in a distant galaxy,",
            "The most important scientific discovery of the century was",
        ]
    
    print(f"\nConfiguration:")
    print(f"  Target Model: {target_model_name}")
    print(f"  Draft Model: {draft_model_name}")
    print(f"  Max New Tokens: {max_new_tokens}")
    print(f"  Number of Prompts: {len(prompts)}")
    print(f"  Warmup Iterations: {num_warmup}")
    
    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    target_model.eval()
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    draft_model.eval()
    
    print(f"  Target model memory: {get_memory_usage():.2f} GB")
    
    all_results = {
        "config": {
            "target_model": target_model_name,
            "draft_model": draft_model_name,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "timestamp": datetime.now().isoformat(),
        },
        "results": []
    }
    
    # Warmup
    print("\nWarming up...")
    warmup_prompt = prompts[0]
    for _ in range(num_warmup):
        with torch.inference_mode():
            input_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to(device)
            _ = target_model.generate(input_ids, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    # 1. Baseline
    print("\n[1/5] Benchmarking Baseline...")
    baseline_results = benchmark_baseline(
        target_model, tokenizer, prompts, max_new_tokens, device
    )
    all_results["results"].append(baseline_results)
    print(f"  Throughput: {baseline_results['avg_throughput']:.1f} tokens/s")
    print(f"  TPOT: {baseline_results['tpot_ms']:.2f} ms")
    
    # 2. HuggingFace Assisted
    print("\n[2/5] Benchmarking HuggingFace Assisted Generation...")
    try:
        hf_results = benchmark_hf_assisted(
            target_model, draft_model, tokenizer, prompts, max_new_tokens, 
            num_assistant_tokens=5, device=device
        )
        all_results["results"].append(hf_results)
        print(f"  Throughput: {hf_results['avg_throughput']:.1f} tokens/s")
        print(f"  TPOT: {hf_results['tpot_ms']:.2f} ms")
        print(f"  Speedup vs Baseline: {hf_results['avg_throughput'] / baseline_results['avg_throughput']:.2f}x")
    except Exception as e:
        print(f"  Skipped due to error: {e}")
    
    # 3. Linear Speculative Decoding
    print("\n[3/5] Benchmarking Linear Speculative Decoding (K=5)...")
    linear_results = benchmark_linear_spec_decode(
        target_model, draft_model, tokenizer, prompts, max_new_tokens, K=5, device=device
    )
    all_results["results"].append(linear_results)
    print(f"  Throughput: {linear_results['avg_throughput']:.1f} tokens/s")
    print(f"  TPOT: {linear_results['tpot_ms']:.2f} ms")
    print(f"  Acceptance Rate: {linear_results['avg_acceptance_rate']:.1%}")
    print(f"  Speedup vs Baseline: {linear_results['avg_throughput'] / baseline_results['avg_throughput']:.2f}x")
    
    # 4. Tree Speculative Decoding
    print("\n[4/5] Benchmarking Tree Speculative Decoding (D=4, B=2)...")
    tree_results = benchmark_tree_spec_decode(
        target_model, draft_model, tokenizer, prompts, max_new_tokens,
        tree_depth=4, branch_factor=2, device=device
    )
    all_results["results"].append(tree_results)
    print(f"  Throughput: {tree_results['avg_throughput']:.1f} tokens/s")
    print(f"  TPOT: {tree_results['tpot_ms']:.2f} ms")
    print(f"  Acceptance Rate: {tree_results['avg_acceptance_rate']:.1%}")
    print(f"  Avg Path Length: {tree_results['avg_path_length']:.2f}")
    print(f"  Speedup vs Baseline: {tree_results['avg_throughput'] / baseline_results['avg_throughput']:.2f}x")
    print(f"  Speedup vs Linear: {tree_results['avg_throughput'] / max(0.1, linear_results['avg_throughput']):.2f}x")
    
    # 5. Tree V2 (with pruning) - Using optimal parameters from actual experiments
    # Note: Optimal params vary by sequence length (see papers/Tree_Speculative_Decoding_实验报告.md)
    # For 100 tokens: D=7, B=3, τ=0.03
    # For 500 tokens: D=8, B=3, τ=0.03
    print("\n[5/5] Benchmarking Tree V2 Speculative Decoding (D=8, B=3, τ=0.03)...")
    tree_v2_results = benchmark_tree_spec_decode(
        target_model, draft_model, tokenizer, prompts, max_new_tokens,
        tree_depth=8, branch_factor=3, probability_threshold=0.03, 
        device=device, use_v2=True
    )
    all_results["results"].append(tree_v2_results)
    print(f"  Throughput: {tree_v2_results['avg_throughput']:.1f} tokens/s")
    print(f"  TPOT: {tree_v2_results['tpot_ms']:.2f} ms")
    print(f"  Acceptance Rate: {tree_v2_results['avg_acceptance_rate']:.1%}")
    print(f"  Avg Path Length: {tree_v2_results['avg_path_length']:.2f}")
    print(f"  Speedup vs Baseline: {tree_v2_results['avg_throughput'] / baseline_results['avg_throughput']:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'Throughput':>12} {'TPOT':>10} {'Speedup':>10}")
    print("-" * 67)
    
    for result in all_results["results"]:
        speedup = result['avg_throughput'] / baseline_results['avg_throughput']
        print(f"{result['method']:<35} {result['avg_throughput']:>10.1f} t/s {result['tpot_ms']:>8.2f} ms {speedup:>8.2f}x")
    
    return all_results


def save_results(results: Dict, output_dir: str = "results"):
    """Save benchmark results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tree_vs_linear_benchmark_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    results_serializable = json.loads(
        json.dumps(results, default=convert_numpy)
    )
    
    with open(filepath, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Benchmark Tree vs Linear Speculative Decoding")
    parser.add_argument("--target-model", type=str, default="EleutherAI/pythia-2.8b",
                       help="Target model name or path")
    parser.add_argument("--draft-model", type=str, default="EleutherAI/pythia-70m",
                       help="Draft model name or path")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="Maximum new tokens to generate")
    parser.add_argument("--num-prompts", type=int, default=5,
                       help="Number of prompts to test")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    # Save results if requested
    if args.save:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()









"""
Detailed Benchmark: Speculative Decoding Performance Analysis

This script provides comprehensive performance analysis of speculative decoding
with the following metrics:

1. Throughput (tokens/second)
2. TTFT (Time to First Token)
3. TPOT (Time per Output Token)
4. Acceptance Rate by K value
5. Memory Usage (Peak VRAM)
6. Per-round timing breakdown

Usage:
    python benchmark_detailed.py \
        --target-model /path/to/pythia-2.8b \
        --draft-model /path/to/pythia-70m \
        --k-values 2 3 4 5 6 7 8 \
        --num-samples 10 \
        --max-new-tokens 100
"""

import torch
import time
import argparse
import json
import gc
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

from core import SpeculativeGenerator


@dataclass
class DetailedMetrics:
    """Detailed metrics from a benchmark run."""
    method: str
    k_value: int
    # Throughput metrics
    throughput: float = 0.0
    throughput_std: float = 0.0
    # Timing metrics
    ttft_mean: float = 0.0
    ttft_std: float = 0.0
    tpot_mean: float = 0.0
    tpot_std: float = 0.0
    total_time_mean: float = 0.0
    total_time_std: float = 0.0
    # Acceptance metrics (for custom only)
    acceptance_rate: float = 0.0
    tokens_per_round: float = 0.0
    total_rounds: int = 0
    # Memory metrics
    peak_memory_mb: float = 0.0
    # Token counts
    total_tokens: int = 0
    num_samples: int = 0
    # Per-sample data
    per_sample_throughput: List[float] = field(default_factory=list)
    per_sample_ttft: List[float] = field(default_factory=list)
    per_sample_tpot: List[float] = field(default_factory=list)


def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_baseline(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str
) -> DetailedMetrics:
    """Measure baseline (no speculative decoding) performance."""
    metrics = DetailedMetrics(method="Baseline", k_value=0)
    
    ttfts = []
    tpots = []
    throughputs = []
    total_tokens = 0
    
    # Warmup
    cleanup()
    dummy = tokenizer("Warmup", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=5, do_sample=False, 
                          pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    for prompt in tqdm(prompts, desc="Baseline", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        input_len = inputs.input_ids.shape[1]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            # Generate first token to measure TTFT
            first_output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time
        ttfts.append(ttft)
        
        # Continue generation
        with torch.inference_mode():
            full_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        
        new_tokens = full_output.shape[1] - input_len
        total_tokens += new_tokens
        
        if new_tokens > 1:
            tpot = (total_time - ttft) / (new_tokens - 1)
            tpots.append(tpot)
        
        throughput = new_tokens / total_time if total_time > 0 else 0
        throughputs.append(throughput)
    
    metrics.throughput = np.mean(throughputs)
    metrics.throughput_std = np.std(throughputs)
    metrics.ttft_mean = np.mean(ttfts) * 1000  # Convert to ms
    metrics.ttft_std = np.std(ttfts) * 1000
    metrics.tpot_mean = np.mean(tpots) * 1000 if tpots else 0
    metrics.tpot_std = np.std(tpots) * 1000 if tpots else 0
    metrics.total_tokens = total_tokens
    metrics.num_samples = len(prompts)
    metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics.per_sample_throughput = throughputs
    metrics.per_sample_ttft = [t * 1000 for t in ttfts]
    metrics.per_sample_tpot = [t * 1000 for t in tpots]
    
    return metrics


def measure_huggingface(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    k_value: int,
    max_new_tokens: int,
    device: str
) -> DetailedMetrics:
    """Measure HuggingFace's native speculative decoding performance."""
    metrics = DetailedMetrics(method="HuggingFace", k_value=k_value)
    
    ttfts = []
    tpots = []
    throughputs = []
    total_tokens = 0
    
    # Warmup
    cleanup()
    dummy = tokenizer("Warmup", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = target_model.generate(
            **dummy,
            assistant_model=draft_model,
            max_new_tokens=5,
            num_assistant_tokens=k_value,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    for prompt in tqdm(prompts, desc=f"HF (K={k_value})", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        input_len = inputs.input_ids.shape[1]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            # Generate first token to measure TTFT
            first_output = target_model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=1,
                num_assistant_tokens=k_value,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time
        ttfts.append(ttft)
        
        # Continue generation
        with torch.inference_mode():
            full_output = target_model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=max_new_tokens,
                num_assistant_tokens=k_value,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        
        new_tokens = full_output.shape[1] - input_len
        total_tokens += new_tokens
        
        if new_tokens > 1:
            tpot = (total_time - ttft) / (new_tokens - 1)
            tpots.append(tpot)
        
        throughput = new_tokens / total_time if total_time > 0 else 0
        throughputs.append(throughput)
    
    metrics.throughput = np.mean(throughputs)
    metrics.throughput_std = np.std(throughputs)
    metrics.ttft_mean = np.mean(ttfts) * 1000
    metrics.ttft_std = np.std(ttfts) * 1000
    metrics.tpot_mean = np.mean(tpots) * 1000 if tpots else 0
    metrics.tpot_std = np.std(tpots) * 1000 if tpots else 0
    metrics.total_tokens = total_tokens
    metrics.num_samples = len(prompts)
    metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics.per_sample_throughput = throughputs
    metrics.per_sample_ttft = [t * 1000 for t in ttfts]
    metrics.per_sample_tpot = [t * 1000 for t in tpots]
    
    return metrics


def measure_custom(
    generator: SpeculativeGenerator,
    prompts: List[str],
    k_value: int,
    max_new_tokens: int
) -> DetailedMetrics:
    """Measure custom speculative decoding implementation performance."""
    metrics = DetailedMetrics(method="Custom", k_value=k_value)
    generator.K = k_value
    
    ttfts = []
    tpots = []
    throughputs = []
    total_tokens = 0
    total_rounds = 0
    total_accepted = 0
    total_draft_tokens = 0
    
    # Warmup
    cleanup()
    generator.reset()
    _ = generator.generate("Warmup", max_new_tokens=5)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    for prompt in tqdm(prompts, desc=f"Custom (K={k_value})", leave=False):
        generator.reset()
        
        # Measure TTFT by generating with just 1 token first
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = generator.generate(prompt, max_new_tokens=1)
        
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time
        ttfts.append(ttft)
        
        # Reset and do full generation
        generator.reset()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        
        stats = generator.get_stats()
        new_tokens = stats["total_tokens"]
        total_tokens += new_tokens
        total_rounds += stats["total_rounds"]
        total_accepted += stats["total_accepted"]
        total_draft_tokens += stats["total_draft_tokens"]
        
        if new_tokens > 1:
            tpot = (total_time - ttft) / (new_tokens - 1)
            tpots.append(tpot)
        
        throughput = new_tokens / total_time if total_time > 0 else 0
        throughputs.append(throughput)
    
    metrics.throughput = np.mean(throughputs)
    metrics.throughput_std = np.std(throughputs)
    metrics.ttft_mean = np.mean(ttfts) * 1000
    metrics.ttft_std = np.std(ttfts) * 1000
    metrics.tpot_mean = np.mean(tpots) * 1000 if tpots else 0
    metrics.tpot_std = np.std(tpots) * 1000 if tpots else 0
    metrics.total_tokens = total_tokens
    metrics.num_samples = len(prompts)
    metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics.total_rounds = total_rounds
    metrics.acceptance_rate = total_accepted / total_draft_tokens if total_draft_tokens > 0 else 0
    metrics.tokens_per_round = total_tokens / total_rounds if total_rounds > 0 else 0
    metrics.per_sample_throughput = throughputs
    metrics.per_sample_ttft = [t * 1000 for t in ttfts]
    metrics.per_sample_tpot = [t * 1000 for t in tpots]
    
    return metrics


def print_detailed_summary(
    baseline: DetailedMetrics,
    hf_results: List[DetailedMetrics],
    custom_results: List[DetailedMetrics]
):
    """Print detailed summary of benchmark results."""
    print("\n" + "=" * 100)
    print("DETAILED BENCHMARK SUMMARY")
    print("=" * 100)
    
    # Throughput table
    print("\n📊 Throughput Comparison")
    print("-" * 80)
    print(f"{'Method':<15} {'K':<5} {'Throughput':<20} {'Speedup':<12} {'Accept Rate':<12}")
    print("-" * 80)
    
    print(f"{'Baseline':<15} {'-':<5} {baseline.throughput:.1f} ± {baseline.throughput_std:.1f} t/s{'':<5} {'1.00x':<12} {'-':<12}")
    
    for m in sorted(hf_results + custom_results, key=lambda x: (x.method, x.k_value)):
        speedup = m.throughput / baseline.throughput if baseline.throughput > 0 else 0
        acc_str = f"{m.acceptance_rate:.1%}" if m.acceptance_rate > 0 else "-"
        print(f"{m.method:<15} {m.k_value:<5} {m.throughput:.1f} ± {m.throughput_std:.1f} t/s{'':<5} {speedup:.2f}x{'':<7} {acc_str:<12}")
    
    # Latency table
    print("\n⏱️ Latency Comparison (ms)")
    print("-" * 80)
    print(f"{'Method':<15} {'K':<5} {'TTFT':<20} {'TPOT':<20}")
    print("-" * 80)
    
    print(f"{'Baseline':<15} {'-':<5} {baseline.ttft_mean:.1f} ± {baseline.ttft_std:.1f}{'':<8} {baseline.tpot_mean:.1f} ± {baseline.tpot_std:.1f}")
    
    for m in sorted(hf_results + custom_results, key=lambda x: (x.method, x.k_value)):
        print(f"{m.method:<15} {m.k_value:<5} {m.ttft_mean:.1f} ± {m.ttft_std:.1f}{'':<8} {m.tpot_mean:.1f} ± {m.tpot_std:.1f}")
    
    # Memory table
    print("\n💾 Memory Usage (MB)")
    print("-" * 60)
    print(f"{'Method':<15} {'K':<5} {'Peak VRAM':<15}")
    print("-" * 60)
    
    print(f"{'Baseline':<15} {'-':<5} {baseline.peak_memory_mb:.1f}")
    
    for m in sorted(hf_results + custom_results, key=lambda x: (x.method, x.k_value)):
        print(f"{m.method:<15} {m.k_value:<5} {m.peak_memory_mb:.1f}")
    
    # Custom-specific stats
    print("\n🎯 Custom Implementation Statistics")
    print("-" * 60)
    print(f"{'K':<5} {'Accept Rate':<15} {'Tokens/Round':<15} {'Total Rounds':<15}")
    print("-" * 60)
    
    for m in sorted(custom_results, key=lambda x: x.k_value):
        print(f"{m.k_value:<5} {m.acceptance_rate:.1%}{'':<10} {m.tokens_per_round:.2f}{'':<11} {m.total_rounds}")
    
    print("=" * 100)


def save_results(
    baseline: DetailedMetrics,
    hf_results: List[DetailedMetrics],
    custom_results: List[DetailedMetrics],
    output_path: str
):
    """Save results to JSON file."""
    results = {
        "baseline": asdict(baseline),
        "huggingface": [asdict(m) for m in hf_results],
        "custom": [asdict(m) for m in custom_results]
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detailed Speculative Decoding Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                        help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                        help="Path to draft model")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8],
                        help="K values to test")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of test samples")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--output", type=str, default="benchmark_detailed_results.json",
                        help="Output JSON filename")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace benchmark")
    parser.add_argument("--skip-custom", action="store_true",
                        help="Skip Custom benchmark")
    args = parser.parse_args()
    
    print(f"🚀 Detailed Speculative Decoding Benchmark")
    print(f"   Target: {args.target_model}")
    print(f"   Draft: {args.draft_model}")
    print(f"   K values: {args.k_values}")
    print(f"   Samples: {args.num_samples}")
    print(f"   Max tokens: {args.max_new_tokens}")
    
    # Test prompts - mix of short and longer prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog. This is a test of",
        "In the beginning of time, there was nothing but darkness and",
        "Machine learning is a subset of artificial intelligence that",
        "The capital of France is Paris, which is known for its",
        "Once upon a time in a land far away, there lived a",
        "The theory of relativity, proposed by Albert Einstein,",
        "Python is a popular programming language because it is",
        "Climate change is one of the most pressing issues facing",
        "The human brain contains approximately 86 billion neurons that",
        "Quantum computing represents a paradigm shift in how we",
        "The Renaissance was a period of cultural rebirth in Europe that",
        "Artificial neural networks are inspired by the structure of",
    ][:args.num_samples]
    
    device = args.device
    
    # Load models
    print(f"\n📦 Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16
    ).to(device)
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model.eval()
    draft_model.eval()
    
    # Baseline benchmark
    print(f"\n🏃 Running Baseline benchmark...")
    baseline = measure_baseline(target_model, tokenizer, prompts, args.max_new_tokens, device)
    print(f"   Throughput: {baseline.throughput:.1f} t/s")
    print(f"   TTFT: {baseline.ttft_mean:.1f} ms")
    print(f"   TPOT: {baseline.tpot_mean:.1f} ms")
    
    # HuggingFace benchmarks
    hf_results = []
    if not args.skip_hf:
        print(f"\n🏃 Running HuggingFace benchmarks...")
        for k in args.k_values:
            result = measure_huggingface(
                target_model, draft_model, tokenizer, prompts,
                k, args.max_new_tokens, device
            )
            hf_results.append(result)
            speedup = result.throughput / baseline.throughput
            print(f"   K={k}: {result.throughput:.1f} t/s ({speedup:.2f}x), "
                  f"TTFT={result.ttft_mean:.1f}ms, TPOT={result.tpot_mean:.1f}ms")
    
    # Custom implementation benchmarks
    custom_results = []
    if not args.skip_custom:
        print(f"\n🏃 Running Custom implementation benchmarks...")
        generator = SpeculativeGenerator(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=5,
            max_len=2048,
            device=device,
            use_compile=False
        )
        
        for k in args.k_values:
            result = measure_custom(generator, prompts, k, args.max_new_tokens)
            custom_results.append(result)
            speedup = result.throughput / baseline.throughput
            print(f"   K={k}: {result.throughput:.1f} t/s ({speedup:.2f}x), "
                  f"acceptance={result.acceptance_rate:.1%}, "
                  f"TTFT={result.ttft_mean:.1f}ms, TPOT={result.tpot_mean:.1f}ms")
    
    # Print summary
    print_detailed_summary(baseline, hf_results, custom_results)
    
    # Save results
    save_results(baseline, hf_results, custom_results, args.output)


if __name__ == "__main__":
    main()


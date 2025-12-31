"""
Benchmark: Custom Speculative Decoding vs HuggingFace Implementation

This script compares our custom speculative decoding implementation against
HuggingFace's built-in assistant_model feature.

Metrics:
- Throughput (tokens/second)
- Output correctness (should be identical for greedy decoding)
- Acceptance rate
- Memory usage
"""

import torch
import time
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

from core import SpeculativeGenerator, StaticKVCache


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method: str
    k_value: int
    throughput: float
    acceptance_rate: float
    total_tokens: int
    total_time: float
    memory_mb: float
    outputs: List[str]


def benchmark_huggingface(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    k_value: int,
    max_new_tokens: int,
    device: str
) -> BenchmarkResult:
    """Benchmark HuggingFace's native speculative decoding."""
    
    total_tokens = 0
    total_time = 0.0
    outputs = []
    
    # Warmup
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
    
    # Benchmark
    for prompt in tqdm(prompts, desc=f"HF (K={k_value})", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            output_ids = target_model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=max_new_tokens,
                num_assistant_tokens=k_value,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Count new tokens
        new_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        total_tokens += new_tokens
        total_time += elapsed
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output_text)
    
    throughput = total_tokens / total_time if total_time > 0 else 0
    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return BenchmarkResult(
        method="HuggingFace",
        k_value=k_value,
        throughput=throughput,
        acceptance_rate=0.0,  # HF doesn't expose this directly
        total_tokens=total_tokens,
        total_time=total_time,
        memory_mb=memory_mb,
        outputs=outputs
    )


def benchmark_custom(
    generator: SpeculativeGenerator,
    prompts: List[str],
    k_value: int,
    max_new_tokens: int
) -> BenchmarkResult:
    """Benchmark our custom speculative decoding implementation."""
    
    # Update K value
    generator.K = k_value
    
    total_tokens = 0
    total_time = 0.0
    outputs = []
    total_rounds = 0
    total_accepted = 0
    
    # Warmup
    generator.reset()
    _ = generator.generate("Warmup", max_new_tokens=5)
    torch.cuda.synchronize()
    
    # Benchmark
    for prompt in tqdm(prompts, desc=f"Custom (K={k_value})", leave=False):
        generator.reset()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output_text = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        stats = generator.get_stats()
        total_tokens += stats["total_tokens"]
        total_time += elapsed
        total_rounds += stats["total_rounds"]
        total_accepted += stats["total_accepted"]
        
        outputs.append(output_text)
    
    throughput = total_tokens / total_time if total_time > 0 else 0
    acceptance_rate = generator.get_acceptance_rate()
    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return BenchmarkResult(
        method="Custom",
        k_value=k_value,
        throughput=throughput,
        acceptance_rate=acceptance_rate,
        total_tokens=total_tokens,
        total_time=total_time,
        memory_mb=memory_mb,
        outputs=outputs
    )


def benchmark_baseline(
    target_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str
) -> BenchmarkResult:
    """Benchmark baseline (no speculative decoding)."""
    
    total_tokens = 0
    total_time = 0.0
    outputs = []
    
    # Warmup
    dummy = tokenizer("Warmup", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = target_model.generate(**dummy, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    
    # Benchmark
    for prompt in tqdm(prompts, desc="Baseline", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            output_ids = target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        new_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
        total_tokens += new_tokens
        total_time += elapsed
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output_text)
    
    throughput = total_tokens / total_time if total_time > 0 else 0
    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return BenchmarkResult(
        method="Baseline",
        k_value=0,
        throughput=throughput,
        acceptance_rate=0.0,
        total_tokens=total_tokens,
        total_time=total_time,
        memory_mb=memory_mb,
        outputs=outputs
    )


def verify_correctness(results: List[BenchmarkResult]) -> bool:
    """Verify that all methods produce identical outputs."""
    if len(results) < 2:
        return True
    
    # Compare outputs from different methods
    baseline_outputs = None
    for r in results:
        if r.method == "Baseline":
            baseline_outputs = r.outputs
            break
    
    if baseline_outputs is None:
        baseline_outputs = results[0].outputs
    
    all_match = True
    for r in results:
        for i, (out1, out2) in enumerate(zip(baseline_outputs, r.outputs)):
            if out1 != out2:
                print(f"Output mismatch at sample {i}:")
                print(f"  Baseline: {out1[:100]}...")
                print(f"  {r.method} (K={r.k_value}): {out2[:100]}...")
                all_match = False
    
    return all_match


def plot_results(
    results: List[BenchmarkResult],
    baseline_throughput: float,
    save_path: str = "benchmark_comparison.png"
):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate results by method
    hf_results = [r for r in results if r.method == "HuggingFace"]
    custom_results = [r for r in results if r.method == "Custom"]
    
    # Plot 1: Throughput
    ax1.set_title("Throughput Comparison: Custom vs HuggingFace", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Num Assistant Tokens (K)", fontsize=12)
    ax1.set_ylabel("Throughput (tokens/s)", fontsize=12)
    
    # Baseline
    ax1.axhline(y=baseline_throughput, color='gray', linestyle='--', linewidth=2, 
                label=f'Baseline ({baseline_throughput:.1f} t/s)')
    
    # HuggingFace
    if hf_results:
        k_values = [r.k_value for r in hf_results]
        throughputs = [r.throughput for r in hf_results]
        ax1.plot(k_values, throughputs, 'b-o', linewidth=2, markersize=8, label='HuggingFace')
    
    # Custom
    if custom_results:
        k_values = [r.k_value for r in custom_results]
        throughputs = [r.throughput for r in custom_results]
        ax1.plot(k_values, throughputs, 'r-s', linewidth=2, markersize=8, label='Custom')
    
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Speedup
    ax2.set_title("Speedup vs Baseline", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Num Assistant Tokens (K)", fontsize=12)
    ax2.set_ylabel("Speedup (x)", fontsize=12)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    
    if hf_results:
        k_values = [r.k_value for r in hf_results]
        speedups = [r.throughput / baseline_throughput for r in hf_results]
        ax2.plot(k_values, speedups, 'b-o', linewidth=2, markersize=8, label='HuggingFace')
    
    if custom_results:
        k_values = [r.k_value for r in custom_results]
        speedups = [r.throughput / baseline_throughput for r in custom_results]
        ax2.plot(k_values, speedups, 'r-s', linewidth=2, markersize=8, label='Custom')
    
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Results saved to: {save_path}")


def print_summary(results: List[BenchmarkResult], baseline_throughput: float):
    """Print summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<15} {'K':<5} {'Throughput':<15} {'Speedup':<10} {'Accept Rate':<12}")
    print("-"*60)
    
    print(f"{'Baseline':<15} {'-':<5} {baseline_throughput:<15.1f} {'1.00x':<10} {'-':<12}")
    
    for r in sorted(results, key=lambda x: (x.method, x.k_value)):
        speedup = r.throughput / baseline_throughput
        acc_str = f"{r.acceptance_rate:.2%}" if r.acceptance_rate > 0 else "-"
        print(f"{r.method:<15} {r.k_value:<5} {r.throughput:<15.1f} {speedup:.2f}x{'':<5} {acc_str:<12}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding implementations")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                        help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                        help="Path to draft model")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 7],
                        help="K values to test")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of test samples")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--output", type=str, default="benchmark_custom_vs_hf.png",
                        help="Output plot filename")
    args = parser.parse_args()
    
    print(f"🚀 Speculative Decoding Benchmark")
    print(f"   Target: {args.target_model}")
    print(f"   Draft: {args.draft_model}")
    print(f"   K values: {args.k_values}")
    print(f"   Samples: {args.num_samples}")
    
    # Test prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog. This is a test of",
        "In the beginning of time, there was nothing but darkness and",
        "Machine learning is a subset of artificial intelligence that",
        "The capital of France is Paris, which is known for its",
        "Once upon a time in a land far away, there lived a",
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
    
    torch.cuda.reset_peak_memory_stats()
    
    all_results = []
    
    # Baseline benchmark
    print(f"\n🏃 Running Baseline...")
    baseline_result = benchmark_baseline(
        target_model, tokenizer, prompts, args.max_new_tokens, device
    )
    print(f"   Baseline throughput: {baseline_result.throughput:.1f} t/s")
    
    # HuggingFace benchmark
    print(f"\n🏃 Running HuggingFace benchmarks...")
    for k in args.k_values:
        result = benchmark_huggingface(
            target_model, draft_model, tokenizer, prompts,
            k, args.max_new_tokens, device
        )
        all_results.append(result)
        print(f"   K={k}: {result.throughput:.1f} t/s ({result.throughput/baseline_result.throughput:.2f}x)")
    
    # Custom implementation benchmark
    print(f"\n🏃 Running Custom implementation benchmarks...")
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=not args.no_compile
    )
    
    for k in args.k_values:
        result = benchmark_custom(
            generator, prompts, k, args.max_new_tokens
        )
        all_results.append(result)
        print(f"   K={k}: {result.throughput:.1f} t/s ({result.throughput/baseline_result.throughput:.2f}x), "
              f"acceptance: {result.acceptance_rate:.2%}")
    
    # Verify correctness
    print(f"\n🔍 Verifying output correctness...")
    all_results_with_baseline = [baseline_result] + all_results
    is_correct = verify_correctness(all_results_with_baseline)
    if is_correct:
        print("   ✅ All outputs match!")
    else:
        print("   ⚠️ Some outputs differ (this may be expected due to slight implementation differences)")
    
    # Print summary
    print_summary(all_results, baseline_result.throughput)
    
    # Plot results
    plot_results(all_results, baseline_result.throughput, args.output)


if __name__ == "__main__":
    main()


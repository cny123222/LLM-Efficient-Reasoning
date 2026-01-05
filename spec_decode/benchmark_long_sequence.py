"""
Long Sequence Benchmark: Test StreamingLLM with forced long generation

This script forces the model to generate long sequences by:
1. Using prompts that encourage long outputs
2. Setting a very high eos_token_id to prevent early stopping
3. Tracking actual memory growth over sequence length

Usage:
    python benchmark_long_sequence.py --max-new-tokens 500 1000 2000 4000
"""

import torch
import time
import argparse
import json
import gc
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()

from core import SpeculativeGenerator, StreamingSpeculativeGenerator


@dataclass 
class LongSeqMetrics:
    """Metrics for long sequence generation."""
    method: str
    target_tokens: int
    actual_tokens: int
    max_cache_len: int
    throughput: float
    total_time: float
    peak_memory_mb: float
    memory_start_mb: float
    memory_growth_mb: float
    acceptance_rate: float
    compression_count: int
    tokens_evicted: int


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_long_prompt():
    """Get a prompt that encourages long generation."""
    return """Write a very detailed and comprehensive technical explanation about the following topic. Please be thorough and cover all aspects extensively, including history, current state, future directions, and practical applications. Continue writing until you have covered everything in great detail.

Topic: The development and optimization of large language models for efficient inference.

Begin your detailed explanation:

Large language models have become"""


class ForcedLongGenerator(SpeculativeGenerator):
    """Generator that forces long sequences by ignoring EOS."""
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        # Save original EOS
        original_eos = self.tokenizer.eos_token_id
        
        # Set EOS to a very high value to prevent early stopping
        # This forces the model to generate until max_new_tokens
        self.tokenizer.eos_token_id = 999999
        
        try:
            result = super().generate(prompt, max_new_tokens)
        finally:
            # Restore original EOS
            self.tokenizer.eos_token_id = original_eos
        
        return result


class ForcedLongStreamingGenerator(StreamingSpeculativeGenerator):
    """Streaming generator that forces long sequences."""
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        original_eos = self.tokenizer.eos_token_id
        self.tokenizer.eos_token_id = 999999
        
        try:
            result = super().generate(prompt, max_new_tokens)
        finally:
            self.tokenizer.eos_token_id = original_eos
        
        return result


def measure_standard(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    k_value: int,
    device: str
) -> LongSeqMetrics:
    """Measure standard speculative decoding."""
    cleanup()
    memory_start = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    generator = ForcedLongGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=k_value,
        max_len=8192,
        device=device,
        use_compile=False
    )
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    stats = generator.get_stats()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    
    return LongSeqMetrics(
        method="standard",
        target_tokens=max_new_tokens,
        actual_tokens=stats["total_tokens"],
        max_cache_len=0,
        throughput=stats["total_tokens"] / total_time,
        total_time=total_time,
        peak_memory_mb=peak_memory,
        memory_start_mb=memory_start,
        memory_growth_mb=peak_memory - memory_start,
        acceptance_rate=stats["acceptance_rate"],
        compression_count=0,
        tokens_evicted=0
    )


def measure_streaming(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_cache_len: int,
    k_value: int,
    device: str
) -> LongSeqMetrics:
    """Measure streaming speculative decoding."""
    cleanup()
    memory_start = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    generator = ForcedLongStreamingGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=k_value,
        max_len=8192,
        device=device,
        use_compile=False,
        max_cache_len=max_cache_len,
        start_size=4,
        recent_size=max_cache_len - 4
    )
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    stats = generator.get_stats()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    
    return LongSeqMetrics(
        method="streaming",
        target_tokens=max_new_tokens,
        actual_tokens=stats["total_tokens"],
        max_cache_len=max_cache_len,
        throughput=stats["total_tokens"] / total_time,
        total_time=total_time,
        peak_memory_mb=peak_memory,
        memory_start_mb=memory_start,
        memory_growth_mb=peak_memory - memory_start,
        acceptance_rate=stats["acceptance_rate"],
        compression_count=stats.get("compression_count", 0),
        tokens_evicted=stats.get("tokens_evicted", 0)
    )


def plot_results(results: List[LongSeqMetrics], save_path: str):
    """Plot comparison results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group results
    standard_results = [r for r in results if r.method == "standard"]
    streaming_results = [r for r in results if r.method == "streaming"]
    
    # Sort by actual tokens
    standard_results.sort(key=lambda x: x.target_tokens)
    
    # Get unique cache sizes
    cache_sizes = sorted(set(r.max_cache_len for r in streaming_results if r.max_cache_len > 0))
    
    # Colors
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
    
    # Plot 1: Throughput vs Sequence Length
    ax1 = axes[0, 0]
    if standard_results:
        x_std = [r.target_tokens for r in standard_results]
        y_std = [r.throughput for r in standard_results]
        ax1.plot(x_std, y_std, 'o-', color='#2C3E50', linewidth=2, markersize=8, label='Standard')
    
    for i, cache_size in enumerate(cache_sizes):
        stream_cache = [r for r in streaming_results if r.max_cache_len == cache_size]
        stream_cache.sort(key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.throughput for r in stream_cache]
            ax1.plot(x, y, 'o--', color=colors[i % len(colors)], linewidth=2, markersize=8, 
                    label=f'Stream (cache={cache_size})')
    
    ax1.set_xlabel('Target Tokens')
    ax1.set_ylabel('Throughput (tokens/s)')
    ax1.set_title('Throughput vs Sequence Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory Growth vs Sequence Length
    ax2 = axes[0, 1]
    if standard_results:
        x_std = [r.target_tokens for r in standard_results]
        y_std = [r.memory_growth_mb for r in standard_results]
        ax2.plot(x_std, y_std, 'o-', color='#2C3E50', linewidth=2, markersize=8, label='Standard')
    
    for i, cache_size in enumerate(cache_sizes):
        stream_cache = [r for r in streaming_results if r.max_cache_len == cache_size]
        stream_cache.sort(key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.memory_growth_mb for r in stream_cache]
            ax2.plot(x, y, 'o--', color=colors[i % len(colors)], linewidth=2, markersize=8,
                    label=f'Stream (cache={cache_size})')
    
    ax2.set_xlabel('Target Tokens')
    ax2.set_ylabel('Memory Growth (MB)')
    ax2.set_title('Memory Growth vs Sequence Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Compression Events
    ax3 = axes[1, 0]
    for i, cache_size in enumerate(cache_sizes):
        stream_cache = [r for r in streaming_results if r.max_cache_len == cache_size]
        stream_cache.sort(key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.compression_count for r in stream_cache]
            ax3.bar([xi + i*0.2 - 0.2 for xi in range(len(x))], y, 0.2, 
                   label=f'cache={cache_size}', color=colors[i % len(colors)])
    
    ax3.set_xlabel('Target Tokens')
    ax3.set_ylabel('Compression Events')
    ax3.set_title('KV Cache Compression Events')
    if streaming_results:
        ax3.set_xticks(range(len(set(r.target_tokens for r in streaming_results))))
        ax3.set_xticklabels(sorted(set(r.target_tokens for r in streaming_results)))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Method', 'Tokens', 'Cache', 'Throughput', 'Memory Δ', 'Compress']
    
    for r in results:
        table_data.append([
            r.method,
            str(r.actual_tokens),
            str(r.max_cache_len) if r.max_cache_len > 0 else '-',
            f'{r.throughput:.1f}',
            f'{r.memory_growth_mb:.0f} MB',
            str(r.compression_count) if r.compression_count > 0 else '-'
        ])
    
    table = ax4.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#4ECDC4']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Long Sequence Generation: Standard vs StreamingLLM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Long Sequence Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--max-cache-lens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--k-value", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-json", type=str, default="benchmark_long_seq_results.json")
    parser.add_argument("--output-plot", type=str, default="benchmark_long_seq.png")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 LONG SEQUENCE GENERATION BENCHMARK")
    print("=" * 80)
    print(f"   Target Model:  {args.target_model}")
    print(f"   Draft Model:   {args.draft_model}")
    print(f"   Token Lengths: {args.max_new_tokens}")
    print(f"   Cache Sizes:   {args.max_cache_lens}")
    print(f"   K Value:       {args.k_value}")
    print("=" * 80)
    
    # Load models
    print("\n🔄 Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.float16
    ).to(args.device)
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=torch.float16
    ).to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model.eval()
    draft_model.eval()
    
    prompt = get_long_prompt()
    results = []
    
    # Test standard speculative decoding
    print("\n🏃 Running Standard Speculative Decoding...")
    for max_tokens in args.max_new_tokens:
        print(f"  Testing max_new_tokens={max_tokens}...")
        result = measure_standard(
            target_model, draft_model, tokenizer,
            prompt, max_tokens, args.k_value, args.device
        )
        results.append(result)
        print(f"    Generated: {result.actual_tokens} tokens, "
              f"Throughput: {result.throughput:.1f} t/s, "
              f"Memory Δ: {result.memory_growth_mb:.0f} MB")
    
    # Test streaming speculative decoding
    print("\n🏃 Running StreamingLLM Speculative Decoding...")
    for max_tokens in args.max_new_tokens:
        for cache_len in args.max_cache_lens:
            print(f"  Testing max_new_tokens={max_tokens}, max_cache_len={cache_len}...")
            result = measure_streaming(
                target_model, draft_model, tokenizer,
                prompt, max_tokens, cache_len, args.k_value, args.device
            )
            results.append(result)
            print(f"    Generated: {result.actual_tokens} tokens, "
                  f"Throughput: {result.throughput:.1f} t/s, "
                  f"Compressions: {result.compression_count}, "
                  f"Memory Δ: {result.memory_growth_mb:.0f} MB")
    
    # Print summary
    print("\n" + "=" * 100)
    print("LONG SEQUENCE BENCHMARK RESULTS")
    print("=" * 100)
    print(f"\n{'Method':<12} {'Target':<8} {'Actual':<8} {'Cache':<8} {'Throughput':<12} {'Memory Δ':<12} {'Compress':<10}")
    print("-" * 100)
    
    for r in results:
        cache_str = str(r.max_cache_len) if r.max_cache_len > 0 else "N/A"
        compress_str = str(r.compression_count) if r.compression_count > 0 else "0"
        print(f"{r.method:<12} {r.target_tokens:<8} {r.actual_tokens:<8} {cache_str:<8} "
              f"{r.throughput:.1f} t/s{'':<5} {r.memory_growth_mb:.0f} MB{'':<6} {compress_str:<10}")
    
    # Key insight
    print("\n📊 KEY INSIGHTS:")
    standard_results = [r for r in results if r.method == "standard"]
    streaming_results = [r for r in results if r.method == "streaming"]
    
    if standard_results and streaming_results:
        # Compare at longest sequence
        longest_standard = max(standard_results, key=lambda x: x.actual_tokens)
        longest_streaming = [r for r in streaming_results if r.target_tokens == longest_standard.target_tokens]
        
        if longest_streaming:
            min_mem_stream = min(longest_streaming, key=lambda x: x.memory_growth_mb)
            print(f"   At {longest_standard.target_tokens} tokens:")
            print(f"   - Standard: {longest_standard.memory_growth_mb:.0f} MB memory growth")
            print(f"   - Stream (cache={min_mem_stream.max_cache_len}): {min_mem_stream.memory_growth_mb:.0f} MB memory growth")
            
            if longest_standard.memory_growth_mb > min_mem_stream.memory_growth_mb:
                savings = (1 - min_mem_stream.memory_growth_mb / longest_standard.memory_growth_mb) * 100
                print(f"   - Memory Savings: {savings:.1f}%")
    
    # Save results
    with open(args.output_json, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n✅ Results saved to: {args.output_json}")
    
    # Plot
    plot_results(results, args.output_plot)


if __name__ == "__main__":
    main()









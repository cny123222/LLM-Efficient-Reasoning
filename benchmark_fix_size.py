#!/usr/bin/env python3
"""
Benchmark Script for Fixed-Size KV Cache Compression

This script benchmarks the fix_size_l2_compress method with different
eviction strategies on the PG-19 long text dataset.

Usage:
    # Full benchmark with all baselines
    python benchmark_fix_size.py --fix_kv_sizes 256,512 --strategies keep_low,keep_high,random --keep_ratios 0.1,0.2,0.3

    # Disable baselines for faster testing
    python benchmark_fix_size.py --fix_kv_sizes 512 --strategies keep_low --keep_ratios 0.2 --no_baseline --no_recent_only

    # Only test specific configurations
    python benchmark_fix_size.py --fix_kv_sizes 256,512,1024 --strategies keep_low --keep_ratios 0.1,0.2,0.3,0.5

Baselines (optional):
    - baseline: No compression (full KV cache) [--no_baseline to disable]
    - recent_only: Sliding window, keep only most recent fix_kv_size tokens [--no_recent_only to disable]

Eviction Strategies:
    - keep_low: Keep tokens with low L2 norm (important tokens) - BEST
    - keep_high: Keep tokens with high L2 norm - WORST  
    - random: Random eviction - MEDIUM

Expected Results:
    keep_low > random > recent_only > keep_high
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knormpress import evaluate_fix_size_compression, evaluate_with_compression
from knormpress.compress import fix_size_l2_compress, to_dynamic_cache

# Local PG-19 dataset path
LOCAL_PG19_PATH = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/data/pg19.parquet"


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str = "EleutherAI/pythia-70m-deduped"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def load_pg19_samples(num_samples: int = 3):
    """Load samples from PG-19 dataset."""
    print("\nLoading PG-19 dataset...")
    
    if os.path.exists(LOCAL_PG19_PATH):
        print(f"  Found local file: {LOCAL_PG19_PATH}")
        try:
            dataset = load_dataset(
                "parquet",
                data_files={'test': LOCAL_PG19_PATH},
                split="test"
            )
            print(f"  Loaded {len(dataset)} samples from local file")
        except Exception as e:
            print(f"  Failed to load local file: {e}")
            return []
    else:
        print("  Local file not found.")
        return []
    
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        text = sample.get("text", "")
        if len(text) > 10000:
            samples.append(text)
            print(f"  Sample {i+1}: {len(text)} characters")
    
    print(f"  Total: {len(samples)} samples loaded")
    return samples


def benchmark_fix_size(
    model,
    tokenizer,
    text: str,
    fix_kv_sizes: list,
    strategies: list,
    keep_ratios: list,
    max_tokens: int,
    skip_layers: list,
    device,
    enable_baseline: bool = True,
    enable_recent_only: bool = True,
):
    """
    Run benchmark with different fix_kv_sizes, strategies, and keep_ratios.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        fix_kv_sizes: List of fixed KV cache sizes to test
        strategies: List of eviction strategies to test
        keep_ratios: List of keep_ratio values to test
        max_tokens: Maximum tokens for evaluation
        skip_layers: Layers to skip compression
        device: Device to use
        enable_baseline: Whether to run baseline (no compression)
        enable_recent_only: Whether to run recent_only (sliding window) baseline
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Baseline: no compression (optional)
    if enable_baseline:
        print("\n  [Baseline] No compression...")
        baseline = evaluate_with_compression(
            model=model,
            tokenizer=tokenizer,
            text=text,
            keep_ratio=1.0,
            prune_after=999999,
            max_tokens=max_tokens,
            device=device,
            show_progress=True,
        )
        baseline['fix_kv_size'] = 'baseline'
        baseline['strategy'] = 'none'
        baseline['keep_ratio'] = 1.0
        results.append(baseline)
        print(f"    PPL: {baseline['perplexity']:.2f} | Acc: {baseline['accuracy']:.2%}")
    
    # Test each fix_kv_size
    for fix_kv_size in fix_kv_sizes:
        
        # Recent-only baseline: sliding window (optional)
        if enable_recent_only:
            print(f"\n  [fix_kv={fix_kv_size}] recent_only (sliding window)...")
            recent_only_result = evaluate_fix_size_compression(
                model=model,
                tokenizer=tokenizer,
                text=text,
                fix_kv_size=fix_kv_size,
                keep_ratio=1.0,  # 100% from recent = sliding window
                strategy="keep_low",  # Strategy doesn't matter when keep_ratio=1.0
                skip_layers=skip_layers,
                max_tokens=max_tokens,
                device=device,
                show_progress=True,
            )
            recent_only_result['strategy'] = 'recent_only'
            recent_only_result['keep_ratio'] = 1.0
            results.append(recent_only_result)
            print(f"    PPL: {recent_only_result['perplexity']:.2f} | Acc: {recent_only_result['accuracy']:.2%}")
        
        # Test each combination of strategy and keep_ratio
        for strategy in strategies:
            for keep_ratio in keep_ratios:
                print(f"\n  [fix_kv={fix_kv_size}] {strategy}, keep_ratio={keep_ratio}...")
                
                result = evaluate_fix_size_compression(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    fix_kv_size=fix_kv_size,
                    keep_ratio=keep_ratio,
                    strategy=strategy,
                    skip_layers=skip_layers,
                    max_tokens=max_tokens,
                    device=device,
                    show_progress=True,
                )
                result['keep_ratio'] = keep_ratio
                
                results.append(result)
                print(f"    PPL: {result['perplexity']:.2f} | Acc: {result['accuracy']:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Fixed-Size KV Cache Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark
  python benchmark_fix_size.py --fix_kv_sizes 256,512 --strategies keep_low,random --keep_ratios 0.1,0.2,0.3

  # Quick test without baselines
  python benchmark_fix_size.py --fix_kv_sizes 512 --strategies keep_low --keep_ratios 0.2 --no_baseline --no_recent_only

  # Compare different keep_ratios
  python benchmark_fix_size.py --fix_kv_sizes 512 --strategies keep_low --keep_ratios 0.0,0.1,0.2,0.3,0.5
        """
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="Model ID from HuggingFace"
    )
    parser.add_argument(
        "--fix_kv_sizes",
        type=str,
        default="256,512",
        help="Comma-separated list of fix_kv_size values to test"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="keep_low,keep_high,random",
        help="Comma-separated list of strategies: keep_low, keep_high, random"
    )
    parser.add_argument(
        "--keep_ratios",
        type=str,
        default="0.2",
        help="Comma-separated list of keep_ratio values (0.0-1.0) to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of PG-19 samples to test"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1500,
        help="Maximum tokens for evaluation"
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default="0,1",
        help="Comma-separated list of layer indices to skip"
    )
    # Baseline control options
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Disable baseline (no compression) benchmark"
    )
    parser.add_argument(
        "--no_recent_only",
        action="store_true",
        help="Disable recent_only (sliding window) baseline benchmark"
    )
    
    args = parser.parse_args()
    
    # Parse comma-separated lists
    fix_kv_sizes = [int(x) for x in args.fix_kv_sizes.split(",")]
    strategies = [x.strip() for x in args.strategies.split(",")]
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    skip_layers = [int(x) for x in args.skip_layers.split(",")]
    
    enable_baseline = not args.no_baseline
    enable_recent_only = not args.no_recent_only
    
    print("="*70)
    print("Fixed-Size KV Cache Compression Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Fix KV sizes: {fix_kv_sizes}")
    print(f"  Strategies: {strategies}")
    print(f"  Keep ratios: {keep_ratios}")
    print(f"  Skip layers: {skip_layers}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Enable baseline: {enable_baseline}")
    print(f"  Enable recent_only: {enable_recent_only}")
    
    model, tokenizer, device = load_model_and_tokenizer(args.model_id)
    
    samples = load_pg19_samples(args.num_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return
    
    all_results = []
    
    for i, text in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{len(samples)} ({len(text)} characters)")
        print("="*70)
        
        results = benchmark_fix_size(
            model=model,
            tokenizer=tokenizer,
            text=text,
            fix_kv_sizes=fix_kv_sizes,
            strategies=strategies,
            keep_ratios=keep_ratios,
            max_tokens=args.max_tokens,
            skip_layers=skip_layers,
            device=device,
            enable_baseline=enable_baseline,
            enable_recent_only=enable_recent_only,
        )
        
        all_results.extend(results)
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS")
    print("="*80)
    
    # Group by (fix_kv_size, strategy, keep_ratio)
    grouped = {}
    for r in all_results:
        key = (r['fix_kv_size'], r['strategy'], r.get('keep_ratio', 1.0))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    print(f"\n{'Fix KV':<10} {'Strategy':<12} {'Keep Ratio':<12} {'PPL':>10} {'Accuracy':>12} {'Cache':>10}")
    print("-"*70)
    
    baseline_ppl = None
    baseline_acc = None
    
    # Process baseline first
    if enable_baseline:
        for (fix_kv_size, strategy, keep_ratio), results in grouped.items():
            if fix_kv_size == 'baseline':
                avg_ppl = np.mean([r['perplexity'] for r in results])
                avg_acc = np.mean([r['accuracy'] for r in results])
                avg_cache = np.mean([r['final_cache_size'] for r in results])
                baseline_ppl = avg_ppl
                baseline_acc = avg_acc
                print(f"{'baseline':<10} {'none':<12} {'-':<12} {avg_ppl:>10.2f} {avg_acc:>11.2%} {avg_cache:>10.0f}")
                break
    
    # Sort results for display
    def sort_key(item):
        key, _ = item
        fix_kv, strategy, keep_ratio = key
        if fix_kv == 'baseline':
            return (0, '', 0)
        strategy_order = {'recent_only': 0, 'keep_low': 1, 'random': 2, 'keep_high': 3}
        return (fix_kv, strategy_order.get(strategy, 99), keep_ratio)
    
    # Process other results
    for (fix_kv_size, strategy, keep_ratio), results in sorted(
        [(k, v) for k, v in grouped.items() if k[0] != 'baseline'],
        key=sort_key
    ):
        avg_ppl = np.mean([r['perplexity'] for r in results])
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_cache = np.mean([r['final_cache_size'] for r in results])
        
        kr_str = f"{keep_ratio:.1f}" if strategy != 'recent_only' else "1.0 (sw)"
        print(f"{fix_kv_size:<10} {strategy:<12} {kr_str:<12} {avg_ppl:>10.2f} {avg_acc:>11.2%} {avg_cache:>10.0f}")
    
    print("="*70)
    
    # Print comparison
    if baseline_ppl is not None:
        print("\nComparison with baseline:")
        for (fix_kv_size, strategy, keep_ratio), results in sorted(
            [(k, v) for k, v in grouped.items() if k[0] != 'baseline'],
            key=sort_key
        ):
            avg_ppl = np.mean([r['perplexity'] for r in results])
            avg_acc = np.mean([r['accuracy'] for r in results])
            
            ppl_change = (avg_ppl / baseline_ppl - 1) * 100
            acc_change = (avg_acc / baseline_acc - 1) * 100
            
            kr_str = f"kr={keep_ratio:.1f}" if strategy != 'recent_only' else "sliding_window"
            print(f"  fix_kv={fix_kv_size}, {strategy}, {kr_str}: PPL {ppl_change:+.1f}%, Acc {acc_change:+.1f}%")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()

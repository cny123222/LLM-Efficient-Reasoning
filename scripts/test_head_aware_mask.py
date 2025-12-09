#!/usr/bin/env python3
"""
Test Head-Aware Attention Mask Strategy

This script compares three attention masking strategies:
1. Baseline: Full context (standard causal mask)
2. Uniform StreamingLLM: All heads use same sink+window mask
3. Head-Aware: Different heads use different masks based on their classification

The goal is to validate whether head-aware masking can achieve better PPL/accuracy
than uniform masking at equivalent "effective context" sizes.

Usage:
    # Quick test with small model (MPS compatible)
    python scripts/test_head_aware_mask.py --model EleutherAI/pythia-70m --max-tokens 500
    
    # Full test with pythia-2.8b
    python scripts/test_head_aware_mask.py --model EleutherAI/pythia-2.8b --max-tokens 2000
    
    # Use specific classifications file
    python scripts/test_head_aware_mask.py \
        --model EleutherAI/pythia-2.8b \
        --classifications results/attention_analysis_pythia-2.8b/head_classifications.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, device: torch.device, use_bf16: bool = True):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    print(f"Device: {device}")
    
    # Determine dtype
    if device.type == "cuda" and use_bf16:
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float32  # MPS works better with float32
    else:
        dtype = torch.float32
    
    print(f"Using dtype: {dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",  # Required for custom attention masks
    )
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_sample_text(num_chars: int = 50000) -> str:
    """Load sample text from PG19 dataset."""
    data_path = project_root / "data" / "pg19.parquet"
    
    if data_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            text = df.iloc[0]['text']
            return text[:num_chars]
        except Exception as e:
            print(f"Warning: Could not load PG19 data: {e}")
    
    # Fallback to synthetic text
    print("Using synthetic text for testing...")
    base = "The quick brown fox jumps over the lazy dog. " * 100
    return (base * (num_chars // len(base) + 1))[:num_chars]


def evaluate_baseline(model, tokenizer, text: str, max_tokens: int, device: torch.device):
    """Evaluate with full context (no masking)."""
    from kvcompress.evaluate import evaluate_with_compression
    
    return evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        compress_fn=None,  # No compression
        max_tokens=max_tokens,
        device=device,
        show_progress=True,
    )


def evaluate_uniform_mask(
    model, tokenizer, text: str, max_tokens: int, device: torch.device,
    sink_size: int = 4, window_size: int = 508
):
    """Evaluate with uniform StreamingLLM-style mask for all heads."""
    from kvcompress.evaluate import evaluate_with_head_aware_mask
    from kvcompress.methods.head_aware_compress import HeadAwareMaskGenerator
    
    # Create uniform mask generator
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    mask_gen = HeadAwareMaskGenerator.create_uniform(
        num_layers=num_layers,
        num_heads=num_heads,
        sink_size=sink_size,
        window_size=window_size,
    )
    
    return evaluate_with_head_aware_mask(
        model=model,
        tokenizer=tokenizer,
        text=text,
        mask_generator=mask_gen,
        max_tokens=max_tokens,
        device=device,
        show_progress=True,
    )


def evaluate_head_aware_mask(
    model, tokenizer, text: str, max_tokens: int, device: torch.device,
    classifications_path: str
):
    """Evaluate with head-aware masks based on classifications."""
    from kvcompress.evaluate import evaluate_with_head_aware_mask
    from kvcompress.methods.head_aware_compress import HeadAwareMaskGenerator
    
    if not os.path.exists(classifications_path):
        raise FileNotFoundError(f"Classifications file not found: {classifications_path}")
    
    mask_gen = HeadAwareMaskGenerator.from_classifications(classifications_path)
    
    return evaluate_with_head_aware_mask(
        model=model,
        tokenizer=tokenizer,
        text=text,
        mask_generator=mask_gen,
        max_tokens=max_tokens,
        device=device,
        show_progress=True,
    )


def print_results(results: dict, title: str):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"  PPL:              {results['perplexity']:.2f}")
    print(f"  Accuracy:         {results['accuracy']:.2%}")
    print(f"  Tokens evaluated: {results['num_tokens']}")
    print(f"  Cache size:       {results['final_cache_size']}")
    if 'effective_context' in results:
        print(f"  Effective context:{results['effective_context']:.1f}")
    print(f"  TTFT:             {results['ttft']:.4f}s")
    print(f"  TPOT:             {results['tpot']:.4f}s")
    print(f"  Throughput:       {results['throughput']:.2f} tok/s")
    print(f"  Total time:       {results['total_time']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Test head-aware attention mask strategy")
    parser.add_argument(
        "--model", type=str, default="EleutherAI/pythia-70m",
        help="Model ID (default: pythia-70m for quick testing)"
    )
    parser.add_argument(
        "--classifications", type=str, default=None,
        help="Path to head_classifications.json (auto-detected if not specified)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500,
        help="Maximum tokens to evaluate (default: 500)"
    )
    parser.add_argument(
        "--sink-size", type=int, default=4,
        help="Sink size for uniform mask (default: 4)"
    )
    parser.add_argument(
        "--window-size", type=int, default=508,
        help="Window size for uniform mask (default: 508)"
    )
    parser.add_argument(
        "--no-bf16", action="store_true",
        help="Disable bfloat16 (use float32)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline evaluation (faster for testing masks only)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Auto-detect classifications path
    if args.classifications is None:
        model_name = args.model.split("/")[-1]
        default_path = project_root / "results" / f"attention_analysis_{model_name}" / "head_classifications.json"
        if default_path.exists():
            args.classifications = str(default_path)
            print(f"Auto-detected classifications: {args.classifications}")
        else:
            print(f"Warning: No classifications file found at {default_path}")
            print("Will skip head-aware evaluation.")
    
    # Setup
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        args.model, device, use_bf16=not args.no_bf16
    )
    
    # Load text
    text = load_sample_text(num_chars=args.max_tokens * 10)
    
    results = {}
    
    print("\n" + "="*70)
    print("HEAD-AWARE ATTENTION MASK VALIDATION TEST")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Device: {device}")
    print(f"Classifications: {args.classifications}")
    
    # Test 1: Baseline (full context)
    if not args.skip_baseline:
        print("\n" + "-"*70)
        print("TEST 1: Baseline (Full Context)")
        print("-"*70)
        results['baseline'] = evaluate_baseline(
            model, tokenizer, text, args.max_tokens, device
        )
        print_results(results['baseline'], "Baseline Results")
    
    # Test 2: Uniform mask (StreamingLLM style)
    print("\n" + "-"*70)
    print(f"TEST 2: Uniform Mask (sink={args.sink_size}, window={args.window_size})")
    print("-"*70)
    results['uniform_mask'] = evaluate_uniform_mask(
        model, tokenizer, text, args.max_tokens, device,
        sink_size=args.sink_size, window_size=args.window_size
    )
    print_results(results['uniform_mask'], f"Uniform Mask Results (sink={args.sink_size}, window={args.window_size})")
    
    # Test 3: Head-aware mask
    if args.classifications and os.path.exists(args.classifications):
        print("\n" + "-"*70)
        print("TEST 3: Head-Aware Mask")
        print("-"*70)
        results['head_aware_mask'] = evaluate_head_aware_mask(
            model, tokenizer, text, args.max_tokens, device,
            classifications_path=args.classifications
        )
        print_results(results['head_aware_mask'], "Head-Aware Mask Results")
    else:
        print("\n" + "-"*70)
        print("TEST 3: Head-Aware Mask - SKIPPED (no classifications file)")
        print("-"*70)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    header = f"{'Method':<25} {'PPL':>10} {'Accuracy':>10} {'Eff.Ctx':>10}"
    print(header)
    print("-" * len(header))
    
    if 'baseline' in results:
        r = results['baseline']
        print(f"{'Baseline (full)':<25} {r['perplexity']:>10.2f} {r['accuracy']:>10.2%} {'N/A':>10}")
    
    if 'uniform_mask' in results:
        r = results['uniform_mask']
        eff = r.get('effective_context', args.sink_size + args.window_size)
        print(f"{'Uniform mask':<25} {r['perplexity']:>10.2f} {r['accuracy']:>10.2%} {eff:>10.1f}")
    
    if 'head_aware_mask' in results:
        r = results['head_aware_mask']
        eff = r.get('effective_context', 0)
        print(f"{'Head-aware mask':<25} {r['perplexity']:>10.2f} {r['accuracy']:>10.2%} {eff:>10.1f}")
    
    # Analysis
    if 'uniform_mask' in results and 'head_aware_mask' in results:
        uniform_ppl = results['uniform_mask']['perplexity']
        head_aware_ppl = results['head_aware_mask']['perplexity']
        ppl_diff = (head_aware_ppl - uniform_ppl) / uniform_ppl * 100
        
        uniform_acc = results['uniform_mask']['accuracy']
        head_aware_acc = results['head_aware_mask']['accuracy']
        acc_diff = (head_aware_acc - uniform_acc) / uniform_acc * 100
        
        print("\n" + "-"*70)
        print("HEAD-AWARE vs UNIFORM COMPARISON:")
        print(f"  PPL change:      {ppl_diff:+.2f}% ({'better' if ppl_diff < 0 else 'worse'})")
        print(f"  Accuracy change: {acc_diff:+.2f}% ({'better' if acc_diff > 0 else 'worse'})")
        
        uniform_eff = results['uniform_mask'].get('effective_context', args.sink_size + args.window_size)
        head_aware_eff = results['head_aware_mask'].get('effective_context', 0)
        if head_aware_eff > 0:
            context_ratio = head_aware_eff / uniform_eff
            print(f"  Effective context ratio: {context_ratio:.2f}x")
    
    # Save results
    if args.output:
        output_data = {
            'config': {
                'model': args.model,
                'max_tokens': args.max_tokens,
                'sink_size': args.sink_size,
                'window_size': args.window_size,
                'classifications': args.classifications,
            },
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()


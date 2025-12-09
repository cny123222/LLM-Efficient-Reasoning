#!/usr/bin/env python3
"""
Ablation Study for Head-Aware KV Cache Compression

This script systematically evaluates different compression strategies to understand:
1. The importance of preserving sink tokens
2. Optimal window sizes for different head types
3. Impact of per-head vs uniform compression
4. PPL degradation vs memory savings tradeoffs

The ablation study compares:
- Baseline (no compression)
- StreamingLLM (uniform sink + window)
- Window-only (no sinks)
- Head-aware (per-head strategies based on classification)
- Head-aware variants (ablating specific components)

Usage:
    # Full ablation study
    python scripts/run_ablation_study.py --model_id EleutherAI/pythia-2.8b

    # Quick test with fewer samples
    python scripts/run_ablation_study.py --model_id EleutherAI/pythia-2.8b --num_samples 1 --max_tokens 1000
    
    # Specific ablations only
    python scripts/run_ablation_study.py --ablation sink_importance
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from kvcompress.methods import (
    streaming_llm_compress, 
    recent_only_compress,
    head_aware_compress,
    HeadAwareCompressor,
)
from kvcompress.evaluate import evaluate_with_compression


# Dataset paths
LOCAL_PG19_PATH = os.path.join(project_root, "data", "pg19.parquet")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    print("Using CPU device")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, use_bf16: bool = True):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_id}")
    
    device = get_device()
    
    torch_dtype = torch.bfloat16 if device.type == "cuda" and use_bf16 else torch.float32
    print(f"Using {'bfloat16' if torch_dtype == torch.bfloat16 else 'float32'} precision")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        print("Model loaded from cache")
    except (OSError, ValueError):
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    
    model.to(device)
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads per layer")
    
    return model, tokenizer, device


def load_evaluation_text(num_chars: int = 50000) -> str:
    """Load text for evaluation."""
    if os.path.exists(LOCAL_PG19_PATH):
        try:
            dataset = load_dataset("parquet", data_files={'test': LOCAL_PG19_PATH}, split="test")
            print(f"Loaded from local file")
        except Exception:
            dataset = None
    else:
        dataset = None
    
    if dataset is None:
        try:
            dataset = load_dataset("pg19", split="test")
        except Exception:
            return "The quick brown fox jumps over the lazy dog. " * 1000
    
    text = ""
    for sample in dataset:
        text += sample.get("text", "") + " "
        if len(text) >= num_chars:
            break
    
    return text[:num_chars]


def evaluate_method(
    model,
    tokenizer,
    text: str,
    compress_fn,
    compress_kwargs: dict,
    max_tokens: int,
    device: torch.device,
    method_name: str,
) -> dict:
    """Evaluate a single compression method."""
    print(f"  Evaluating: {method_name}...")
    
    start_time = time.time()
    
    result = evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        compress_fn=compress_fn,
        compress_kwargs=compress_kwargs,
        max_tokens=max_tokens,
        device=device,
    )
    
    elapsed = time.time() - start_time
    
    return {
        "method": method_name,
        "perplexity": result["perplexity"],
        "accuracy": result.get("accuracy", 0),
        "eval_tokens": result.get("eval_tokens", result.get("num_tokens", 0)),
        "elapsed_time": elapsed,
        **compress_kwargs,
    }


def run_sink_importance_ablation(
    model, tokenizer, text: str, max_tokens: int, device: torch.device
) -> List[dict]:
    """
    Ablation: How important are sink tokens?
    
    Compares:
    - StreamingLLM (sink + window)
    - Window only (no sinks)
    """
    results = []
    
    cache_sizes = [64, 128, 256, 512]
    sink_size = 4
    
    print("\n=== Ablation: Sink Token Importance ===")
    
    for total_size in cache_sizes:
        window_size = total_size - sink_size
        
        # With sinks (StreamingLLM)
        result_with_sink = evaluate_method(
            model, tokenizer, text,
            compress_fn=streaming_llm_compress,
            compress_kwargs={"start_size": sink_size, "recent_size": window_size},
            max_tokens=max_tokens,
            device=device,
            method_name=f"sink+window_{total_size}",
        )
        result_with_sink["has_sinks"] = True
        result_with_sink["total_cache_size"] = total_size
        results.append(result_with_sink)
        
        # Without sinks (window only)
        result_no_sink = evaluate_method(
            model, tokenizer, text,
            compress_fn=recent_only_compress,
            compress_kwargs={"window_size": total_size},
            max_tokens=max_tokens,
            device=device,
            method_name=f"window_only_{total_size}",
        )
        result_no_sink["has_sinks"] = False
        result_no_sink["total_cache_size"] = total_size
        results.append(result_no_sink)
    
    return results


def run_window_size_ablation(
    model, tokenizer, text: str, max_tokens: int, device: torch.device
) -> List[dict]:
    """
    Ablation: Optimal window size for different total cache budgets.
    """
    results = []
    
    sink_size = 4
    window_sizes = [4, 8, 16, 32, 64, 128, 256]
    
    print("\n=== Ablation: Window Size Optimization ===")
    
    for window in window_sizes:
        total_size = sink_size + window
        
        result = evaluate_method(
            model, tokenizer, text,
            compress_fn=streaming_llm_compress,
            compress_kwargs={"start_size": sink_size, "recent_size": window},
            max_tokens=max_tokens,
            device=device,
            method_name=f"sink{sink_size}_window{window}",
        )
        result["sink_size"] = sink_size
        result["window_size"] = window
        result["total_cache_size"] = total_size
        results.append(result)
    
    return results


def run_head_aware_ablation(
    model, tokenizer, text: str, max_tokens: int, device: torch.device,
    classifications_path: str,
) -> List[dict]:
    """
    Ablation: Head-aware vs uniform compression.
    
    Compares:
    - Full head-aware (per-head strategies)
    - Uniform StreamingLLM with equivalent average cache size
    - Variants with different strategy mappings
    """
    results = []
    
    print("\n=== Ablation: Head-Aware vs Uniform Compression ===")
    
    if not os.path.exists(classifications_path):
        print(f"  Warning: Classifications file not found: {classifications_path}")
        return results
    
    # Load classifications to analyze head distribution
    with open(classifications_path, 'r') as f:
        class_data = json.load(f)
    
    classifications = class_data.get('classifications', class_data)
    if isinstance(classifications, dict):
        classifications = classifications.get('classifications', [])
    
    # Count head types
    type_counts = {}
    for c in classifications:
        t = c.get('head_type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"  Head type distribution: {type_counts}")
    
    # 1. Full head-aware compression
    result_head_aware = evaluate_method(
        model, tokenizer, text,
        compress_fn=head_aware_compress,
        compress_kwargs={"classifications_path": classifications_path},
        max_tokens=max_tokens,
        device=device,
        method_name="head_aware_full",
    )
    result_head_aware["strategy"] = "head_aware"
    results.append(result_head_aware)
    
    # 2. Uniform StreamingLLM baselines
    for total_size in [128, 256, 512]:
        result_uniform = evaluate_method(
            model, tokenizer, text,
            compress_fn=streaming_llm_compress,
            compress_kwargs={"start_size": 4, "recent_size": total_size - 4},
            max_tokens=max_tokens,
            device=device,
            method_name=f"uniform_streaming_{total_size}",
        )
        result_uniform["strategy"] = "uniform"
        result_uniform["total_cache_size"] = total_size
        results.append(result_uniform)
    
    # 3. Ablate: All heads use sink+window (ignore classification)
    compressor_all_sink = HeadAwareCompressor(
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        default_strategy="sink_window",
        default_sink_size=4,
        default_window_size=8,
    )
    
    result_all_sink = evaluate_method(
        model, tokenizer, text,
        compress_fn=compressor_all_sink.compress,
        compress_kwargs={},
        max_tokens=max_tokens,
        device=device,
        method_name="all_sink_window",
    )
    result_all_sink["strategy"] = "all_sink_window"
    results.append(result_all_sink)
    
    # 4. Ablate: No sinks for any head
    compressor_no_sink = HeadAwareCompressor(
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        default_strategy="window_only",
        default_sink_size=0,
        default_window_size=12,  # Same total as sink(4)+window(8)
    )
    
    result_no_sink = evaluate_method(
        model, tokenizer, text,
        compress_fn=compressor_no_sink.compress,
        compress_kwargs={},
        max_tokens=max_tokens,
        device=device,
        method_name="all_window_only",
    )
    result_no_sink["strategy"] = "all_window_only"
    results.append(result_no_sink)
    
    return results


def run_baseline_evaluation(
    model, tokenizer, text: str, max_tokens: int, device: torch.device
) -> dict:
    """Evaluate baseline (no compression)."""
    print("\n=== Baseline Evaluation ===")
    
    result = evaluate_method(
        model, tokenizer, text,
        compress_fn=None,
        compress_kwargs={},
        max_tokens=max_tokens,
        device=device,
        method_name="baseline",
    )
    result["strategy"] = "none"
    
    return result


def print_ablation_results(results: List[dict], baseline_ppl: float):
    """Print ablation study results."""
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    
    print(f"\n{'Method':<30} {'PPL':>10} {'PPL Δ%':>10} {'Accuracy':>10} {'Time(s)':>10}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x['perplexity']):
        ppl = r['perplexity']
        ppl_delta = (ppl / baseline_ppl - 1) * 100 if baseline_ppl > 0 else 0
        acc = r.get('accuracy', 0)
        elapsed = r.get('elapsed_time', 0)
        
        print(f"{r['method']:<30} {ppl:>10.2f} {ppl_delta:>+10.1f}% {acc:>10.2%} {elapsed:>10.1f}")


def save_results(results: List[dict], output_path: str):
    """Save results to JSON file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study for Head-Aware KV Cache Compression"
    )
    
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--classifications_path", type=str,
                       default=os.path.join(project_root, "results", 
                                           "attention_analysis_pythia-2.8b", 
                                           "head_classifications.json"))
    parser.add_argument("--output_dir", type=str,
                       default=os.path.join(project_root, "results", "ablation_study"))
    parser.add_argument("--ablation", type=str, choices=["all", "sink_importance", 
                                                         "window_size", "head_aware"],
                       default="all", help="Which ablation to run")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("HEAD-AWARE COMPRESSION ABLATION STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Ablation: {args.ablation}")
    print(f"  Classifications: {args.classifications_path}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_id, not args.no_bf16)
    
    # Load evaluation text
    print("\nLoading evaluation text...")
    text = load_evaluation_text(num_chars=args.max_tokens * 10)
    print(f"  Loaded {len(text)} characters")
    
    all_results = []
    
    # Run baseline first
    baseline_result = run_baseline_evaluation(model, tokenizer, text, args.max_tokens, device)
    all_results.append(baseline_result)
    baseline_ppl = baseline_result['perplexity']
    
    print(f"\nBaseline PPL: {baseline_ppl:.2f}")
    
    # Run ablations
    if args.ablation in ["all", "sink_importance"]:
        results = run_sink_importance_ablation(model, tokenizer, text, args.max_tokens, device)
        all_results.extend(results)
    
    if args.ablation in ["all", "window_size"]:
        results = run_window_size_ablation(model, tokenizer, text, args.max_tokens, device)
        all_results.extend(results)
    
    if args.ablation in ["all", "head_aware"]:
        results = run_head_aware_ablation(
            model, tokenizer, text, args.max_tokens, device, args.classifications_path
        )
        all_results.extend(results)
    
    # Print results
    print_ablation_results(all_results, baseline_ppl)
    
    # Save results
    output_path = os.path.join(args.output_dir, f"ablation_{args.ablation}_{args.model_id.split('/')[-1]}.json")
    save_results(all_results, output_path)
    
    # Print key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    # Analyze sink importance
    sink_results = [r for r in all_results if 'has_sinks' in r]
    if sink_results:
        with_sinks = [r for r in sink_results if r['has_sinks']]
        without_sinks = [r for r in sink_results if not r['has_sinks']]
        
        if with_sinks and without_sinks:
            avg_ppl_with = np.mean([r['perplexity'] for r in with_sinks])
            avg_ppl_without = np.mean([r['perplexity'] for r in without_sinks])
            
            print(f"\nSink Token Importance:")
            print(f"  With sinks: avg PPL = {avg_ppl_with:.2f}")
            print(f"  Without sinks: avg PPL = {avg_ppl_without:.2f}")
            print(f"  PPL degradation without sinks: +{(avg_ppl_without/avg_ppl_with-1)*100:.1f}%")
    
    # Analyze head-aware vs uniform
    head_aware_results = [r for r in all_results if r.get('strategy') == 'head_aware']
    uniform_results = [r for r in all_results if r.get('strategy') == 'uniform']
    
    if head_aware_results and uniform_results:
        ha_ppl = head_aware_results[0]['perplexity']
        uniform_ppls = [r['perplexity'] for r in uniform_results]
        
        print(f"\nHead-Aware vs Uniform:")
        print(f"  Head-aware PPL: {ha_ppl:.2f}")
        print(f"  Best uniform PPL: {min(uniform_ppls):.2f}")
        print(f"  Head-aware advantage: {(min(uniform_ppls)/ha_ppl-1)*100:.1f}% better PPL")
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


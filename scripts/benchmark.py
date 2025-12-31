#!/usr/bin/env python3
"""
Unified Benchmark Script for KV Cache Compression Methods

This script benchmarks multiple KV cache compression methods on the PG-19 dataset.

Usage Examples:
    # Test L2 compression (KnormPress)
    python scripts/benchmark.py --method l2_compress --keep_ratios 1.0,0.8,0.5

    # Test fixed-size L2 compression
    python scripts/benchmark.py --method fix_size_l2 --fix_kv_sizes 256,512 --strategies keep_low

    # Test StreamingLLM
    python scripts/benchmark.py --method streaming_llm --start_size 4 --recent_sizes 252,508,1020

    # Compare all methods
    python scripts/benchmark.py --compare_all

    # Load model from local path (instead of HuggingFace)
    python scripts/benchmark.py --method streaming_llm --model_id /mnt/disk1/models/pythia-2.8b

    # Load model from HuggingFace (default)
    python scripts/benchmark.py --method streaming_llm --model_id EleutherAI/pythia-2.8b

Model Loading:
    The --model_id argument accepts either:
    - HuggingFace model ID (e.g., "EleutherAI/pythia-2.8b")
    - Local directory path (e.g., "/mnt/disk1/models/pythia-2.8b")
    
    The script will first try to load from local cache, then download if needed.
    For local paths, ensure the directory contains model files (config.json, pytorch_model.bin, etc.)

Supported Methods:
    - l2_compress: KnormPress ratio-based compression
    - fix_size_l2: Fixed-size KV cache with L2-based eviction
    - streaming_llm: StreamingLLM with attention sinks
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from kvcompress.methods import (
    get_compress_fn, list_methods, 
    l2_compress, fix_size_l2_compress, streaming_llm_compress, recent_only_compress,
    head_aware_compress, HeadAwareCompressor,
)
from kvcompress.benchmark import benchmark, run_benchmark_suite, print_benchmark_summary
from kvcompress.evaluate import evaluate_with_compression


# Dataset paths
LOCAL_PG19_PATH = os.path.join(project_root, "data", "pg19.parquet")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str = "EleutherAI/pythia-2.8b"):
    """
    Load model and tokenizer using Auto classes.
    
    AutoModelForCausalLM automatically detects the model architecture from config.json
    and loads the appropriate model class (e.g., GPTNeoXForCausalLM for Pythia,
    LlamaForCausalLM for Llama, etc.)
    
    Args:
        model_id: HuggingFace model ID or local path
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    print(f"Loading model: {model_id}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Use AutoModelForCausalLM for automatic model class detection
    # This provides better flexibility for different model architectures
    import os
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"  # Enable transformers logging
    
    print("Loading model (this may take a while for large models)...")
    
    # Try offline first to avoid network timeouts for cached models
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            local_files_only=True,  # Try cache first
        )
        print("Model loaded from cache")
    except (OSError, ValueError) as e:
        # Model not in cache, download it
        print(f"Model not in cache, downloading from HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            local_files_only=False,  # Allow download
        )
        print("Model downloaded and loaded")
    
    print("Moving model to device...")
    model.to(device)
    model.eval()
    print("Model ready!")
    
    # Same logic for tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=False
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def warmup_model(model, tokenizer, device, num_warmup: int = 3):
    """
    Perform warmup runs to eliminate cold-start effects.
    
    This helps avoid abnormally high TTFT in the first benchmark run by:
    1. Warming up GPU/MPS
    2. Triggering PyTorch JIT compilation
    3. Pre-allocating memory
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Device to use
        num_warmup: Number of warmup iterations
    """
    print(f"\nPerforming {num_warmup} warmup iterations...")
    
    # Simple warmup text
    warmup_text = "The quick brown fox jumps over the lazy dog. " * 10
    input_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        for i in range(num_warmup):
            # Forward pass (prefill)
            outputs = model(input_ids, use_cache=True)
            
            # Generate a few tokens
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            
            for _ in range(10):  # Generate 10 tokens
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            
            print(f"  Warmup {i+1}/{num_warmup} completed")
    
    # Clear any cached memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("  Warmup finished!\n")


def load_pg19_samples(num_samples: int = 3):
    """Load samples from PG-19 dataset."""
    print("\nLoading PG-19 dataset...")
    
    # Try local file first
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
            dataset = None
    else:
        dataset = None
    
    # Fallback to HuggingFace
    if dataset is None:
        try:
            print("  Loading from HuggingFace...")
            dataset = load_dataset("pg19", split="test")
        except Exception as e:
            print(f"  Failed to load from HuggingFace: {e}")
            return []
    
    # Extract text samples
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


def build_methods_config(args) -> list:
    """Build methods configuration based on command line arguments."""
    methods = []
    
    # Always add baseline if not disabled
    if not args.no_baseline:
        methods.append({
            "name": "baseline",
            "compress_fn": None,
            "kwargs": {}
        })
    
    if args.method == "l2_compress":
        # L2 compression with different keep ratios (no recent_only control for dynamic size)
        keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
        for kr in keep_ratios:
            if kr >= 1.0 and not args.no_baseline:
                continue  # Skip duplicate baseline
            methods.append({
                "name": f"l2_kr={kr:.1f}",
                "compress_fn": l2_compress,
                "kwargs": {
                    "keep_ratio": kr,
                    "prune_after": args.prune_after,
                }
            })
    
    elif args.method == "fix_size_l2":
        # Fixed-size L2 compression
        fix_kv_sizes = [int(x) for x in args.fix_kv_sizes.split(",")]
        strategies = [x.strip() for x in args.strategies.split(",")]
        keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
        
        # Add recent_only control group for each fix_kv_size if not disabled
        if not args.no_recent_only:
            for fix_size in fix_kv_sizes:
                methods.append({
                    "name": f"recent_only_{fix_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": fix_size,
                    }
                })
        
        for fix_size in fix_kv_sizes:
            for strategy in strategies:
                for kr in keep_ratios:
                    methods.append({
                        "name": f"fix{fix_size}_{strategy}_kr={kr:.1f}",
                        "compress_fn": fix_size_l2_compress,
                        "kwargs": {
                            "fix_kv_size": fix_size,
                            "strategy": strategy,
                            "keep_ratio": kr,
                        }
                    })
    
    elif args.method == "streaming_llm":
        # StreamingLLM with different cache sizes
        recent_sizes = [int(x) for x in args.recent_sizes.split(",")]
        
        # Add recent_only control group for each total cache size if not disabled
        if not args.no_recent_only:
            for recent_size in recent_sizes:
                total_size = args.start_size + recent_size
                methods.append({
                    "name": f"recent_only_{total_size}",
                    "compress_fn": recent_only_compress,
                    "kwargs": {
                        "window_size": total_size,
                    }
                })
        
        for recent_size in recent_sizes:
            total_size = args.start_size + recent_size
            methods.append({
                "name": f"streaming_{total_size}",
                "compress_fn": streaming_llm_compress,
                "kwargs": {
                    "start_size": args.start_size,
                    "recent_size": recent_size,
                }
            })
    
    elif args.method == "head_aware":
        # Head-aware compression with classification-based strategies
        classifications_path = args.classifications_path
        
        if classifications_path and os.path.exists(classifications_path):
            print(f"Using head classifications from: {classifications_path}")
            methods.append({
                "name": "head_aware_classified",
                "compress_fn": head_aware_compress,
                "kwargs": {
                    "classifications_path": classifications_path,
                }
            })
        else:
            print("No classification file found, using uniform StreamingLLM-style fallback")
        
        # Add StreamingLLM comparison with same total cache size
        # Estimate total cache: assume ~30% heads need full cache (gathering), 
        # ~70% heads use sink(4) + window(8) = 12 tokens per head
        # This is approximate; actual savings depend on classification
        
        # Add uniform StreamingLLM baselines for comparison
        for total_size in [512, 256, 128]:
            recent_size = total_size - args.start_size
            methods.append({
                "name": f"streaming_{total_size}",
                "compress_fn": streaming_llm_compress,
                "kwargs": {"start_size": args.start_size, "recent_size": recent_size}
            })
    
    elif args.compare_all:
        # Compare all methods with default configurations
        methods.extend([
            {
                "name": "l2_kr=0.8",
                "compress_fn": l2_compress,
                "kwargs": {"keep_ratio": 0.8, "prune_after": args.prune_after}
            },
            {
                "name": "l2_kr=0.5",
                "compress_fn": l2_compress,
                "kwargs": {"keep_ratio": 0.5, "prune_after": args.prune_after}
            },
            {
                "name": "fix512_keep_low",
                "compress_fn": fix_size_l2_compress,
                "kwargs": {"fix_kv_size": 512, "strategy": "keep_low", "keep_ratio": 0.5}
            },
            {
                "name": "streaming_512",
                "compress_fn": streaming_llm_compress,
                "kwargs": {"start_size": 4, "recent_size": 508}
            },
            {
                "name": "streaming_1024",
                "compress_fn": streaming_llm_compress,
                "kwargs": {"start_size": 4, "recent_size": 1020}
            },
        ])
        
        # Add head-aware if classification exists
        default_class_path = os.path.join(
            project_root, "results", "attention_analysis_pythia-2.8b", "head_classifications.json"
        )
        if os.path.exists(default_class_path):
            methods.append({
                "name": "head_aware",
                "compress_fn": head_aware_compress,
                "kwargs": {"classifications_path": default_class_path}
            })
    
    return methods


def save_results_to_json(results_data: dict, output_dir: str):
    """Save benchmark results to JSON file."""
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, indent=2, fp=f)
    print(f"\n✅ Results saved to: {json_path}")
    return json_path


def plot_benchmark_results(grouped_results: dict, output_dir: str, baseline_stats: dict = None):
    """Generate comparison plots for benchmark results."""
    
    methods = list(grouped_results.keys())
    if not methods:
        print("No results to plot.")
        return
    
    # Prepare data
    data = {
        'throughput': [],
        'tpot': [],
        'ttft': [],
        'ppl': [],
        'vram': [],
        'cache': []
    }
    
    for method in methods:
        results = grouped_results[method]
        data['throughput'].append(np.mean([r['throughput'] for r in results]))
        data['tpot'].append(np.mean([r['tpot'] for r in results]) * 1000)  # Convert to ms
        data['ttft'].append(np.mean([r['ttft'] for r in results]) * 1000)  # Convert to ms
        data['ppl'].append(np.mean([r['perplexity'] for r in results]))
        data['vram'].append(np.mean([r.get('peak_vram_gb', 0) for r in results]))
        data['cache'].append(np.mean([r['final_cache_size'] for r in results]))
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('KV Cache Compression Benchmark Results', fontsize=16, fontweight='bold')
    
    # Color mapping
    colors = []
    for method in methods:
        if method == 'baseline':
            colors.append('#808080')  # Gray
        elif 'streaming' in method:
            colors.append('#2ecc71')  # Green
        elif 'recent_only' in method:
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#3498db')  # Blue
    
    # Plot 1: Throughput (Higher is better)
    ax = axes[0, 0]
    bars = ax.bar(range(len(methods)), data['throughput'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Throughput\n(Higher is Better)', fontweight='bold')
    ax.set_ylabel('Tokens / Second')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: TPOT (Lower is better)
    ax = axes[0, 1]
    bars = ax.bar(range(len(methods)), data['tpot'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('TPOT / Latency\n(Lower is Better)', fontweight='bold')
    ax.set_ylabel('Milliseconds')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: TTFT (Lower is better)
    ax = axes[0, 2]
    bars = ax.bar(range(len(methods)), data['ttft'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('TTFT / Prefill\n(Lower is Better)', fontweight='bold')
    ax.set_ylabel('Milliseconds')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Perplexity (Lower is better)
    ax = axes[1, 0]
    bars = ax.bar(range(len(methods)), data['ppl'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Perplexity / Quality\n(Lower is Better)', fontweight='bold')
    ax.set_ylabel('PPL Score')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    # Dynamic Y-axis for better visibility
    if len(data['ppl']) > 0:
        min_ppl, max_ppl = min(data['ppl']), max(data['ppl'])
        ax.set_ylim(min_ppl * 0.95, max_ppl * 1.05)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Peak VRAM Usage (Lower is better)
    ax = axes[1, 1]
    bars = ax.bar(range(len(methods)), data['vram'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Peak VRAM Usage\n(Lower is Better)', fontweight='bold')
    ax.set_ylabel('Memory (GB)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Cache Size
    ax = axes[1, 2]
    bars = ax.bar(range(len(methods)), data['cache'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('KV Cache Size', fontweight='bold')
    ax.set_ylabel('Tokens')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {plot_path}")
    plt.close()
    
    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark for KV Cache Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # L2 compression (KnormPress)
  python scripts/benchmark.py --method l2_compress --keep_ratios 1.0,0.8,0.5

  # Fixed-size L2
  python scripts/benchmark.py --method fix_size_l2 --fix_kv_sizes 256,512 --strategies keep_low

  # StreamingLLM
  python scripts/benchmark.py --method streaming_llm --start_size 4 --recent_sizes 252,508

  # Compare all methods
  python scripts/benchmark.py --compare_all

  # Load model from local path
  python scripts/benchmark.py --method streaming_llm --model_id /mnt/disk1/models/pythia-2.8b

  # Load model from HuggingFace (default)
  python scripts/benchmark.py --method streaming_llm --model_id EleutherAI/pythia-2.8b

Model Loading:
  The --model_id argument accepts either:
  - HuggingFace model ID: "EleutherAI/pythia-2.8b"
  - Local directory path: "/mnt/disk1/models/pythia-2.8b"
  
  For local paths, ensure the directory contains:
  - config.json
  - pytorch_model.bin (or model.safetensors)
  - tokenizer files (tokenizer.json, vocab.json, etc.)
        """
    )
    
    # General arguments
    # EleutherAI/pythia-6.9b
    # EleutherAI/pythia-70m-deduped
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-2.8b",
                       help="Model ID from HuggingFace or local directory path (e.g., '/mnt/disk1/models/pythia-2.8b')")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of PG-19 samples to test")
    parser.add_argument("--max_tokens", type=int, default=2000,
                       help="Maximum tokens for PPL evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                       help="Number of tokens to generate for TTFT/TPOT")
    parser.add_argument("--skip_layers", type=str, default="0,1",
                       help="Comma-separated layer indices to skip")
    parser.add_argument("--no_baseline", action="store_true",
                       help="Skip baseline (no compression) benchmark")
    parser.add_argument("--no_recent_only", action="store_true",
                       help="Skip recent_only (sliding window) control group for fixed-size methods")
    parser.add_argument("--num_warmup", type=int, default=3,
                       help="Number of warmup iterations before benchmark (default: 3)")
    
    # Method selection
    parser.add_argument("--method", type=str, choices=["l2_compress", "fix_size_l2", "streaming_llm", "head_aware"],
                       help="Compression method to benchmark")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all methods with default configurations")
    
    # Head-aware compression arguments
    parser.add_argument("--classifications_path", type=str, 
                       default=os.path.join(project_root, "results", "attention_analysis_pythia-2.8b", "head_classifications.json"),
                       help="Path to head classifications JSON file (for head_aware method)")
    
    # L2 compress arguments
    parser.add_argument("--keep_ratios", type=str, default="0.8,0.5,0.3",
                       help="Comma-separated keep_ratio values (for l2_compress)")
    parser.add_argument("--prune_after", type=int, default=100,
                       help="Only compress after this many tokens (for l2_compress)")
    
    # Fix-size L2 arguments
    parser.add_argument("--fix_kv_sizes", type=str, default="256,512",
                       help="Comma-separated fix_kv_size values (for fix_size_l2)")
    parser.add_argument("--strategies", type=str, default="keep_low",
                       help="Comma-separated strategies: keep_low,keep_high,random")
    
    # StreamingLLM arguments
    parser.add_argument("--start_size", type=int, default=4,
                       help="Number of initial tokens (attention sinks) for StreamingLLM")
    parser.add_argument("--recent_sizes", type=str, default="252,508,1020",
                       help="Comma-separated recent_size values for StreamingLLM")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.method and not args.compare_all:
        parser.error("Must specify --method or --compare_all")
    
    # Parse skip_layers
    skip_layers = [int(x) for x in args.skip_layers.split(",")]
    
    print("="*70)
    print("KV Cache Compression Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Method: {args.method or 'compare_all'}")
    print(f"  Skip layers: {skip_layers}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Max eval tokens: {args.max_tokens}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Warmup iterations: {args.num_warmup}")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = args.method or 'compare_all'
    model_short_name = os.path.basename(args.model_id.rstrip('/'))
    results_dir = os.path.join(project_root, "results", f"{method_name}_{model_short_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"  Results directory: {results_dir}")
    
    # args.model_id = "EleutherAI/pythia-6.9b"
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_id)
    
    # Perform warmup to avoid cold-start effects
    if args.num_warmup > 0:
        warmup_model(model, tokenizer, device, num_warmup=args.num_warmup)
    
    # Load PG-19 samples
    samples = load_pg19_samples(args.num_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return
    
    # Build methods configuration
    methods_config = build_methods_config(args)
    
    print(f"\nMethods to test:")
    for m in methods_config:
        print(f"  - {m['name']}: {m['kwargs']}")
    
    # Run benchmark on each sample
    all_results = []
    
    for i, text in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{len(samples)} ({len(text)} characters)")
        print("="*70)
        
        results = run_benchmark_suite(
            model=model,
            tokenizer=tokenizer,
            text=text,
            methods_config=methods_config,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.max_tokens,
            skip_layers=skip_layers,
            device=device,
        )
        
        all_results.extend(results)
    
    # Aggregate results by method
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (averaged across samples)")
    print("="*80)
    
    # Group by method name
    grouped = {}
    for r in all_results:
        method = r.get('method', 'unknown')
        if method not in grouped:
            grouped[method] = []
        grouped[method].append(r)
    
    # Print header
    print(f"\n{'Method':<25} {'TTFT(s)':>10} {'TPOT(s)':>10} "
          f"{'Thruput':>10} {'PPL':>10} {'Acc':>10} {'VRAM(GB)':>10} {'Cache':>8}")
    print("-"*100)
    
    # Find baseline for comparison
    baseline_ppl = None
    baseline_acc = None
    baseline_throughput = None
    baseline_tpot = None
    
    if 'baseline' in grouped:
        baseline_results = grouped['baseline']
        baseline_ppl = np.mean([r['perplexity'] for r in baseline_results])
        baseline_acc = np.mean([r['accuracy'] for r in baseline_results])
        baseline_throughput = np.mean([r['throughput'] for r in baseline_results])
        baseline_tpot = np.mean([r['tpot'] for r in baseline_results])
    
    # Print results
    aggregated_stats = {}
    for method, results in grouped.items():
        avg_ttft = np.mean([r['ttft'] for r in results])
        avg_tpot = np.mean([r['tpot'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_ppl = np.mean([r['perplexity'] for r in results])
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_vram = np.mean([r.get('peak_vram_gb', 0) for r in results])
        avg_cache = np.mean([r['final_cache_size'] for r in results])
        
        aggregated_stats[method] = {
            'ttft': avg_ttft,
            'tpot': avg_tpot,
            'throughput': avg_throughput,
            'perplexity': avg_ppl,
            'accuracy': avg_acc,
            'peak_vram_gb': avg_vram,
            'cache_size': avg_cache
        }
        
        print(f"{method:<25} {avg_ttft:>10.4f} {avg_tpot:>10.4f} "
              f"{avg_throughput:>10.2f} {avg_ppl:>10.2f} {avg_acc:>10.2%} "
              f"{avg_vram:>10.2f} {avg_cache:>8.0f}")
    
    print("="*100)
    
    # Print comparison with baseline
    if baseline_ppl is not None and len(grouped) > 1:
        print("\nComparison with baseline (Throughput ↑ better, TPOT ↓ better, PPL ↓ better):")
        for method, results in grouped.items():
            if method == 'baseline':
                continue
            
            avg_throughput = np.mean([r['throughput'] for r in results])
            avg_tpot = np.mean([r['tpot'] for r in results])
            avg_ppl = np.mean([r['perplexity'] for r in results])
            avg_acc = np.mean([r['accuracy'] for r in results])
            
            # Throughput improvement (higher is better)
            throughput_imp = (avg_throughput / baseline_throughput - 1) * 100 if baseline_throughput > 0 else 0
            # TPOT improvement (lower is better)
            tpot_imp = (1 - avg_tpot / baseline_tpot) * 100 if baseline_tpot > 0 else 0
            ppl_change = (avg_ppl / baseline_ppl - 1) * 100 if baseline_ppl > 0 else 0
            acc_change = (avg_acc / baseline_acc - 1) * 100 if baseline_acc > 0 else 0
            
            print(f"  {method}: Throughput {throughput_imp:+.1f}%, TPOT {tpot_imp:+.1f}%, PPL {ppl_change:+.1f}%, Acc {acc_change:+.1f}%")
    
    # Save results to JSON
    results_data = {
        "config": {
            "model_id": args.model_id,
            "method": args.method or 'compare_all',
            "num_samples": args.num_samples,
            "max_tokens": args.max_tokens,
            "max_new_tokens": args.max_new_tokens,
            "skip_layers": skip_layers,
            "timestamp": timestamp
        },
        "raw_results": all_results,
        "aggregated_stats": aggregated_stats,
        "baseline_stats": {
            "perplexity": baseline_ppl,
            "accuracy": baseline_acc,
            "throughput": baseline_throughput,
            "tpot": baseline_tpot
        } if baseline_ppl is not None else None
    }
    
    save_results_to_json(results_data, results_dir)
    
    # Generate comparison plots
    plot_benchmark_results(grouped, results_dir, 
                          baseline_stats=results_data.get("baseline_stats"))
    
    print("\n" + "="*70)
    print(f"✅ Benchmark completed!")
    print(f"📁 Results saved to: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()


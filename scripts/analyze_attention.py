#!/usr/bin/env python3
"""
Attention Head Analysis Script

This script analyzes attention patterns in Pythia models to identify:
1. Positional Heads - Low entropy, focus on fixed relative positions
2. Gathering Heads - High entropy, content-dependent attention  
3. Dead Heads - Near-uniform distribution, can be pruned

Usage:
    # Analyze Pythia-70m with default settings
    python scripts/analyze_attention.py --model_id EleutherAI/pythia-70m-deduped
    
    # Analyze with custom sequence length
    python scripts/analyze_attention.py --model_id EleutherAI/pythia-70m-deduped --max_tokens 2048
    
    # Analyze Pythia-2.8b
    python scripts/analyze_attention.py --model_id EleutherAI/pythia-2.8b --max_tokens 1024
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from kvcompress.analysis import (
    AttentionAnalyzer,
    analyze_attention_heads,
    create_full_report,
)


# Dataset paths
LOCAL_PG19_PATH = os.path.join(project_root, "data", "pg19.parquet")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    elif torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    print("Using CPU device")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, use_bf16: bool = True):
    """
    Load model and tokenizer.
    
    Args:
        model_id: HuggingFace model ID
        use_bf16: Whether to use bfloat16 precision (recommended for CUDA)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    print(f"\nLoading model: {model_id}")
    
    device = get_device()
    
    # Determine dtype based on device and user preference
    if device.type == "cuda" and use_bf16:
        torch_dtype = torch.bfloat16
        print("Using bfloat16 precision")
    else:
        torch_dtype = torch.float32
        print("Using float32 precision")
    
    print("Loading model (this may take a while)...")
    
    # Try offline first
    # Use attn_implementation="eager" to support output_attentions=True
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="eager",  # Required for output_attentions
        )
        print("Model loaded from cache")
    except (OSError, ValueError):
        print("Model not in cache, downloading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=False,
            attn_implementation="eager",  # Required for output_attentions
        )
        print("Model downloaded and loaded")
    
    model.to(device)
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads per layer: {model.config.num_attention_heads}")
    print(f"  Total heads: {model.config.num_hidden_layers * model.config.num_attention_heads}")
    
    return model, tokenizer, device


def load_sample_text(num_chars: int = 50000) -> str:
    """Load sample text from PG-19 dataset."""
    print("\nLoading sample text from PG-19...")
    
    # Try local file first
    if os.path.exists(LOCAL_PG19_PATH):
        try:
            dataset = load_dataset(
                "parquet",
                data_files={'test': LOCAL_PG19_PATH},
                split="test"
            )
            print(f"Loaded from local file: {LOCAL_PG19_PATH}")
        except Exception as e:
            print(f"Failed to load local file: {e}")
            dataset = None
    else:
        dataset = None
    
    # Fallback to HuggingFace
    if dataset is None:
        try:
            print("Loading from HuggingFace...")
            dataset = load_dataset("pg19", split="test")
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            # Return default text
            return "The quick brown fox jumps over the lazy dog. " * 1000
    
    # Concatenate text from samples
    text = ""
    for sample in dataset:
        text += sample.get("text", "") + " "
        if len(text) >= num_chars:
            break
    
    print(f"Loaded {len(text)} characters of text")
    return text[:num_chars]


def warmup_model(model, tokenizer, device, num_warmup: int = 3):
    """Perform warmup runs to eliminate cold-start effects."""
    print(f"\nPerforming {num_warmup} warmup iterations...")
    
    warmup_text = "The quick brown fox jumps over the lazy dog. " * 20
    input_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    input_ids = input_ids[:, :256]  # Limit warmup size
    
    with torch.inference_mode():
        for i in range(num_warmup):
            _ = model(input_ids, output_attentions=True, use_cache=False)
            print(f"  Warmup {i+1}/{num_warmup} completed")
    
    # Clear cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("Warmup finished!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention head patterns in language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze Pythia-70m
    python scripts/analyze_attention.py --model_id EleutherAI/pythia-70m-deduped
    
    # Analyze with custom settings
    python scripts/analyze_attention.py --model_id EleutherAI/pythia-70m-deduped --max_tokens 2048 --chunk_size 512
    
    # Analyze Pythia-2.8b
    python scripts/analyze_attention.py --model_id EleutherAI/pythia-2.8b --max_tokens 1024
        """
    )
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="EleutherAI/pythia-70m-deduped",
        help="HuggingFace model ID (default: EleutherAI/pythia-70m-deduped)"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2048,
        help="Maximum tokens to analyze (default: 2048)"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=512,
        help="Chunk size for processing (default: 512)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory (default: results/attention_analysis_{model_name})"
    )
    parser.add_argument(
        "--no_bf16", 
        action="store_true",
        help="Disable bfloat16 precision (use float32)"
    )
    parser.add_argument(
        "--num_warmup", 
        type=int, 
        default=3,
        help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--sink_size", 
        type=int, 
        default=4,
        help="Number of initial tokens considered as sinks (default: 4)"
    )
    parser.add_argument(
        "--local_window", 
        type=int, 
        default=8,
        help="Window size for local attention (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        model_name = args.model_id.split("/")[-1]
        args.output_dir = os.path.join(project_root, "results", f"attention_analysis_{model_name}")
    
    print("=" * 70)
    print("ATTENTION HEAD ANALYSIS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  BF16: {not args.no_bf16}")
    print(f"  Sink size: {args.sink_size}")
    print(f"  Local window: {args.local_window}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_id, 
        use_bf16=not args.no_bf16
    )
    
    # Warmup
    if args.num_warmup > 0:
        warmup_model(model, tokenizer, device, args.num_warmup)
    
    # Load sample text
    text = load_sample_text(num_chars=args.max_tokens * 10)
    
    # Create analyzer
    analyzer = AttentionAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        sink_size=args.sink_size,
        local_window=args.local_window,
    )
    
    # Run analysis
    print("\n" + "=" * 70)
    print("RUNNING ATTENTION ANALYSIS")
    print("=" * 70)
    
    stats, classifications = analyzer.analyze(
        text=text,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        show_progress=True,
    )
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    analyzer.save_results(stats, classifications, args.output_dir)
    
    # Generate visualizations
    model_name = args.model_id.split("/")[-1]
    create_full_report(stats, classifications, args.output_dir, model_name=model_name)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    total_heads = len(classifications)
    type_counts = {}
    prunable = 0
    limitable = 0
    
    for c in classifications:
        type_name = c.head_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        if c.can_prune:
            prunable += 1
        if c.can_limit_window:
            limitable += 1
    
    print(f"\nHead Type Distribution:")
    for head_type, count in sorted(type_counts.items()):
        pct = count / total_heads * 100
        print(f"  {head_type.capitalize():12}: {count:3} heads ({pct:5.1f}%)")
    
    print(f"\nOptimization Opportunities:")
    print(f"  Prunable heads: {prunable} ({prunable/total_heads*100:.1f}%)")
    print(f"  Limitable heads: {limitable} ({limitable/total_heads*100:.1f}%)")
    
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - head_statistics.json")
    print("  - head_classifications.json")
    print("  - entropy_heatmap.png")
    print("  - position_preference.png")
    print("  - sink_ratio_analysis.png")
    print("  - head_clustering.png")
    print("  - relative_position_heatmap.png")
    print("  - analysis_summary.txt")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


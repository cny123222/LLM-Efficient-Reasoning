"""
Unified Benchmark Module for KV Cache Compression

This module provides functions to measure generation performance metrics:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Throughput (tokens/sec)
- PPL (Perplexity)
- Accuracy (Next token prediction accuracy)

All functions support any compression method through the compress_fn parameter.
"""

import time
from typing import Callable, Dict, List, Optional, Union
import torch
from transformers import DynamicCache

from .utils import to_dynamic_cache, normalize_kv_cache
from .evaluate import evaluate_with_compression


def measure_generation_metrics(
    model,
    tokenizer,
    text: str,
    compress_fn: Optional[Callable] = None,
    compress_kwargs: Optional[Dict] = None,
    max_new_tokens: int = 1000,
    max_input_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure generation performance metrics with KV cache compression.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text (prompt)
        compress_fn: Compression function (None for no compression)
        compress_kwargs: Additional kwargs for compress_fn
        max_new_tokens: Number of tokens to generate
        max_input_tokens: Maximum input tokens (to prevent OOM)
        skip_layers: Layer indices to skip compression
        device: Device to use
    
    Returns:
        Dict containing:
        - ttft: Time to first token (seconds)
        - tpot: Time per output token (seconds)
        - throughput: Tokens per second
        - total_time: Total generation time (seconds)
        - num_tokens: Number of tokens generated
        - input_length: Input sequence length
    """
    if device is None:
        device = next(model.parameters()).device
    
    if compress_kwargs is None:
        compress_kwargs = {}
    
    # Tokenize input (truncate to prevent OOM)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_input_tokens].to(device)
    input_length = input_ids.shape[1]
    
    # Pad token handling
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    generated_tokens = []
    past_key_values = None
    ttft = None
    
    # Reset memory stats for VRAM measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    total_start = time.perf_counter()
    
    with torch.inference_mode():
        # First forward pass (prefill) - measure TTFT
        first_start = time.perf_counter()
        outputs = model(
            input_ids,
            use_cache=True,
            return_dict=True,
        )
        
        # Get first token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        
        ttft = time.perf_counter() - first_start
        
        # Get and compress KV cache
        past_key_values = outputs.past_key_values
        if compress_fn is not None and past_key_values is not None:
            kv_list = list(normalize_kv_cache(past_key_values))
            compressed_kv = compress_fn(kv_list, skip_layers=skip_layers, **compress_kwargs)
            past_key_values = to_dynamic_cache(compressed_kv)
        
        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Get and compress KV cache
            past_key_values = outputs.past_key_values
            if compress_fn is not None and past_key_values is not None:
                kv_list = list(normalize_kv_cache(past_key_values))
                compressed_kv = compress_fn(kv_list, skip_layers=skip_layers, **compress_kwargs)
                past_key_values = to_dynamic_cache(compressed_kv)
    
    total_time = time.perf_counter() - total_start
    num_generated = len(generated_tokens)
    
    # Calculate metrics
    tpot = (total_time - ttft) / max(num_generated - 1, 1)
    throughput = num_generated / total_time if total_time > 0 else 0
    
    # Measure peak VRAM usage
    peak_memory_gb = 0.0
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    return {
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "num_tokens": num_generated,
        "input_length": input_length,
        "peak_vram_gb": peak_memory_gb,
    }


def benchmark(
    model,
    tokenizer,
    text: str,
    compress_fn: Optional[Callable] = None,
    compress_kwargs: Optional[Dict] = None,
    max_new_tokens: int = 1000,
    eval_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run full benchmark including timing metrics and quality metrics.
    
    NOTE: TTFT/TPOT are now measured across ALL eval_tokens (not just max_new_tokens).
    This provides more accurate timing for long sequence scenarios.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        compress_fn: Compression function (None for no compression)
        compress_kwargs: Additional kwargs for compress_fn
        max_new_tokens: (Deprecated, kept for compatibility) Number of tokens for old generation benchmark
        eval_tokens: Number of tokens for evaluation (TTFT/TPOT now measured here!)
        skip_layers: Layer indices to skip compression
        device: Device to use
    
    Returns:
        Dict containing all metrics:
        - ttft, tpot, throughput, total_time (measured across eval_tokens)
        - perplexity, accuracy, eval_tokens, final_cache_size
    """
    if compress_kwargs is None:
        compress_kwargs = {}
    
    # Measure ALL metrics in a single pass through eval_tokens
    # This includes both quality metrics (PPL, Accuracy) and timing metrics (TTFT, TPOT)
    metrics = evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        compress_fn=compress_fn,
        compress_kwargs=compress_kwargs,
        max_tokens=eval_tokens,
        skip_layers=skip_layers,
        device=device,
        show_progress=True,
    )
    
    # Return all metrics (timing now comes from the full evaluation)
    result = {
        # Timing metrics (measured across ALL eval_tokens)
        "ttft": metrics["ttft"],
        "tpot": metrics["tpot"],
        "throughput": metrics["throughput"],
        "total_time": metrics["total_time"],
        # Quality metrics
        "perplexity": metrics["perplexity"],
        "accuracy": metrics["accuracy"],
        "eval_tokens": metrics["num_tokens"],
        "final_cache_size": metrics["final_cache_size"],
        # VRAM metrics
        "peak_vram_gb": metrics.get("peak_vram_gb", 0.0),
    }
    
    return result


def run_benchmark_suite(
    model,
    tokenizer,
    text: str,
    methods_config: List[Dict],
    max_new_tokens: int = 1000,
    eval_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Run complete benchmark suite across multiple compression methods.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        methods_config: List of dicts, each containing:
                       - "name": Method name for display
                       - "compress_fn": Compression function (None for baseline)
                       - "kwargs": Dict of compression parameters
        max_new_tokens: Number of tokens to generate
        eval_tokens: Number of tokens for PPL evaluation
        skip_layers: Layer indices to skip compression
        device: Device to use
    
    Returns:
        List of result dicts, one per method configuration
    
    Example:
        >>> from kvcompress.methods import l2_compress, streaming_llm_compress
        >>> methods = [
        ...     {"name": "baseline", "compress_fn": None, "kwargs": {}},
        ...     {"name": "l2_0.8", "compress_fn": l2_compress, "kwargs": {"keep_ratio": 0.8}},
        ...     {"name": "streaming_512", "compress_fn": streaming_llm_compress, 
        ...      "kwargs": {"start_size": 4, "recent_size": 508}},
        ... ]
        >>> results = run_benchmark_suite(model, tokenizer, text, methods)
    """
    results = []
    
    for config in methods_config:
        name = config.get("name", "unknown")
        compress_fn = config.get("compress_fn", None)
        kwargs = config.get("kwargs", {})
        
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        result = benchmark(
            model=model,
            tokenizer=tokenizer,
            text=text,
            compress_fn=compress_fn,
            compress_kwargs=kwargs,
            max_new_tokens=max_new_tokens,
            eval_tokens=eval_tokens,
            skip_layers=skip_layers,
            device=device,
        )
        
        result['method'] = name
        result['config'] = kwargs
        results.append(result)
        
        # Print results
        print(f"\nTiming Metrics (across {result['eval_tokens']} tokens):")
        print(f"  TTFT:       {result['ttft']:.4f} seconds")
        print(f"  TPOT:       {result['tpot']:.4f} seconds")
        print(f"  Throughput: {result['throughput']:.2f} tokens/sec")
        print(f"  Total time: {result['total_time']:.2f} seconds")
        
        print(f"\nQuality Metrics:")
        print(f"  PPL:        {result['perplexity']:.2f}")
        print(f"  Accuracy:   {result['accuracy']:.2%}")
        print(f"  Cache size: {result['final_cache_size']} tokens")
    
    return results


def print_benchmark_summary(results: List[Dict[str, float]]) -> None:
    """
    Print a summary table of benchmark results.
    
    Args:
        results: List of result dicts from run_benchmark_suite
    """
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    
    # Header
    print(f"{'Method':<20} {'TTFT(s)':>10} {'TPOT(s)':>10} "
          f"{'Thruput':>10} {'PPL':>10} {'Acc':>10} {'Cache':>8}")
    print("-"*90)
    
    # Find baseline for comparison
    baseline = None
    for r in results:
        if r.get('method') == 'baseline' or r.get('compress_fn') is None:
            baseline = r
            break
    
    if baseline is None and results:
        baseline = results[0]
    
    for r in results:
        method_name = r.get('method', 'unknown')[:20]
        
        print(f"{method_name:<20} "
              f"{r['ttft']:>10.4f} {r['tpot']:>10.4f} "
              f"{r['throughput']:>10.2f} "
              f"{r['perplexity']:>10.2f} {r['accuracy']:>10.2%} "
              f"{r['final_cache_size']:>8}")
    
    print("="*90)
    
    # Print comparison with baseline
    if baseline and len(results) > 1:
        print("\nComparison with baseline (Throughput ↑ better, TPOT ↓ better, PPL ↓ better):")
        for r in results:
            if r.get('method') == baseline.get('method'):
                continue
            
            # Throughput improvement (higher is better, positive means improvement)
            throughput_imp = (r['throughput'] / baseline['throughput'] - 1) * 100 if baseline['throughput'] > 0 else 0
            # TPOT improvement (lower is better, so we invert: positive means faster)
            tpot_imp = (1 - r['tpot'] / baseline['tpot']) * 100 if baseline['tpot'] > 0 else 0
            # PPL change (lower is better, so negative is improvement)
            ppl_change = (r['perplexity'] / baseline['perplexity'] - 1) * 100 if baseline['perplexity'] > 0 else 0
            # Accuracy change (higher is better, positive is improvement)
            acc_change = (r['accuracy'] / baseline['accuracy'] - 1) * 100 if baseline['accuracy'] > 0 else 0
            
            method_name = r.get('method', 'unknown')
            print(f"  {method_name}: "
                  f"Throughput {throughput_imp:+.1f}%, "
                  f"TPOT {tpot_imp:+.1f}%, "
                  f"PPL {ppl_change:+.1f}%, "
                  f"Acc {acc_change:+.1f}%")


__all__ = [
    'measure_generation_metrics',
    'benchmark',
    'run_benchmark_suite',
    'print_benchmark_summary',
]


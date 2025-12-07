"""
KnormPress: L2 Norm-Based KV Cache Compression

This module implements the KnormPress algorithm for KV cache compression.
Reference: "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
           (EMNLP 2024)

Core idea: Tokens with low L2 norm in their key embeddings correlate strongly
with high attention scores. By keeping these low-norm tokens, we preserve
the most important information while reducing memory usage.

Usage:
    from knormpress import l2_compress, evaluate_with_compression, benchmark

    # Compress KV cache
    compressed_cache = l2_compress(past_key_values, keep_ratio=0.8)

    # Evaluate PPL and Accuracy with compression
    results = evaluate_with_compression(model, tokenizer, text, keep_ratio=0.8)

    # Run full benchmark
    metrics = benchmark(model, tokenizer, text, keep_ratio=0.8)
"""

from .compress import (
    l2_compress,
    fix_size_l2_compress,
    get_cache_size_mb,
    get_cache_info,
    to_dynamic_cache,
    get_compress_fn,
)
from .evaluate import (
    evaluate_with_compression,
    evaluate_baseline,
    compare_compression_levels,
    evaluate_fix_size_compression,
)
from .benchmark import (
    benchmark,
    measure_generation_metrics,
    run_benchmark_suite,
    print_benchmark_summary,
)

__all__ = [
    # Compression functions
    'l2_compress',
    'fix_size_l2_compress',
    'get_cache_size_mb',
    'get_cache_info',
    'to_dynamic_cache',
    'get_compress_fn',
    # Evaluation functions
    'evaluate_with_compression',
    'evaluate_baseline',
    'compare_compression_levels',
    'evaluate_fix_size_compression',
    # Benchmark functions
    'benchmark',
    'measure_generation_metrics',
    'run_benchmark_suite',
    'print_benchmark_summary',
]

__version__ = '1.0.0'


#!/usr/bin/env python3
"""
全面性能对比 Benchmark

多维度对比以下方法:
1. Baseline (纯自回归, 无 Spec Decode)
2. Linear Speculative Decoding
3. Linear + StreamingLLM
4. Tree V2 Speculative Decoding
5. Tree V2 + StreamingLLM

测试维度:
- 吞吐量 (tokens/s)
- 内存使用 (峰值 MB)
- 接受率 (%)
- 每轮 tokens
- 端到端延迟 (s)

Usage:
    python papers/benchmark_comprehensive_comparison.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import gc
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer

from spec_decode.core import (
    SpeculativeGenerator,
    StreamingSpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
    TreeStreamingSpeculativeGeneratorV2,
)


# ============================================================================
# 配置
# ============================================================================
TARGET_MODEL_PATH = "/mnt/disk1/models/pythia-2.8b"
DRAFT_MODEL_PATH = "/mnt/disk1/models/pythia-70m"
DEVICE = "cuda"

# 测试 Token 长度
TOKEN_LENGTHS = [500, 1000, 2000, 3000]

# Linear Spec Decode 参数
LINEAR_K_VALUES = [5, 6, 7, 8]

# Tree V2 参数搜索空间
TREE_DEPTHS = [6, 8]
TREE_BRANCHES = [2, 3]
TREE_THRESHOLDS = [0.03, 0.05]

# StreamingLLM 参数
STREAMING_CACHE_SIZES = [1024, 2048]
START_SIZE = 4

# 运行配置
NUM_RUNS = 3
SKIP_FIRST = True

PROMPT = """Write a comprehensive and detailed technical explanation about the development and evolution of large language models in artificial intelligence. Cover the complete history from early neural networks to modern transformers, discuss all major architecture innovations including attention mechanisms and scaling laws, explain the training techniques such as pre-training and fine-tuning, and provide insights into future directions and challenges.

Begin your detailed explanation:

The field of artificial intelligence has witnessed remarkable progress"""


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class BenchmarkResult:
    method: str
    tokens: int
    throughput: float  # tokens/s
    latency: float     # seconds
    memory_peak_mb: float
    acceptance_rate: float
    tokens_per_round: float
    compression_count: int
    config: Dict


# ============================================================================
# 工具函数
# ============================================================================
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_gpu_memory() -> Tuple[float, float]:
    """获取 GPU 内存 (allocated, peak) in MB"""
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.max_memory_allocated() / 1024**2
        )
    return 0.0, 0.0


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


# ============================================================================
# Benchmark 函数
# ============================================================================
def benchmark_baseline(
    target_model,
    tokenizer,
    max_new_tokens: int,
    num_runs: int = NUM_RUNS
) -> BenchmarkResult:
    """测试 Baseline (无 Spec Decode)"""
    
    results = []
    
    for i in range(num_runs):
        cleanup()
        torch.cuda.reset_peak_memory_stats()
        
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            output = target_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=tokenizer.pad_token_id
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        _, mem_peak = get_gpu_memory()
        tokens_generated = output.shape[1] - input_ids.shape[1]
        throughput = tokens_generated / elapsed
        
        if not SKIP_FIRST or i > 0:
            results.append({
                'tokens': tokens_generated,
                'time': elapsed,
                'throughput': throughput,
                'memory_peak': mem_peak
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        print(f"      Run {i+1}: {throughput:.1f} t/s, {elapsed:.2f}s {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_latency = sum(r['time'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak'] for r in results) / len(results)
    
    return BenchmarkResult(
        method="Baseline (AR)",
        tokens=max_new_tokens,
        throughput=avg_throughput,
        latency=avg_latency,
        memory_peak_mb=avg_memory,
        acceptance_rate=0.0,
        tokens_per_round=1.0,
        compression_count=0,
        config={}
    )


def benchmark_linear_spec_decode(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    K: int,
    num_runs: int = NUM_RUNS
) -> BenchmarkResult:
    """测试 Linear Speculative Decoding"""
    
    gen = SpeculativeGenerator(
        target_model, draft_model, tokenizer,
        K=K, max_len=8192, device=DEVICE, use_compile=False
    )
    
    results = []
    
    for i in range(num_runs):
        cleanup()
        gen.reset()
        torch.cuda.reset_peak_memory_stats()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = gen.generate(PROMPT, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        _, mem_peak = get_gpu_memory()
        stats = gen.get_stats()
        throughput = stats['total_tokens'] / elapsed
        
        if not SKIP_FIRST or i > 0:
            results.append({
                'tokens': stats['total_tokens'],
                'time': elapsed,
                'throughput': throughput,
                'memory_peak': mem_peak,
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'tokens_per_round': stats.get('tokens_per_round', 0)
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        print(f"      Run {i+1}: {throughput:.1f} t/s {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_latency = sum(r['time'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_tpr = sum(r['tokens_per_round'] for r in results) / len(results)
    
    return BenchmarkResult(
        method=f"Linear K={K}",
        tokens=max_new_tokens,
        throughput=avg_throughput,
        latency=avg_latency,
        memory_peak_mb=avg_memory,
        acceptance_rate=avg_accept,
        tokens_per_round=avg_tpr,
        compression_count=0,
        config={'K': K}
    )


def benchmark_linear_streaming(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    K: int,
    max_cache_len: int,
    num_runs: int = NUM_RUNS
) -> BenchmarkResult:
    """测试 Linear + StreamingLLM"""
    
    recent_size = max_cache_len - START_SIZE
    
    gen = StreamingSpeculativeGenerator(
        target_model, draft_model, tokenizer,
        K=K, max_len=8192, max_cache_len=max_cache_len,
        start_size=START_SIZE, recent_size=recent_size,
        device=DEVICE, use_compile=False
    )
    
    results = []
    
    for i in range(num_runs):
        cleanup()
        gen.reset()
        torch.cuda.reset_peak_memory_stats()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = gen.generate(PROMPT, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        _, mem_peak = get_gpu_memory()
        stats = gen.get_stats()
        throughput = stats['total_tokens'] / elapsed
        
        if not SKIP_FIRST or i > 0:
            results.append({
                'tokens': stats['total_tokens'],
                'time': elapsed,
                'throughput': throughput,
                'memory_peak': mem_peak,
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'tokens_per_round': stats.get('tokens_per_round', 0),
                'compression_count': stats.get('compress_count', 0)
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        compress = stats.get('compress_count', 0)
        print(f"      Run {i+1}: {throughput:.1f} t/s, compress={compress} {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_latency = sum(r['time'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_tpr = sum(r['tokens_per_round'] for r in results) / len(results)
    avg_compress = sum(r['compression_count'] for r in results) / len(results)
    
    return BenchmarkResult(
        method=f"Linear K={K} + Stream(c={max_cache_len})",
        tokens=max_new_tokens,
        throughput=avg_throughput,
        latency=avg_latency,
        memory_peak_mb=avg_memory,
        acceptance_rate=avg_accept,
        tokens_per_round=avg_tpr,
        compression_count=int(avg_compress),
        config={'K': K, 'max_cache_len': max_cache_len}
    )


def benchmark_tree_v2(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    tree_depth: int,
    branch_factor: int,
    probability_threshold: float,
    num_runs: int = NUM_RUNS
) -> BenchmarkResult:
    """测试 Tree V2 Speculative Decoding"""
    
    gen = TreeSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=tree_depth,
        branch_factor=branch_factor,
        probability_threshold=probability_threshold,
        max_tree_nodes=128,
        device=DEVICE, use_compile=False
    )
    
    results = []
    
    for i in range(num_runs):
        cleanup()
        gen.reset()
        torch.cuda.reset_peak_memory_stats()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = gen.generate(PROMPT, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        _, mem_peak = get_gpu_memory()
        stats = gen.get_stats()
        throughput = stats['total_tokens'] / elapsed
        
        if not SKIP_FIRST or i > 0:
            results.append({
                'tokens': stats['total_tokens'],
                'time': elapsed,
                'throughput': throughput,
                'memory_peak': mem_peak,
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'avg_path_length': stats.get('avg_path_length', 0)
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        print(f"      Run {i+1}: {throughput:.1f} t/s {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_latency = sum(r['time'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_path = sum(r['avg_path_length'] for r in results) / len(results)
    
    return BenchmarkResult(
        method=f"Tree D={tree_depth} B={branch_factor} t={probability_threshold}",
        tokens=max_new_tokens,
        throughput=avg_throughput,
        latency=avg_latency,
        memory_peak_mb=avg_memory,
        acceptance_rate=avg_accept,
        tokens_per_round=avg_path,
        compression_count=0,
        config={
            'tree_depth': tree_depth,
            'branch_factor': branch_factor,
            'probability_threshold': probability_threshold
        }
    )


def benchmark_tree_streaming_v2(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    tree_depth: int,
    branch_factor: int,
    probability_threshold: float,
    max_cache_len: int,
    num_runs: int = NUM_RUNS
) -> BenchmarkResult:
    """测试 Tree V2 + StreamingLLM"""
    
    recent_size = max_cache_len - START_SIZE
    
    gen = TreeStreamingSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=tree_depth,
        branch_factor=branch_factor,
        probability_threshold=probability_threshold,
        max_tree_nodes=128,
        max_cache_len=max_cache_len,
        start_size=START_SIZE,
        recent_size=recent_size,
        device=DEVICE, use_compile=False
    )
    
    results = []
    
    for i in range(num_runs):
        cleanup()
        gen.reset()
        torch.cuda.reset_peak_memory_stats()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = gen.generate(PROMPT, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        _, mem_peak = get_gpu_memory()
        stats = gen.get_stats()
        throughput = stats['total_tokens'] / elapsed
        
        if not SKIP_FIRST or i > 0:
            results.append({
                'tokens': stats['total_tokens'],
                'time': elapsed,
                'throughput': throughput,
                'memory_peak': mem_peak,
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'avg_path_length': stats.get('avg_path_length', 0),
                'compression_count': stats.get('compression_count', 0)
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        compress = stats.get('compression_count', 0)
        print(f"      Run {i+1}: {throughput:.1f} t/s, compress={compress} {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_latency = sum(r['time'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_path = sum(r['avg_path_length'] for r in results) / len(results)
    avg_compress = sum(r['compression_count'] for r in results) / len(results)
    
    return BenchmarkResult(
        method=f"Tree D={tree_depth}B={branch_factor}t={probability_threshold} + Stream(c={max_cache_len})",
        tokens=max_new_tokens,
        throughput=avg_throughput,
        latency=avg_latency,
        memory_peak_mb=avg_memory,
        acceptance_rate=avg_accept,
        tokens_per_round=avg_path,
        compression_count=int(avg_compress),
        config={
            'tree_depth': tree_depth,
            'branch_factor': branch_factor,
            'probability_threshold': probability_threshold,
            'max_cache_len': max_cache_len
        }
    )


# ============================================================================
# 主函数
# ============================================================================
def main():
    print_header("全面性能对比 Benchmark")
    
    print(f"\n配置:")
    print(f"  Token 长度: {TOKEN_LENGTHS}")
    print(f"  Linear K: {LINEAR_K_VALUES}")
    print(f"  Tree Depth: {TREE_DEPTHS}, Branch: {TREE_BRANCHES}, Threshold: {TREE_THRESHOLDS}")
    print(f"  StreamingLLM Cache: {STREAMING_CACHE_SIZES}")
    
    # 加载模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
    )
    
    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        cleanup()
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
        with torch.inference_mode():
            _ = target_model.generate(input_ids, max_new_tokens=50, do_sample=False,
                                      eos_token_id=None, assistant_model=draft_model)
        torch.cuda.synchronize()
    
    all_results = []
    
    # 对每个 token 长度进行测试
    for max_tokens in TOKEN_LENGTHS:
        print_header(f"测试 {max_tokens} Tokens")
        
        token_results = []
        
        # 1. Baseline
        print_subheader("1. Baseline (无 Spec Decode)")
        baseline_result = benchmark_baseline(target_model, tokenizer, max_tokens)
        token_results.append(baseline_result)
        baseline_tp = baseline_result.throughput
        print(f"    >>> {baseline_result.throughput:.1f} t/s (baseline)")
        
        # 2. Linear Spec Decode (最佳 K)
        print_subheader("2. Linear Speculative Decoding")
        best_linear = None
        for K in LINEAR_K_VALUES:
            print(f"    K={K}:")
            result = benchmark_linear_spec_decode(
                target_model, draft_model, tokenizer, max_tokens, K
            )
            token_results.append(result)
            speedup = result.throughput / baseline_tp
            print(f"    >>> {result.throughput:.1f} t/s ({speedup:.2f}x)")
            
            if best_linear is None or result.throughput > best_linear.throughput:
                best_linear = result
        
        # 3. Linear + StreamingLLM
        print_subheader("3. Linear + StreamingLLM")
        best_K = best_linear.config['K']
        for cache_len in STREAMING_CACHE_SIZES:
            print(f"    K={best_K}, cache={cache_len}:")
            result = benchmark_linear_streaming(
                target_model, draft_model, tokenizer, max_tokens, best_K, cache_len
            )
            token_results.append(result)
            speedup = result.throughput / baseline_tp
            print(f"    >>> {result.throughput:.1f} t/s ({speedup:.2f}x)")
        
        # 4. Tree V2 Spec Decode
        print_subheader("4. Tree V2 Speculative Decoding")
        best_tree = None
        for depth in TREE_DEPTHS:
            for branch in TREE_BRANCHES:
                for threshold in TREE_THRESHOLDS:
                    print(f"    D={depth}, B={branch}, t={threshold}:")
                    result = benchmark_tree_v2(
                        target_model, draft_model, tokenizer, max_tokens,
                        depth, branch, threshold
                    )
                    token_results.append(result)
                    speedup = result.throughput / baseline_tp
                    print(f"    >>> {result.throughput:.1f} t/s ({speedup:.2f}x)")
                    
                    if best_tree is None or result.throughput > best_tree.throughput:
                        best_tree = result
        
        # 5. Tree V2 + StreamingLLM
        print_subheader("5. Tree V2 + StreamingLLM")
        best_config = best_tree.config
        for cache_len in STREAMING_CACHE_SIZES:
            print(f"    D={best_config['tree_depth']}, B={best_config['branch_factor']}, "
                  f"t={best_config['probability_threshold']}, cache={cache_len}:")
            result = benchmark_tree_streaming_v2(
                target_model, draft_model, tokenizer, max_tokens,
                best_config['tree_depth'], best_config['branch_factor'],
                best_config['probability_threshold'], cache_len
            )
            token_results.append(result)
            speedup = result.throughput / baseline_tp
            print(f"    >>> {result.throughput:.1f} t/s ({speedup:.2f}x)")
        
        all_results.extend(token_results)
        
        # Token 长度总结
        print_subheader(f"[{max_tokens} tokens 总结]")
        sorted_results = sorted(token_results, key=lambda x: x.throughput, reverse=True)
        
        print(f"\n{'排名':<4} {'方法':<50} {'吞吐量':>10} {'加速比':>8} {'内存':>10} {'接受率':>8}")
        print("-" * 100)
        
        for i, r in enumerate(sorted_results[:10]):
            speedup = r.throughput / baseline_tp
            accept_str = f"{r.acceptance_rate:.1%}" if r.acceptance_rate > 0 else "N/A"
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{marker}{i+1:<3} {r.method:<50} {r.throughput:>8.1f} t/s {speedup:>6.2f}x "
                  f"{r.memory_peak_mb:>8.0f} MB {accept_str:>8}")
    
    # ========================================================================
    # 全局分析
    # ========================================================================
    print_header("📊 全局分析")
    
    # 按方法类型分组
    method_groups = {
        'Baseline': [],
        'Linear': [],
        'Linear+Stream': [],
        'Tree': [],
        'Tree+Stream': []
    }
    
    for r in all_results:
        if 'Baseline' in r.method:
            method_groups['Baseline'].append(r)
        elif 'Stream' in r.method and 'Tree' in r.method:
            method_groups['Tree+Stream'].append(r)
        elif 'Tree' in r.method:
            method_groups['Tree'].append(r)
        elif 'Stream' in r.method:
            method_groups['Linear+Stream'].append(r)
        elif 'Linear' in r.method:
            method_groups['Linear'].append(r)
    
    # 找到每个类别的最佳配置
    print("\n各方法类型最佳配置:")
    print("-" * 80)
    
    best_configs = {}
    for group_name, results in method_groups.items():
        if results:
            best = max(results, key=lambda x: x.throughput)
            best_configs[group_name] = best
            baseline_for_tokens = next(
                (r for r in method_groups['Baseline'] if r.tokens == best.tokens),
                method_groups['Baseline'][0]
            )
            speedup = best.throughput / baseline_for_tokens.throughput
            
            print(f"\n{group_name}:")
            print(f"  最佳配置: {best.method}")
            print(f"  Tokens: {best.tokens}")
            print(f"  吞吐量: {best.throughput:.1f} t/s ({speedup:.2f}x)")
            print(f"  内存: {best.memory_peak_mb:.0f} MB")
            if best.acceptance_rate > 0:
                print(f"  接受率: {best.acceptance_rate:.1%}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/comprehensive_comparison_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    
    serializable_results = []
    for r in all_results:
        serializable_results.append(asdict(r))
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'target_model': TARGET_MODEL_PATH,
                'draft_model': DRAFT_MODEL_PATH,
                'token_lengths': TOKEN_LENGTHS,
                'linear_k_values': LINEAR_K_VALUES,
                'tree_depths': TREE_DEPTHS,
                'tree_branches': TREE_BRANCHES,
                'tree_thresholds': TREE_THRESHOLDS,
                'streaming_cache_sizes': STREAMING_CACHE_SIZES,
                'num_runs': NUM_RUNS
            },
            'results': serializable_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n\n结果已保存到: {result_file}")
    
    # 最终建议
    print_header("📝 最终建议")
    
    for tokens in TOKEN_LENGTHS:
        token_results = [r for r in all_results if r.tokens == tokens]
        baseline = next((r for r in token_results if 'Baseline' in r.method), None)
        best = max(token_results, key=lambda x: x.throughput)
        
        if baseline and best:
            speedup = best.throughput / baseline.throughput
            print(f"\n{tokens} tokens:")
            print(f"  推荐方法: {best.method}")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  吞吐量: {best.throughput:.1f} t/s (vs baseline {baseline.throughput:.1f} t/s)")


if __name__ == "__main__":
    main()







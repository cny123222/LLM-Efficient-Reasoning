#!/usr/bin/env python3
"""
Tree V2 + StreamingLLM Benchmark

对比 Tree V2 和 Tree V2 + StreamingLLM 在不同序列长度下的性能差异。

测试内容:
1. 短序列性能 (500 tokens) - StreamingLLM 可能有额外开销
2. 长序列性能 (2000+ tokens) - StreamingLLM 应该发挥优势
3. 内存使用对比
4. 压缩次数统计

Usage:
    python papers/benchmark_tree_streaming_v2.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import gc
import json
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer

from spec_decode.core import (
    TreeSpeculativeGeneratorV2,
    TreeStreamingSpeculativeGeneratorV2,
)


# 配置
TARGET_MODEL_PATH = "/mnt/disk1/models/pythia-2.8b"
DRAFT_MODEL_PATH = "/mnt/disk1/models/pythia-70m"
DEVICE = "cuda"

# Tree V2 最优配置
TREE_DEPTH = 8
TREE_BRANCH = 3
TREE_THRESHOLD = 0.03

# StreamingLLM 配置
STREAMING_CACHE_SIZES = [512, 1024, 2048]
START_SIZE = 4

# 测试配置
TOKEN_LENGTHS = [500, 1000, 2000]
NUM_RUNS = 3
SKIP_FIRST = True

PROMPT = """Write a comprehensive and detailed technical explanation about the development and evolution of large language models in artificial intelligence. Cover the complete history from early neural networks to modern transformers, discuss all major architecture innovations including attention mechanisms and scaling laws, explain the training techniques such as pre-training and fine-tuning, and provide insights into future directions and challenges.

Begin your detailed explanation:

The field of artificial intelligence has witnessed remarkable progress"""


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_gpu_memory() -> tuple:
    """获取 GPU 内存使用 (allocated, peak) in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return allocated, peak
    return 0.0, 0.0


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def benchmark_tree_v2(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    num_runs: int = NUM_RUNS
) -> Dict:
    """测试 Tree V2 性能"""
    
    gen = TreeSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=TREE_DEPTH,
        branch_factor=TREE_BRANCH,
        probability_threshold=TREE_THRESHOLD,
        max_tree_nodes=128,
        device=DEVICE,
        use_compile=False
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
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'memory_peak_mb': mem_peak
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        print(f"    Run {i+1}: {throughput:.1f} t/s {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak_mb'] for r in results) / len(results)
    
    return {
        'method': 'Tree V2',
        'max_new_tokens': max_new_tokens,
        'avg_throughput': avg_throughput,
        'avg_acceptance_rate': avg_accept,
        'avg_memory_peak_mb': avg_memory,
        'compression_count': 0,
        'runs': results
    }


def benchmark_tree_streaming_v2(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    max_cache_len: int,
    num_runs: int = NUM_RUNS
) -> Dict:
    """测试 Tree V2 + StreamingLLM 性能"""
    
    recent_size = max_cache_len - START_SIZE
    
    gen = TreeStreamingSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=TREE_DEPTH,
        branch_factor=TREE_BRANCH,
        probability_threshold=TREE_THRESHOLD,
        max_tree_nodes=128,
        device=DEVICE,
        use_compile=False,
        start_size=START_SIZE,
        recent_size=recent_size,
        max_cache_len=max_cache_len,
        compress_threshold=0.9
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
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'memory_peak_mb': mem_peak,
                'compression_count': stats.get('compression_count', 0)
            })
        
        status = "(warmup)" if SKIP_FIRST and i == 0 else ""
        compress = stats.get('compression_count', 0)
        print(f"    Run {i+1}: {throughput:.1f} t/s, compress={compress} {status}")
    
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak_mb'] for r in results) / len(results)
    avg_compress = sum(r['compression_count'] for r in results) / len(results)
    
    return {
        'method': f'Tree V2 + Streaming (cache={max_cache_len})',
        'max_new_tokens': max_new_tokens,
        'max_cache_len': max_cache_len,
        'avg_throughput': avg_throughput,
        'avg_acceptance_rate': avg_accept,
        'avg_memory_peak_mb': avg_memory,
        'compression_count': avg_compress,
        'runs': results
    }


def main():
    print_header("Tree V2 + StreamingLLM Benchmark")
    
    print(f"\n配置:")
    print(f"  Tree: D={TREE_DEPTH}, B={TREE_BRANCH}, t={TREE_THRESHOLD}")
    print(f"  StreamingLLM Cache Sizes: {STREAMING_CACHE_SIZES}")
    print(f"  Token Lengths: {TOKEN_LENGTHS}")
    
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
    gen = TreeSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=TREE_DEPTH, branch_factor=TREE_BRANCH,
        probability_threshold=TREE_THRESHOLD,
        device=DEVICE, use_compile=False
    )
    for _ in range(5):
        gen.reset()
        _ = gen.generate(PROMPT, max_new_tokens=50)
    torch.cuda.synchronize()
    del gen
    cleanup()
    
    all_results = []
    
    # 测试不同 token 长度
    for max_tokens in TOKEN_LENGTHS:
        print_header(f"测试 {max_tokens} tokens")
        
        # Tree V2 baseline
        print(f"\n[1] Tree V2 (无 StreamingLLM):")
        tree_result = benchmark_tree_v2(
            target_model, draft_model, tokenizer, max_tokens
        )
        all_results.append(tree_result)
        print(f"  >>> 平均: {tree_result['avg_throughput']:.1f} t/s, 内存: {tree_result['avg_memory_peak_mb']:.0f} MB")
        
        baseline_throughput = tree_result['avg_throughput']
        
        # Tree V2 + StreamingLLM (不同 cache 大小)
        for cache_len in STREAMING_CACHE_SIZES:
            print(f"\n[2] Tree V2 + StreamingLLM (cache={cache_len}):")
            
            try:
                stream_result = benchmark_tree_streaming_v2(
                    target_model, draft_model, tokenizer,
                    max_tokens, cache_len
                )
                all_results.append(stream_result)
                
                speedup = stream_result['avg_throughput'] / baseline_throughput
                mem_save = (tree_result['avg_memory_peak_mb'] - stream_result['avg_memory_peak_mb']) / tree_result['avg_memory_peak_mb'] * 100
                
                print(f"  >>> 平均: {stream_result['avg_throughput']:.1f} t/s ({speedup:.2f}x vs baseline)")
                print(f"      内存: {stream_result['avg_memory_peak_mb']:.0f} MB ({mem_save:+.1f}%)")
                print(f"      压缩次数: {stream_result['compression_count']:.0f}")
                
            except Exception as e:
                print(f"  ❌ 错误: {e}")
    
    # 结果汇总
    print_header("📊 结果汇总")
    
    print(f"\n{'方法':<40} {'Tokens':>8} {'吞吐量':>12} {'内存':>10} {'压缩':>8}")
    print("-" * 85)
    
    for r in all_results:
        method = r['method'][:38]
        tokens = r['max_new_tokens']
        throughput = r['avg_throughput']
        memory = r['avg_memory_peak_mb']
        compress = r.get('compression_count', 0)
        
        print(f"{method:<40} {tokens:>8} {throughput:>10.1f} t/s {memory:>8.0f} MB {compress:>8.0f}")
    
    # 分析
    print_header("🔍 分析")
    
    # 按 token 长度分组
    for max_tokens in TOKEN_LENGTHS:
        token_results = [r for r in all_results if r['max_new_tokens'] == max_tokens]
        
        if len(token_results) < 2:
            continue
        
        baseline = token_results[0]
        print(f"\n{max_tokens} tokens:")
        print(f"  Baseline (Tree V2): {baseline['avg_throughput']:.1f} t/s, {baseline['avg_memory_peak_mb']:.0f} MB")
        
        for r in token_results[1:]:
            speedup = r['avg_throughput'] / baseline['avg_throughput']
            mem_save = (baseline['avg_memory_peak_mb'] - r['avg_memory_peak_mb']) / baseline['avg_memory_peak_mb'] * 100
            
            cache_len = r.get('max_cache_len', 'N/A')
            compress = r.get('compression_count', 0)
            
            status = "✓" if speedup >= 0.95 else "⚠"
            print(f"  {status} + Streaming (cache={cache_len}): {r['avg_throughput']:.1f} t/s ({speedup:.2f}x), "
                  f"内存{mem_save:+.1f}%, 压缩{compress:.0f}次")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/tree_streaming_v2_benchmark_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'target_model': TARGET_MODEL_PATH,
                'draft_model': DRAFT_MODEL_PATH,
                'tree_depth': TREE_DEPTH,
                'tree_branch': TREE_BRANCH,
                'tree_threshold': TREE_THRESHOLD,
                'streaming_cache_sizes': STREAMING_CACHE_SIZES,
                'token_lengths': TOKEN_LENGTHS,
                'num_runs': NUM_RUNS
            },
            'results': all_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n\n结果已保存到: {result_file}")
    
    # 结论
    print_header("📝 结论")
    
    # 找到长序列下 StreamingLLM 的表现
    long_seq_results = [r for r in all_results if r['max_new_tokens'] == max(TOKEN_LENGTHS)]
    
    if len(long_seq_results) > 1:
        baseline = long_seq_results[0]
        best_streaming = max(long_seq_results[1:], key=lambda x: x['avg_throughput'])
        
        speedup = best_streaming['avg_throughput'] / baseline['avg_throughput']
        mem_save = (baseline['avg_memory_peak_mb'] - best_streaming['avg_memory_peak_mb']) / baseline['avg_memory_peak_mb'] * 100
        
        print(f"\n长序列 ({max(TOKEN_LENGTHS)} tokens) 下:")
        
        if speedup >= 0.95 and mem_save > 10:
            print(f"  ✓ StreamingLLM 推荐使用")
            print(f"    - 速度损失仅 {(1-speedup)*100:.1f}%")
            print(f"    - 内存节省 {mem_save:.1f}%")
        elif mem_save > 20:
            print(f"  ✓ StreamingLLM 适合内存受限场景")
            print(f"    - 速度损失 {(1-speedup)*100:.1f}%")
            print(f"    - 内存节省 {mem_save:.1f}%")
        else:
            print(f"  ⚠ StreamingLLM 在当前配置下收益有限")
            print(f"    - 考虑增加生成长度或减小 cache 大小")


if __name__ == "__main__":
    main()







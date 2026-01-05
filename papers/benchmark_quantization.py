#!/usr/bin/env python3
"""
INT8 量化性能 Benchmark

对比 FP16 和 INT8 量化下 Tree-based Speculative Decoding 的性能差异。

测试内容:
1. 吞吐量对比 (FP16 vs INT8)
2. 内存使用对比
3. 接受率变化
4. 不同 Token 长度下的性能

Usage:
    python papers/benchmark_quantization.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import gc
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer

from spec_decode.core import (
    TreeSpeculativeGeneratorV2,
    load_model_fp16,
    load_model_int8,
    BITSANDBYTES_AVAILABLE,
    get_model_memory_footprint,
)


# 配置
TARGET_MODEL_PATH = "/mnt/disk1/models/pythia-2.8b"
DRAFT_MODEL_PATH = "/mnt/disk1/models/pythia-70m"
DEVICE = "cuda"

# Tree V2 最优配置
TREE_DEPTH = 8
TREE_BRANCH = 3
TREE_THRESHOLD = 0.03

# 测试配置
TOKEN_LENGTHS = [100, 300, 500]
NUM_RUNS = 4
SKIP_FIRST = True

PROMPT = """Write a detailed technical explanation about the development of large language models. 
Cover the history, architecture innovations, training techniques, and future directions.
Begin your explanation:

Large language models have become"""


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_gpu_memory() -> Tuple[float, float]:
    """获取 GPU 内存使用 (allocated, reserved) in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_allocated() / 1024**2
        return allocated, reserved
    return 0.0, 0.0


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def benchmark_config(
    target_model,
    draft_model,
    tokenizer,
    max_new_tokens: int,
    config_name: str,
    num_runs: int = NUM_RUNS
) -> Dict:
    """测试单个配置的性能"""
    
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
        
        # 记录初始内存
        mem_before, _ = get_gpu_memory()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = gen.generate(PROMPT, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # 记录峰值内存
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
        print(f"    Run {i+1}: {throughput:.1f} t/s, accept={stats.get('acceptance_rate', 0):.1%} {status}")
    
    # 计算平均值
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    avg_accept = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_memory = sum(r['memory_peak_mb'] for r in results) / len(results)
    
    return {
        'config': config_name,
        'max_new_tokens': max_new_tokens,
        'avg_throughput': avg_throughput,
        'avg_acceptance_rate': avg_accept,
        'avg_memory_peak_mb': avg_memory,
        'runs': results
    }


def main():
    print_header("INT8 量化性能 Benchmark")
    
    # 检查 bitsandbytes
    if not BITSANDBYTES_AVAILABLE:
        print("\n⚠️  bitsandbytes 未安装，跳过 INT8 测试")
        print("   安装: pip install bitsandbytes")
        print("\n仅运行 FP16 baseline 测试...")
        run_int8_test = False
    else:
        print("\n✓ bitsandbytes 可用")
        run_int8_test = True
    
    # 加载 tokenizer
    print("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_results = []
    
    # =========================================================================
    # FP16 Baseline
    # =========================================================================
    print_header("FP16 Baseline 测试")
    
    print("\n加载 FP16 模型...")
    cleanup()
    
    target_fp16 = load_model_fp16(TARGET_MODEL_PATH, device_map=DEVICE)
    draft_fp16 = load_model_fp16(DRAFT_MODEL_PATH, device_map=DEVICE)
    
    fp16_mem = get_model_memory_footprint(target_fp16)
    print(f"  Target 模型内存: {fp16_mem['total_size_mb']:.1f} MB")
    print(f"  GPU 已分配: {fp16_mem['gpu_allocated_mb']:.1f} MB")
    
    # Warmup
    print("\nWarmup (5 runs)...")
    gen = TreeSpeculativeGeneratorV2(
        target_fp16, draft_fp16, tokenizer,
        tree_depth=TREE_DEPTH, branch_factor=TREE_BRANCH,
        probability_threshold=TREE_THRESHOLD,
        device=DEVICE, use_compile=False
    )
    for _ in range(5):
        gen.reset()
        _ = gen.generate(PROMPT, max_new_tokens=50)
    torch.cuda.synchronize()
    
    # 测试不同 token 长度
    fp16_results = []
    for max_tokens in TOKEN_LENGTHS:
        print(f"\n测试 FP16 - {max_tokens} tokens:")
        result = benchmark_config(
            target_fp16, draft_fp16, tokenizer,
            max_tokens, f"FP16_{max_tokens}"
        )
        fp16_results.append(result)
        all_results.append(result)
        print(f"  >>> 平均: {result['avg_throughput']:.1f} t/s, 接受率: {result['avg_acceptance_rate']:.1%}")
    
    # 释放 FP16 模型
    del target_fp16, draft_fp16, gen
    cleanup()
    
    # =========================================================================
    # INT8 测试
    # =========================================================================
    if run_int8_test:
        print_header("INT8 量化测试")
        
        print("\n加载 INT8 量化模型...")
        cleanup()
        
        try:
            target_int8 = load_model_int8(TARGET_MODEL_PATH, device_map=DEVICE)
            draft_fp16 = load_model_fp16(DRAFT_MODEL_PATH, device_map=DEVICE)  # Draft 保持 FP16
            
            int8_mem = get_model_memory_footprint(target_int8)
            print(f"  Target 模型内存 (INT8): {int8_mem['total_size_mb']:.1f} MB")
            print(f"  GPU 已分配: {int8_mem['gpu_allocated_mb']:.1f} MB")
            print(f"  内存节省: {(fp16_mem['gpu_allocated_mb'] - int8_mem['gpu_allocated_mb']):.1f} MB")
            
            # Warmup
            print("\nWarmup (5 runs)...")
            gen = TreeSpeculativeGeneratorV2(
                target_int8, draft_fp16, tokenizer,
                tree_depth=TREE_DEPTH, branch_factor=TREE_BRANCH,
                probability_threshold=TREE_THRESHOLD,
                device=DEVICE, use_compile=False
            )
            for _ in range(5):
                gen.reset()
                _ = gen.generate(PROMPT, max_new_tokens=50)
            torch.cuda.synchronize()
            
            # 测试不同 token 长度
            int8_results = []
            for max_tokens in TOKEN_LENGTHS:
                print(f"\n测试 INT8 - {max_tokens} tokens:")
                result = benchmark_config(
                    target_int8, draft_fp16, tokenizer,
                    max_tokens, f"INT8_{max_tokens}"
                )
                int8_results.append(result)
                all_results.append(result)
                print(f"  >>> 平均: {result['avg_throughput']:.1f} t/s, 接受率: {result['avg_acceptance_rate']:.1%}")
            
            # 释放 INT8 模型
            del target_int8, draft_fp16, gen
            cleanup()
            
        except Exception as e:
            print(f"\n❌ INT8 加载失败: {e}")
            int8_results = []
    else:
        int8_results = []
    
    # =========================================================================
    # 结果汇总
    # =========================================================================
    print_header("📊 结果汇总")
    
    print(f"\n{'配置':<20} {'Tokens':<8} {'吞吐量':>12} {'接受率':>10} {'内存峰值':>12}")
    print("-" * 70)
    
    for r in all_results:
        config = r['config'].split('_')[0]
        tokens = r['max_new_tokens']
        print(f"{config:<20} {tokens:<8} {r['avg_throughput']:>10.1f} t/s {r['avg_acceptance_rate']:>9.1%} {r['avg_memory_peak_mb']:>10.1f} MB")
    
    # 对比分析
    if fp16_results and int8_results:
        print_header("🔍 FP16 vs INT8 对比")
        
        for fp16_r, int8_r in zip(fp16_results, int8_results):
            tokens = fp16_r['max_new_tokens']
            speedup = int8_r['avg_throughput'] / fp16_r['avg_throughput']
            memory_save = (fp16_r['avg_memory_peak_mb'] - int8_r['avg_memory_peak_mb']) / fp16_r['avg_memory_peak_mb'] * 100
            accept_diff = int8_r['avg_acceptance_rate'] - fp16_r['avg_acceptance_rate']
            
            print(f"\n{tokens} tokens:")
            print(f"  FP16: {fp16_r['avg_throughput']:.1f} t/s, INT8: {int8_r['avg_throughput']:.1f} t/s")
            print(f"  速度变化: {speedup:.2f}x ({'提升' if speedup > 1 else '下降'})")
            print(f"  内存节省: {memory_save:.1f}%")
            print(f"  接受率变化: {accept_diff:+.1%}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/quantization_benchmark_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'target_model': TARGET_MODEL_PATH,
                'draft_model': DRAFT_MODEL_PATH,
                'tree_depth': TREE_DEPTH,
                'tree_branch': TREE_BRANCH,
                'tree_threshold': TREE_THRESHOLD,
                'token_lengths': TOKEN_LENGTHS,
                'num_runs': NUM_RUNS
            },
            'results': all_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n\n结果已保存到: {result_file}")
    
    # 结论
    print_header("📝 结论")
    
    if int8_results:
        avg_speedup = sum(
            int8_r['avg_throughput'] / fp16_r['avg_throughput']
            for fp16_r, int8_r in zip(fp16_results, int8_results)
        ) / len(fp16_results)
        
        if avg_speedup > 1:
            print(f"\n✓ INT8 量化平均提速: {avg_speedup:.2f}x")
        else:
            print(f"\n⚠ INT8 量化平均降速: {avg_speedup:.2f}x")
        
        print("\n建议:")
        if avg_speedup > 1.1:
            print("  - INT8 量化在该硬件上有显著收益，推荐使用")
        elif avg_speedup > 0.95:
            print("  - INT8 量化性能基本持平，可根据内存需求选择")
        else:
            print("  - INT8 量化有性能损失，建议继续使用 FP16")
    else:
        print("\n未能完成 INT8 测试，请安装 bitsandbytes 后重试")


if __name__ == "__main__":
    main()







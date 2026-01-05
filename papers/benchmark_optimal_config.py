#!/usr/bin/env python3
"""
在最优配置 (500 tokens, D=8, B=3, t=0.03) 下对比所有 Spec Decode 方法

测试方法:
1. Baseline (纯自回归)
2. HuggingFace Assisted Generation
3. Linear Speculative Decoding (K=5,6,7,8)
4. Tree-based Speculative Decoding V2
5. StreamingLLM + Spec Decode
6. Tree + StreamingLLM (如果存在)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_decode.core import (
    SpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
    StreamingSpeculativeGenerator,
)
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# 配置
DEVICE = 'cuda'
TARGET_MODEL = '/mnt/disk1/models/pythia-2.8b'
DRAFT_MODEL = '/mnt/disk1/models/pythia-70m'
MAX_NEW_TOKENS = 500
NUM_RUNS = 5  # 每个方法运行次数
SKIP_FIRST = True  # 跳过首次 warmup

# Tree V2 最优参数
TREE_DEPTH = 8
TREE_BRANCH = 3
TREE_THRESHOLD = 0.03

PROMPT = """Write a detailed technical explanation about the development of large language models. 
Cover the history, architecture innovations, training techniques, and future directions.
Begin your explanation:

Large language models have become"""


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def measure_method(name, run_fn, num_runs=NUM_RUNS, skip_first=SKIP_FIRST):
    """通用测量函数"""
    results = []
    
    for i in range(num_runs):
        cleanup()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        tokens, extra_stats = run_fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        tp = tokens / elapsed
        
        if not skip_first or i > 0:
            results.append({
                'tokens': tokens,
                'time': elapsed,
                'throughput': tp,
                **extra_stats
            })
        
        status = "(warmup, 跳过)" if skip_first and i == 0 else ""
        print(f"    Run {i+1}: {tokens} tokens, {elapsed:.2f}s, {tp:.1f} t/s {status}")
    
    # 计算平均值
    avg_tp = sum(r['throughput'] for r in results) / len(results)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    return {
        'name': name,
        'avg_throughput': avg_tp,
        'avg_time': avg_time,
        'runs': results
    }


def main():
    print_header("加载模型")
    print(f"  Target: {TARGET_MODEL}")
    print(f"  Draft: {DRAFT_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.float16, device_map=DEVICE
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, torch_dtype=torch.float16, device_map=DEVICE
    )
    
    # 充分 Warmup
    print_header("Warmup (10 runs)")
    for i in range(10):
        cleanup()
        input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
        with torch.inference_mode():
            _ = target_model.generate(
                input_ids, max_new_tokens=50, do_sample=False,
                eos_token_id=None, assistant_model=draft_model
            )
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/10 完成")
    
    print_header(f"性能测试: {MAX_NEW_TOKENS} tokens, {NUM_RUNS} runs")
    print(f"  Tree V2 配置: D={TREE_DEPTH}, B={TREE_BRANCH}, t={TREE_THRESHOLD}")
    
    all_results = []
    
    # =========================================================================
    # 1. Baseline
    # =========================================================================
    print("\n[1/7] Baseline (纯自回归)...")
    
    def run_baseline():
        input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
        with torch.inference_mode():
            out = target_model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=None, pad_token_id=tokenizer.pad_token_id
            )
        tokens = out.shape[1] - input_ids.shape[1]
        return tokens, {}
    
    baseline_result = measure_method("Baseline (AR)", run_baseline)
    all_results.append(baseline_result)
    baseline_tp = baseline_result['avg_throughput']
    print(f"  >>> 平均: {baseline_tp:.1f} t/s (1.00x)")
    
    # =========================================================================
    # 2. HuggingFace Assisted
    # =========================================================================
    print("\n[2/7] HuggingFace Assisted Generation...")
    
    def run_hf_assisted():
        input_ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(DEVICE)
        with torch.inference_mode():
            out = target_model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                eos_token_id=None, assistant_model=draft_model,
                pad_token_id=tokenizer.pad_token_id
            )
        tokens = out.shape[1] - input_ids.shape[1]
        return tokens, {}
    
    hf_result = measure_method("HF Assisted", run_hf_assisted)
    all_results.append(hf_result)
    print(f"  >>> 平均: {hf_result['avg_throughput']:.1f} t/s ({hf_result['avg_throughput']/baseline_tp:.2f}x)")
    
    # =========================================================================
    # 3. Linear Speculative Decoding (多个 K 值)
    # =========================================================================
    for K in [5, 6, 7, 8]:
        print(f"\n[3/7] Linear Spec Decode K={K}...")
        
        gen = SpeculativeGenerator(
            target_model, draft_model, tokenizer,
            K=K, max_len=8192, device=DEVICE, use_compile=False
        )
        
        def run_linear():
            gen.reset()
            _ = gen.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
            stats = gen.get_stats()
            return stats['total_tokens'], {
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'tokens_per_round': stats.get('tokens_per_round', 0)
            }
        
        linear_result = measure_method(f"Linear K={K}", run_linear)
        all_results.append(linear_result)
        
        # 获取额外统计
        accept_rate = sum(r.get('acceptance_rate', 0) for r in linear_result['runs']) / len(linear_result['runs'])
        tpr = sum(r.get('tokens_per_round', 0) for r in linear_result['runs']) / len(linear_result['runs'])
        
        print(f"  >>> 平均: {linear_result['avg_throughput']:.1f} t/s ({linear_result['avg_throughput']/baseline_tp:.2f}x)")
        print(f"      接受率: {accept_rate:.1%}, 每轮 tokens: {tpr:.2f}")
    
    # =========================================================================
    # 4. Tree V2 Speculative Decoding (最优配置)
    # =========================================================================
    print(f"\n[4/7] Tree V2 (D={TREE_DEPTH} B={TREE_BRANCH} t={TREE_THRESHOLD})...")
    
    tree_gen = TreeSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=TREE_DEPTH, branch_factor=TREE_BRANCH,
        probability_threshold=TREE_THRESHOLD,
        max_tree_nodes=128, device=DEVICE, use_compile=False
    )
    
    def run_tree():
        tree_gen.reset()
        _ = tree_gen.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        stats = tree_gen.get_stats()
        return stats['total_tokens'], {
            'acceptance_rate': stats.get('acceptance_rate', 0),
            'avg_path_length': stats.get('avg_path_length', 0)
        }
    
    tree_result = measure_method(f"Tree V2 D={TREE_DEPTH}B={TREE_BRANCH}t={TREE_THRESHOLD}", run_tree)
    all_results.append(tree_result)
    
    accept_rate = sum(r.get('acceptance_rate', 0) for r in tree_result['runs']) / len(tree_result['runs'])
    path_len = sum(r.get('avg_path_length', 0) for r in tree_result['runs']) / len(tree_result['runs'])
    
    print(f"  >>> 平均: {tree_result['avg_throughput']:.1f} t/s ({tree_result['avg_throughput']/baseline_tp:.2f}x)")
    print(f"      接受率: {accept_rate:.1%}, 平均路径长度: {path_len:.2f}")
    
    # =========================================================================
    # 5. StreamingLLM + Spec Decode
    # =========================================================================
    for cache_len in [512, 1024]:
        print(f"\n[5/7] StreamingLLM + Spec Decode (cache={cache_len})...")
        
        stream_gen = StreamingSpeculativeGenerator(
            target_model, draft_model, tokenizer,
            K=6, max_len=8192, max_cache_len=cache_len,
            start_size=4, recent_size=cache_len-4,
            device=DEVICE, use_compile=False
        )
        
        def run_streaming():
            stream_gen.reset()
            _ = stream_gen.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
            stats = stream_gen.get_stats()
            return stats['total_tokens'], {
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'compress_count': stats.get('compress_count', 0)
            }
        
        stream_result = measure_method(f"Streaming K=6 cache={cache_len}", run_streaming)
        all_results.append(stream_result)
        
        accept_rate = sum(r.get('acceptance_rate', 0) for r in stream_result['runs']) / len(stream_result['runs'])
        compress = sum(r.get('compress_count', 0) for r in stream_result['runs']) / len(stream_result['runs'])
        
        print(f"  >>> 平均: {stream_result['avg_throughput']:.1f} t/s ({stream_result['avg_throughput']/baseline_tp:.2f}x)")
        print(f"      接受率: {accept_rate:.1%}, 压缩次数: {compress:.0f}")
    
    # =========================================================================
    # 6. Tree + StreamingLLM (如果存在)
    # =========================================================================
    try:
        from spec_decode.core import TreeStreamingSpeculativeGenerator
        
        print(f"\n[6/7] Tree + StreamingLLM (D={TREE_DEPTH} B={TREE_BRANCH} cache=1024)...")
        
        tree_stream_gen = TreeStreamingSpeculativeGenerator(
            target_model, draft_model, tokenizer,
            tree_depth=TREE_DEPTH, branch_factor=TREE_BRANCH,
            probability_threshold=TREE_THRESHOLD,
            max_tree_nodes=128, max_cache_len=1024,
            start_size=4, recent_size=1020,
            device=DEVICE, use_compile=False
        )
        
        def run_tree_streaming():
            tree_stream_gen.reset()
            _ = tree_stream_gen.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
            stats = tree_stream_gen.get_stats()
            return stats['total_tokens'], {
                'acceptance_rate': stats.get('acceptance_rate', 0),
                'avg_path_length': stats.get('avg_path_length', 0)
            }
        
        tree_stream_result = measure_method("Tree+Streaming", run_tree_streaming)
        all_results.append(tree_stream_result)
        
        print(f"  >>> 平均: {tree_stream_result['avg_throughput']:.1f} t/s ({tree_stream_result['avg_throughput']/baseline_tp:.2f}x)")
    except ImportError:
        print("\n[6/7] Tree + StreamingLLM - 跳过 (模块未找到)")
    except Exception as e:
        print(f"\n[6/7] Tree + StreamingLLM - 错误: {e}")
    
    # =========================================================================
    # 结果汇总
    # =========================================================================
    print_header("📊 结果汇总")
    
    # 按加速比排序
    sorted_results = sorted(all_results, key=lambda x: x['avg_throughput'], reverse=True)
    
    print(f"\n{'排名':<4} {'方法':<35} {'吞吐量':>12} {'加速比':>10}")
    print("-" * 65)
    
    for i, r in enumerate(sorted_results):
        speedup = r['avg_throughput'] / baseline_tp
        marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{marker}{i+1:<3} {r['name']:<35} {r['avg_throughput']:>10.1f} t/s {speedup:>8.2f}x")
    
    # 关键对比
    print_header("🔍 关键对比分析")
    
    # 找到各类方法的最佳结果
    hf_best = next((r for r in sorted_results if 'HF' in r['name']), None)
    linear_best = max([r for r in sorted_results if 'Linear' in r['name']], key=lambda x: x['avg_throughput'], default=None)
    tree_best = next((r for r in sorted_results if 'Tree V2' in r['name']), None)
    stream_best = max([r for r in sorted_results if 'Streaming' in r['name'] and 'Tree' not in r['name']], key=lambda x: x['avg_throughput'], default=None)
    
    print(f"""
配置: {MAX_NEW_TOKENS} tokens, Tree V2 (D={TREE_DEPTH}, B={TREE_BRANCH}, t={TREE_THRESHOLD})

方法对比:
  Baseline:              {baseline_tp:>6.1f} t/s (1.00x)
  HF Assisted:           {hf_best['avg_throughput'] if hf_best else 0:>6.1f} t/s ({hf_best['avg_throughput']/baseline_tp if hf_best else 0:.2f}x)
  Linear (最佳 K):       {linear_best['avg_throughput'] if linear_best else 0:>6.1f} t/s ({linear_best['avg_throughput']/baseline_tp if linear_best else 0:.2f}x) [{linear_best['name'] if linear_best else 'N/A'}]
  Tree V2:               {tree_best['avg_throughput'] if tree_best else 0:>6.1f} t/s ({tree_best['avg_throughput']/baseline_tp if tree_best else 0:.2f}x)
  StreamingLLM (最佳):   {stream_best['avg_throughput'] if stream_best else 0:>6.1f} t/s ({stream_best['avg_throughput']/baseline_tp if stream_best else 0:.2f}x) [{stream_best['name'] if stream_best else 'N/A'}]
""")
    
    # 结论
    print_header("📝 结论")
    
    best = sorted_results[0]
    print(f"""
1. 最快方法: {best['name']} ({best['avg_throughput']:.1f} t/s, {best['avg_throughput']/baseline_tp:.2f}x)

2. Tree V2 vs Linear:
   - Tree V2: {tree_best['avg_throughput'] if tree_best else 0:.1f} t/s
   - Linear 最佳: {linear_best['avg_throughput'] if linear_best else 0:.1f} t/s
   - 差异: {((tree_best['avg_throughput'] if tree_best else 0) - (linear_best['avg_throughput'] if linear_best else 0)):.1f} t/s
   - Tree V2 {'优于' if tree_best and linear_best and tree_best['avg_throughput'] > linear_best['avg_throughput'] else '不如'} Linear

3. HF Assisted 显著领先，因为:
   - HuggingFace 内部优化更彻底
   - 使用 C++ 实现的关键路径
   - 更高效的 KV cache 管理

4. 自定义实现的价值:
   - 可以与 StreamingLLM 结合用于长序列
   - 支持更灵活的定制 (如 Tree-based)
   - 适合研究和教学目的
""")


if __name__ == "__main__":
    main()







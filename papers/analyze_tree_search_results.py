#!/usr/bin/env python3
"""
分析 Tree V2 参数搜索结果
"""

import json
import sys
from collections import defaultdict

def analyze_results(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    config = data['config']
    
    print("=" * 70)
    print("Tree V2 参数搜索结果分析")
    print("=" * 70)
    print(f"\n配置信息:")
    print(f"  Target Model: {config['target_model']}")
    print(f"  Draft Model: {config['draft_model']}")
    print(f"  Depths: {config['depths']}")
    print(f"  Branches: {config['branches']}")
    print(f"  Thresholds: {config['thresholds']}")
    print(f"  Token Lengths: {config['token_lengths']}")
    print(f"  总配置数: {len(results)}")
    
    # 过滤有效结果
    valid_results = [r for r in results if r['throughput'] > 0 and r['speedup'] > 0]
    print(f"  有效结果数: {len(valid_results)}")
    
    # 按加速比排序
    sorted_by_speedup = sorted(valid_results, key=lambda x: x['speedup'], reverse=True)
    
    print("\n" + "=" * 70)
    print("TOP 20 最高加速比配置")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Tokens':<7} {'D':<3} {'B':<3} {'t':<6} {'Throughput':>12} {'Speedup':>10} {'PathLen':>8} {'AccRate':>8}")
    print("-" * 70)
    
    for i, r in enumerate(sorted_by_speedup[:20]):
        print(f"{i+1:<5} {r['tokens']:<7} {r['depth']:<3} {r['branch']:<3} {r['threshold']:<6.2f} "
              f"{r['throughput']:>10.1f} t/s {r['speedup']:>8.2f}x {r['avg_path_length']:>8.2f} {r['acceptance_rate']:>7.1%}")
    
    # 按 token 长度分组分析
    print("\n" + "=" * 70)
    print("按 Token 长度分组的最优配置")
    print("=" * 70)
    
    by_tokens = defaultdict(list)
    for r in valid_results:
        by_tokens[r['tokens']].append(r)
    
    for tokens in sorted(by_tokens.keys()):
        token_results = by_tokens[tokens]
        best = max(token_results, key=lambda x: x['speedup'])
        baseline = best['baseline_throughput']
        
        print(f"\n{tokens} tokens (baseline: {baseline:.1f} t/s):")
        print(f"  最优: D={best['depth']} B={best['branch']} t={best['threshold']:.2f}")
        print(f"  吞吐量: {best['throughput']:.1f} t/s")
        print(f"  加速比: {best['speedup']:.2f}x")
        print(f"  路径长度: {best['avg_path_length']:.2f}")
        
        # Top 3 for this token length
        top3 = sorted(token_results, key=lambda x: x['speedup'], reverse=True)[:3]
        print(f"  Top 3:")
        for i, r in enumerate(top3):
            print(f"    {i+1}. D={r['depth']} B={r['branch']} t={r['threshold']:.2f} -> {r['speedup']:.2f}x")
    
    # 按参数维度分析
    print("\n" + "=" * 70)
    print("按参数维度分析 (取各参数下的平均加速比)")
    print("=" * 70)
    
    # By Depth
    by_depth = defaultdict(list)
    for r in valid_results:
        by_depth[r['depth']].append(r['speedup'])
    
    print("\n按 Depth:")
    for d in sorted(by_depth.keys()):
        avg = sum(by_depth[d]) / len(by_depth[d])
        max_s = max(by_depth[d])
        print(f"  D={d}: avg={avg:.2f}x, max={max_s:.2f}x")
    
    # By Branch
    by_branch = defaultdict(list)
    for r in valid_results:
        by_branch[r['branch']].append(r['speedup'])
    
    print("\n按 Branch:")
    for b in sorted(by_branch.keys()):
        avg = sum(by_branch[b]) / len(by_branch[b])
        max_s = max(by_branch[b])
        print(f"  B={b}: avg={avg:.2f}x, max={max_s:.2f}x")
    
    # By Threshold
    by_threshold = defaultdict(list)
    for r in valid_results:
        by_threshold[r['threshold']].append(r['speedup'])
    
    print("\n按 Threshold:")
    for t in sorted(by_threshold.keys()):
        avg = sum(by_threshold[t]) / len(by_threshold[t])
        max_s = max(by_threshold[t])
        print(f"  t={t:.2f}: avg={avg:.2f}x, max={max_s:.2f}x")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("统计信息")
    print("=" * 70)
    
    speedups = [r['speedup'] for r in valid_results]
    print(f"\n加速比统计:")
    print(f"  最大: {max(speedups):.2f}x")
    print(f"  最小: {min(speedups):.2f}x")
    print(f"  平均: {sum(speedups)/len(speedups):.2f}x")
    print(f"  中位数: {sorted(speedups)[len(speedups)//2]:.2f}x")
    
    # 超过特定阈值的配置数
    print(f"\n加速比分布:")
    for threshold in [1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        count = sum(1 for s in speedups if s >= threshold)
        print(f"  >= {threshold:.1f}x: {count} 个配置 ({count/len(speedups)*100:.1f}%)")
    
    # 全局最优
    global_best = sorted_by_speedup[0]
    print("\n" + "=" * 70)
    print("全局最优配置")
    print("=" * 70)
    print(f"\n  Tokens: {global_best['tokens']}")
    print(f"  Depth (D): {global_best['depth']}")
    print(f"  Branch (B): {global_best['branch']}")
    print(f"  Threshold (t): {global_best['threshold']}")
    print(f"  Throughput: {global_best['throughput']:.1f} t/s")
    print(f"  Baseline: {global_best['baseline_throughput']:.1f} t/s")
    print(f"  Speedup: {global_best['speedup']:.2f}x")
    print(f"  Avg Path Length: {global_best['avg_path_length']:.2f}")
    print(f"  Acceptance Rate: {global_best['acceptance_rate']:.1%}")
    
    # 检查是否有异常高的加速比
    print("\n" + "=" * 70)
    print("异常检查")
    print("=" * 70)
    
    # 检查 baseline 一致性
    baselines_by_tokens = {}
    for r in valid_results:
        t = r['tokens']
        if t not in baselines_by_tokens:
            baselines_by_tokens[t] = []
        baselines_by_tokens[t].append(r['baseline_throughput'])
    
    print("\nBaseline 一致性检查:")
    for t in sorted(baselines_by_tokens.keys()):
        baselines = baselines_by_tokens[t]
        if max(baselines) - min(baselines) > 1:
            print(f"  {t} tokens: baseline 不一致! min={min(baselines):.1f}, max={max(baselines):.1f}")
        else:
            print(f"  {t} tokens: baseline={baselines[0]:.1f} t/s (一致)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认使用最新的搜索结果
        import glob
        files = sorted(glob.glob("results/tree_param_search_*.json"))
        if files:
            json_path = files[-1]
        else:
            print("Usage: python analyze_tree_search_results.py <json_path>")
            sys.exit(1)
    else:
        json_path = sys.argv[1]
    
    print(f"分析文件: {json_path}\n")
    analyze_results(json_path)


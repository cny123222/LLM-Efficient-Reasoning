#!/usr/bin/env python3
"""
分析加速比差异问题

探究为什么之前报告的加速比与现在测试结果不一致
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_decode.core import SpeculativeGenerator, TreeSpeculativeGeneratorV2
import time
import gc
import warnings
warnings.filterwarnings('ignore')

device = 'cuda'
target_path = '/mnt/disk1/models/pythia-2.8b'
draft_path = '/mnt/disk1/models/pythia-70m'

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(target_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch.float16, device_map=device)
draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=torch.float16, device_map=device)

prompt = 'Write a detailed explanation about the development of large language models:'

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def measure_baseline(max_new_tokens, num_runs=5, skip_first=True):
    """精确测量 baseline"""
    tps = []
    for i in range(num_runs):
        cleanup()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = target_model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=None, pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = out.shape[1] - input_ids.shape[1]
        tp = tokens / elapsed
        if not skip_first or i > 0:
            tps.append(tp)
    return sum(tps) / len(tps), tps

def measure_hf_assisted(max_new_tokens, num_runs=5, skip_first=True):
    """精确测量 HF Assisted"""
    tps = []
    for i in range(num_runs):
        cleanup()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = target_model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=None, assistant_model=draft_model,
                pad_token_id=tokenizer.pad_token_id
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = out.shape[1] - input_ids.shape[1]
        tp = tokens / elapsed
        if not skip_first or i > 0:
            tps.append(tp)
    return sum(tps) / len(tps), tps

def measure_linear(max_new_tokens, K=5, num_runs=5, skip_first=True):
    """精确测量 Linear Spec Decode"""
    cleanup()
    gen = SpeculativeGenerator(target_model, draft_model, tokenizer, K=K, max_len=8192, device=device, use_compile=False)
    tps = []
    for i in range(num_runs):
        gen.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = gen.generate(prompt, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        stats = gen.get_stats()
        tp = stats['total_tokens'] / elapsed
        if not skip_first or i > 0:
            tps.append(tp)
    return sum(tps) / len(tps), tps

def measure_tree(max_new_tokens, D=8, B=3, t=0.03, num_runs=5, skip_first=True):
    """精确测量 Tree V2"""
    cleanup()
    gen = TreeSpeculativeGeneratorV2(
        target_model, draft_model, tokenizer,
        tree_depth=D, branch_factor=B, probability_threshold=t,
        max_tree_nodes=128, device=device, use_compile=False
    )
    tps = []
    for i in range(num_runs):
        gen.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = gen.generate(prompt, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        stats = gen.get_stats()
        tp = stats['total_tokens'] / elapsed
        if not skip_first or i > 0:
            tps.append(tp)
    return sum(tps) / len(tps), tps

# Warmup
print("\n大量 Warmup (10 runs)...")
for _ in range(10):
    cleanup()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.inference_mode():
        _ = target_model.generate(input_ids, max_new_tokens=50, do_sample=False,
                                  eos_token_id=None, assistant_model=draft_model)
    torch.cuda.synchronize()

print("\n" + "=" * 70)
print("详细性能分析")
print("=" * 70)

# 测试不同 token 长度
for max_tokens in [100, 300, 500, 1000]:
    print(f"\n{'='*70}")
    print(f"测试: {max_tokens} tokens (5 runs, 跳过首次)")
    print(f"{'='*70}")
    
    baseline_tp, baseline_runs = measure_baseline(max_tokens, num_runs=5)
    print(f"\nBaseline:")
    print(f"  Runs: {[f'{t:.1f}' for t in baseline_runs]}")
    print(f"  Avg: {baseline_tp:.1f} t/s")
    
    hf_tp, hf_runs = measure_hf_assisted(max_tokens, num_runs=5)
    print(f"\nHF Assisted:")
    print(f"  Runs: {[f'{t:.1f}' for t in hf_runs]}")
    print(f"  Avg: {hf_tp:.1f} t/s ({hf_tp/baseline_tp:.2f}x)")
    
    linear_tp, linear_runs = measure_linear(max_tokens, K=6, num_runs=5)
    print(f"\nLinear K=6:")
    print(f"  Runs: {[f'{t:.1f}' for t in linear_runs]}")
    print(f"  Avg: {linear_tp:.1f} t/s ({linear_tp/baseline_tp:.2f}x)")
    
    tree_tp, tree_runs = measure_tree(max_tokens, D=8, B=3, t=0.03, num_runs=5)
    print(f"\nTree V2 (D=8 B=3 t=0.03):")
    print(f"  Runs: {[f'{t:.1f}' for t in tree_runs]}")
    print(f"  Avg: {tree_tp:.1f} t/s ({tree_tp/baseline_tp:.2f}x)")
    
    print(f"\n>>> 总结 ({max_tokens} tokens):")
    print(f"    Baseline:    {baseline_tp:>6.1f} t/s (1.00x)")
    print(f"    HF Assisted: {hf_tp:>6.1f} t/s ({hf_tp/baseline_tp:.2f}x)")
    print(f"    Linear K=6:  {linear_tp:>6.1f} t/s ({linear_tp/baseline_tp:.2f}x)")
    print(f"    Tree V2:     {tree_tp:>6.1f} t/s ({tree_tp/baseline_tp:.2f}x)")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)


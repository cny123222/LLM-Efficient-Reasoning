
import sys
sys.path.insert(0, "/mnt/disk1/ljm/LLM-Efficient-Reasoning")

import json
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
import time
import gc

# Import models and generators
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_decode.core.tree_speculative_generator_adaptive import TreeSpeculativeGeneratorV2AdaptiveV3
from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
from spec_decode.core.speculative_generator import SpeculativeGenerator
from datasets import load_dataset

@dataclass
class PaperMetrics:
    method: str
    experiment: str
    config: dict
    throughput_tps: float = 0.0
    speedup: float = 1.0
    ttft_ms: float = 0.0
    ttft_std: float = 0.0
    tpot_ms: float = 0.0
    tpot_std: float = 0.0
    total_time_ms: float = 0.0
    total_tokens_generated: int = 0
    tokens_per_round: float = 0.0
    acceptance_rate: float = 0.0
    avg_path_length: float = 0.0
    total_rounds: int = 0
    high_conf_ratio: float = 0.0
    medium_conf_ratio: float = 0.0
    low_conf_ratio: float = 0.0
    early_stops: int = 0
    deep_expansions: int = 0
    total_adjustments: int = 0
    peak_memory_mb: float = 0.0
    throughput_std: float = 0.0
    acceptance_std: float = 0.0
    path_length_std: float = 0.0

def run_experiment(token_length: int, num_samples: int, warmup_runs: int, max_prompt_length: int):
    print(f"\n{'='*60}")
    print(f"Running scalability experiment for {token_length} tokens")
    print(f"{'='*60}")
    
    # Load models
    print("Loading models...")
    target_model_path = "/mnt/disk1/models/pythia-2.8b"
    draft_model_path = "/mnt/disk1/models/pythia-70m"
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    target_model.eval()
    draft_model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    prompts = []
    for item in dataset:
        text = item["text"].strip()
        if len(text) > 100:
            if len(text) > max_prompt_length:
                text = text[:max_prompt_length]
            prompts.append(text)
            if len(prompts) >= num_samples + warmup_runs:
                break
    
    results = []
    baseline_throughput = None
    
    # ===== Baseline =====
    print(f"\n[1/4] Baseline (AR)...")
    all_throughput, all_ttft, all_tpot = [], [], []
    
    for i, prompt in enumerate(prompts):
        if i < warmup_runs:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
            with torch.no_grad():
                target_model.generate(input_ids, max_new_tokens=token_length, do_sample=False)
            continue
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = target_model.generate(input_ids, max_new_tokens=token_length, do_sample=False)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
        generated = output.shape[1] - input_ids.shape[1]
        throughput = generated / total_time
        all_throughput.append(throughput)
        all_ttft.append(total_time / generated * 1000)
        all_tpot.append(total_time / generated * 1000)
    
    baseline_throughput = np.mean(all_throughput)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    metrics = PaperMetrics(
        method="Baseline (AR)",
        experiment=f"scalability_{token_length}",
        config={"max_new_tokens": token_length},
        throughput_tps=baseline_throughput,
        speedup=1.0,
        ttft_ms=np.mean(all_ttft),
        ttft_std=np.std(all_ttft),
        tpot_ms=np.mean(all_tpot),
        tpot_std=np.std(all_tpot),
        total_tokens_generated=token_length,
        throughput_std=np.std(all_throughput),
        peak_memory_mb=peak_mem
    )
    results.append(asdict(metrics))
    print(f"  Throughput: {baseline_throughput:.1f} t/s")
    
    # ===== Linear Spec =====
    print(f"\n[2/4] Linear Spec (K=5)...")
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        device="cuda"
    )
    
    all_throughput, all_ttft, all_tpot, all_acceptance = [], [], [], []
    
    for i, prompt in enumerate(prompts):
        if i < warmup_runs:
            generator.generate(prompt, max_new_tokens=min(50, token_length))
            continue
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        output, stats = generator.generate(prompt, max_new_tokens=token_length, return_stats=True)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
        generated = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
        throughput = generated / total_time
        all_throughput.append(throughput)
        all_ttft.append(total_time / generated * 1000)
        all_tpot.append(total_time / generated * 1000)
        if stats:
            all_acceptance.append(stats.get('acceptance_rate', 0))
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    metrics = PaperMetrics(
        method="Linear Spec (K=5)",
        experiment=f"scalability_{token_length}",
        config={"K": 5, "max_new_tokens": token_length},
        throughput_tps=np.mean(all_throughput),
        speedup=np.mean(all_throughput) / baseline_throughput,
        ttft_ms=np.mean(all_ttft),
        ttft_std=np.std(all_ttft),
        tpot_ms=np.mean(all_tpot),
        tpot_std=np.std(all_tpot),
        total_tokens_generated=token_length,
        acceptance_rate=np.mean(all_acceptance) if all_acceptance else 0,
        acceptance_std=np.std(all_acceptance) if all_acceptance else 0,
        throughput_std=np.std(all_throughput),
        peak_memory_mb=peak_mem
    )
    results.append(asdict(metrics))
    print(f"  Throughput: {np.mean(all_throughput):.1f} t/s, Speedup: {metrics.speedup:.2f}x")
    
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== Fixed Tree =====
    print(f"\n[3/4] Fixed Tree (D=5, B=2)...")
    generator = TreeSpeculativeGeneratorV2(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        tree_depth=5,
        branch_factor=2,
        device="cuda"
    )
    
    all_throughput, all_ttft, all_tpot = [], [], []
    all_acceptance, all_path_lengths, all_rounds = [], [], []
    
    for i, prompt in enumerate(prompts):
        if i < warmup_runs:
            generator.generate(prompt, max_new_tokens=min(50, token_length))
            continue
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = generator.generate(prompt, max_new_tokens=token_length)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
        stats = generator.get_stats()
        generated = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
        throughput = generated / total_time
        
        all_throughput.append(throughput)
        all_ttft.append(total_time / generated * 1000)
        all_tpot.append(total_time / generated * 1000)
        all_acceptance.append(stats.get('acceptance_rate', 0))
        all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
        all_rounds.append(stats.get('total_rounds', 0))
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    metrics = PaperMetrics(
        method="Fixed Tree (D=5, B=2)",
        experiment=f"scalability_{token_length}",
        config={"tree_depth": 5, "branch_factor": 2, "max_new_tokens": token_length},
        throughput_tps=np.mean(all_throughput),
        speedup=np.mean(all_throughput) / baseline_throughput,
        ttft_ms=np.mean(all_ttft),
        ttft_std=np.std(all_ttft),
        tpot_ms=np.mean(all_tpot),
        tpot_std=np.std(all_tpot),
        total_tokens_generated=token_length,
        acceptance_rate=np.mean(all_acceptance),
        acceptance_std=np.std(all_acceptance),
        avg_path_length=np.mean(all_path_lengths),
        path_length_std=np.std(all_path_lengths),
        total_rounds=int(np.mean(all_rounds)),
        throughput_std=np.std(all_throughput),
        peak_memory_mb=peak_mem
    )
    results.append(asdict(metrics))
    print(f"  Throughput: {np.mean(all_throughput):.1f} t/s, Speedup: {metrics.speedup:.2f}x")
    
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== Adaptive Tree (Phase 3) =====
    print(f"\n[4/4] Adaptive Tree (Phase 3)...")
    generator = TreeSpeculativeGeneratorV2AdaptiveV3(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        tree_depth=5,
        branch_factor=2,
        max_depth=8,
        high_conf_threshold=0.9,
        low_conf_threshold=0.4,
        min_branch=1,
        max_branch=3,
        device="cuda"
    )
    
    all_throughput, all_ttft, all_tpot = [], [], []
    all_acceptance, all_path_lengths, all_rounds = [], [], []
    all_high_conf, all_medium_conf, all_low_conf = [], [], []
    all_early_stops, all_deep_expansions = [], []
    
    for i, prompt in enumerate(prompts):
        if i < warmup_runs:
            generator.generate(prompt, max_new_tokens=min(50, token_length))
            continue
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = generator.generate(prompt, max_new_tokens=token_length)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
        stats = generator.get_stats()
        generated = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
        throughput = generated / total_time
        
        all_throughput.append(throughput)
        all_ttft.append(total_time / generated * 1000)
        all_tpot.append(total_time / generated * 1000)
        all_acceptance.append(stats.get('acceptance_rate', 0))
        all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
        all_rounds.append(stats.get('total_rounds', 0))
        all_high_conf.append(stats.get('high_conf_ratio', 0))
        all_medium_conf.append(stats.get('medium_conf_ratio', 0))
        all_low_conf.append(stats.get('low_conf_ratio', 0))
        all_early_stops.append(stats.get('early_stops', 0))
        all_deep_expansions.append(stats.get('deep_expansions', 0))
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    final_stats = generator.get_stats()
    
    metrics = PaperMetrics(
        method="Phase 3: + History Adjust",
        experiment=f"scalability_{token_length}",
        config={
            "phase": 3,
            "max_new_tokens": token_length,
            "base_depth": 5,
            "max_depth": 8,
            "high_conf_threshold": 0.9,
            "low_conf_threshold": 0.4,
            "min_branch": 1,
            "max_branch": 3
        },
        throughput_tps=np.mean(all_throughput),
        speedup=np.mean(all_throughput) / baseline_throughput,
        ttft_ms=np.mean(all_ttft),
        ttft_std=np.std(all_ttft),
        tpot_ms=np.mean(all_tpot),
        tpot_std=np.std(all_tpot),
        total_tokens_generated=token_length,
        acceptance_rate=np.mean(all_acceptance),
        acceptance_std=np.std(all_acceptance),
        avg_path_length=np.mean(all_path_lengths),
        path_length_std=np.std(all_path_lengths),
        total_rounds=int(np.mean(all_rounds)),
        high_conf_ratio=np.mean(all_high_conf),
        medium_conf_ratio=np.mean(all_medium_conf),
        low_conf_ratio=np.mean(all_low_conf),
        early_stops=int(np.mean(all_early_stops)),
        deep_expansions=int(np.mean(all_deep_expansions)),
        total_adjustments=final_stats.get('total_adjustments', 0),
        throughput_std=np.std(all_throughput),
        peak_memory_mb=peak_mem
    )
    results.append(asdict(metrics))
    print(f"  Throughput: {np.mean(all_throughput):.1f} t/s, Speedup: {metrics.speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    token_length = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    warmup_runs = int(sys.argv[3])
    max_prompt_length = int(sys.argv[4])
    output_file = sys.argv[5]
    
    results = run_experiment(token_length, num_samples, warmup_runs, max_prompt_length)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

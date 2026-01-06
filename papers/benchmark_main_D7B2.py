#!/usr/bin/env python3
"""
Main Experiment Benchmark with Best Fixed Tree Configuration

Fixed Tree Configuration: D=8, B=3, τ=0.10 (Best from hyperparameter sweep)
Linear Spec: K=8

This script runs the main comparison experiment using:
- Baseline (AR)
- Linear Spec (K=8)
- Fixed Tree (D=8, B=3, τ=0.10)  <-- Best configuration from sweep
- Adaptive Tree Phase 1, 2, 3

Usage:
    cd /mnt/disk1/ljm/LLM-Efficient-Reasoning && \
    source ../llm-inference/bin/activate && \
    CUDA_VISIBLE_DEVICES=0 python papers/benchmark_main_D7B2.py \
        --max-new-tokens 1500 \
        --output results/adaptive/main_D8B3/results.json
"""

import os
import sys
import json
import time
import gc
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Fixed Tree Configuration - Best from sweep
# =============================================================================
FIXED_TREE_DEPTH = 8
FIXED_TREE_BRANCH = 3
FIXED_TREE_THRESHOLD = 0.10  # τ = 0.10


@dataclass
class PaperMetrics:
    """Complete metrics for paper experiments."""
    method: str
    experiment: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Primary metrics
    throughput_tps: float = 0.0
    speedup: float = 1.0
    
    # Latency metrics
    ttft_ms: float = 0.0
    ttft_std: float = 0.0
    tpot_ms: float = 0.0
    tpot_std: float = 0.0
    total_time_ms: float = 0.0
    
    # Token metrics
    total_tokens_generated: int = 0
    tokens_per_round: float = 0.0
    
    # Acceptance metrics
    acceptance_rate: float = 0.0
    avg_path_length: float = 0.0
    total_rounds: int = 0
    
    # Adaptive stats
    high_conf_ratio: float = 0.0
    medium_conf_ratio: float = 0.0
    low_conf_ratio: float = 0.0
    early_stops: int = 0
    deep_expansions: int = 0
    total_adjustments: int = 0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    
    # Variance
    throughput_std: float = 0.0
    acceptance_std: float = 0.0
    path_length_std: float = 0.0


class MainBenchmark:
    """Main experiment benchmark with D=7, B=2, τ=0.10 Fixed Tree."""
    
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        device: str = "cuda",
        num_samples: int = 10,
        warmup_runs: int = 2,
        min_prompt_length: int = 200,
        max_prompt_length: int = 800,
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.num_samples = num_samples
        self.warmup_runs = warmup_runs
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        
        print("="*80)
        print("MAIN EXPERIMENT BENCHMARK")
        print(f"Fixed Tree Configuration: D={FIXED_TREE_DEPTH}, B={FIXED_TREE_BRANCH}, τ={FIXED_TREE_THRESHOLD}")
        print("="*80)
        
        print("\nLoading models...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=torch.float16,
        ).to(device)
        self.target_model.eval()
        
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=torch.float16,
        ).to(device)
        self.draft_model.eval()
        
        self.prompts = self._load_wikitext_prompts()
        print(f"Loaded {len(self.prompts)} prompts from WikiText-2 dataset")
        
        self.all_results: List[PaperMetrics] = []
        self.baseline_throughput: Optional[float] = None
    
    def _load_wikitext_prompts(self) -> List[str]:
        """Load prompts from WikiText dataset via ModelScope."""
        print("Loading WikiText-2 dataset from ModelScope...")
        
        try:
            from modelscope.msdatasets import MsDataset
            
            dataset = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='validation')
            
            prompts = []
            current_text = ""
            
            for item in dataset:
                text = item.get('text', '')
                if not text or text.strip() == '' or text.startswith(' ='):
                    if current_text and len(current_text) >= self.min_prompt_length:
                        prompts.append(current_text.strip())
                        current_text = ""
                        if len(prompts) >= self.num_samples * 3:
                            break
                else:
                    current_text += " " + text
            
            if current_text and len(current_text) >= self.min_prompt_length:
                prompts.append(current_text.strip())
            
            valid_prompts = []
            for prompt in prompts:
                prompt = prompt.strip()
                if len(prompt) >= self.min_prompt_length:
                    if len(prompt) > self.max_prompt_length:
                        prompt = prompt[:self.max_prompt_length]
                    valid_prompts.append(prompt)
                    if len(valid_prompts) >= self.num_samples:
                        break
            
            return valid_prompts[:self.num_samples]
            
        except Exception as e:
            print(f"Warning: Failed to load WikiText dataset: {e}")
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> List[str]:
        """Fallback prompts if WikiText loading fails."""
        fallback = [
            """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.""",
            """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.""",
            """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.""",
            """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.""",
            """The transformer is a deep learning model introduced in 2017 that utilizes the mechanism of self-attention.""",
            """Large language models are neural network models that are trained on massive amounts of text data.""",
            """Speculative decoding is an inference optimization technique for large language models.""",
            """The attention mechanism allows neural networks to focus on specific parts of the input.""",
            """Transfer learning focuses on storing knowledge gained while solving one problem and applying it to a different problem.""",
            """Reinforcement learning from human feedback trains models to align with human preferences.""",
        ]
        return fallback[:self.num_samples]
    
    @contextmanager
    def _memory_tracking(self):
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            pass
    
    def _cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _disable_eos(self):
        original_eos = self.tokenizer.eos_token_id
        self.tokenizer.eos_token_id = 999999
        return original_eos
    
    def _restore_eos(self, original_eos):
        self.tokenizer.eos_token_id = original_eos
    
    @torch.inference_mode()
    def benchmark_baseline(self, max_new_tokens: int) -> PaperMetrics:
        """Benchmark pure autoregressive generation."""
        method_name = "Baseline (AR)"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        metrics = PaperMetrics(
            method=method_name,
            experiment="main",
            config={"max_new_tokens": max_new_tokens}
        )
        
        original_eos = self._disable_eos()
        
        all_ttft, all_tpot, all_throughput = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_prefill = time.perf_counter()
                outputs = self.target_model(input_ids=input_ids, use_cache=True, return_dict=True)
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - start_prefill) * 1000
                
                past_kv = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                torch.cuda.synchronize()
                start_decode = time.perf_counter()
                
                for _ in range(max_new_tokens):
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    outputs = self.target_model(input_ids=next_token, past_key_values=past_kv, use_cache=True, return_dict=True)
                    past_kv = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                
                torch.cuda.synchronize()
                decode_time = (time.perf_counter() - start_decode) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            total_time = ttft + decode_time
            tpot = decode_time / max_new_tokens
            throughput = max_new_tokens / (total_time / 1000)
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.total_tokens_generated = max_new_tokens
        metrics.total_rounds = max_new_tokens
        metrics.tokens_per_round = 1.0
        metrics.peak_memory_mb = peak_memory
        metrics.speedup = 1.0
        
        self.baseline_throughput = metrics.throughput_tps
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_linear_spec(self, max_new_tokens: int, K: int = 5) -> PaperMetrics:
        """Benchmark linear speculative decoding."""
        method_name = f"Linear Spec (K={K})"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        from spec_decode.core.speculative_generator import SpeculativeGenerator
        
        metrics = PaperMetrics(
            method=method_name,
            experiment="main",
            config={"max_new_tokens": max_new_tokens, "K": K}
        )
        
        original_eos = self._disable_eos()
        
        generator = SpeculativeGenerator(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            K=K,
            device=self.device,
            use_compile=False
        )
        
        all_ttft, all_tpot, all_throughput, all_acceptance, all_path_lengths, all_rounds = [], [], [], [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
            torch.cuda.synchronize()
            start_ttft = time.perf_counter()
            _ = self.target_model(input_ids=input_ids, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - start_ttft) * 1000
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            acceptance_rate = stats['total_accepted'] / stats['total_draft_tokens'] if stats['total_draft_tokens'] > 0 else 0
            avg_accepted = stats['total_accepted'] / stats['total_rounds'] if stats['total_rounds'] > 0 else 0
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                all_acceptance.append(acceptance_rate)
                all_path_lengths.append(avg_accepted)
                all_rounds.append(stats.get('total_rounds', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, accept={acceptance_rate:.1%}")
        
        self._restore_eos(original_eos)
        
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.acceptance_rate = np.mean(all_acceptance)
        metrics.acceptance_std = np.std(all_acceptance)
        metrics.avg_path_length = np.mean(all_path_lengths)
        metrics.path_length_std = np.std(all_path_lengths)
        metrics.total_rounds = int(np.mean(all_rounds))
        metrics.tokens_per_round = max_new_tokens / metrics.total_rounds if metrics.total_rounds > 0 else 0
        metrics.total_tokens_generated = max_new_tokens
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, Speedup={metrics.speedup:.2f}x")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_fixed_tree(self, max_new_tokens: int) -> PaperMetrics:
        """Benchmark fixed tree with D=7, B=2, τ=0.10."""
        method_name = f"Fixed Tree (D={FIXED_TREE_DEPTH}, B={FIXED_TREE_BRANCH}, τ={FIXED_TREE_THRESHOLD})"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
        
        metrics = PaperMetrics(
            method=method_name,
            experiment="main",
            config={
                "max_new_tokens": max_new_tokens,
                "tree_depth": FIXED_TREE_DEPTH,
                "branch_factor": FIXED_TREE_BRANCH,
                "probability_threshold": FIXED_TREE_THRESHOLD,
                "max_tree_nodes": 256,
            }
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=FIXED_TREE_DEPTH,
            branch_factor=FIXED_TREE_BRANCH,
            probability_threshold=FIXED_TREE_THRESHOLD,  # τ = 0.10
            max_tree_nodes=256,
            device=self.device,
            use_compile=False
        )
        
        all_ttft, all_tpot, all_throughput, all_acceptance, all_path_lengths, all_rounds = [], [], [], [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
            torch.cuda.synchronize()
            start_ttft = time.perf_counter()
            _ = self.target_model(input_ids=input_ids, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - start_ttft) * 1000
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, "
                      f"accept={stats.get('acceptance_rate', 0):.1%}, path={stats.get('avg_accepted_path_length', 0):.2f}")
        
        self._restore_eos(original_eos)
        
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.acceptance_rate = np.mean(all_acceptance)
        metrics.acceptance_std = np.std(all_acceptance)
        metrics.avg_path_length = np.mean(all_path_lengths)
        metrics.path_length_std = np.std(all_path_lengths)
        metrics.total_rounds = int(np.mean(all_rounds))
        metrics.tokens_per_round = max_new_tokens / metrics.total_rounds if metrics.total_rounds > 0 else 0
        metrics.total_tokens_generated = max_new_tokens
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, Speedup={metrics.speedup:.2f}x")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_adaptive(self, max_new_tokens: int, phase: int) -> PaperMetrics:
        """Benchmark adaptive tree (Phase 1/2/3)."""
        phase_names = {1: "Adaptive Branch", 2: "+ Dynamic Depth", 3: "+ History Adjust"}
        method_name = f"Phase {phase}: {phase_names.get(phase, 'Unknown')}"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        from spec_decode.core.tree_speculative_generator_adaptive import (
            TreeSpeculativeGeneratorV2Adaptive,
            TreeSpeculativeGeneratorV2AdaptiveV2,
            TreeSpeculativeGeneratorV2AdaptiveV3
        )
        
        # Adaptive parameters
        base_depth = 5
        max_depth = 8
        high_conf_threshold = 0.9
        low_conf_threshold = 0.4
        min_branch = 1
        max_branch = 3
        
        metrics = PaperMetrics(
            method=method_name,
            experiment="main",
            config={
                "phase": phase,
                "max_new_tokens": max_new_tokens,
                "base_depth": base_depth,
                "max_depth": max_depth,
                "high_conf_threshold": high_conf_threshold,
                "low_conf_threshold": low_conf_threshold,
                "min_branch": min_branch,
                "max_branch": max_branch,
            }
        )
        
        original_eos = self._disable_eos()
        
        if phase == 1:
            generator = TreeSpeculativeGeneratorV2Adaptive(
                target_model=self.target_model, draft_model=self.draft_model, tokenizer=self.tokenizer,
                tree_depth=base_depth, branch_factor=2, probability_threshold=0.05, max_tree_nodes=256,
                device=self.device, use_compile=False,
                high_conf_threshold=high_conf_threshold, low_conf_threshold=low_conf_threshold,
                min_branch=min_branch, max_branch=max_branch,
            )
        elif phase == 2:
            generator = TreeSpeculativeGeneratorV2AdaptiveV2(
                target_model=self.target_model, draft_model=self.draft_model, tokenizer=self.tokenizer,
                tree_depth=base_depth, branch_factor=2, probability_threshold=0.05, max_tree_nodes=256,
                device=self.device, use_compile=False,
                high_conf_threshold=high_conf_threshold, low_conf_threshold=low_conf_threshold,
                min_branch=min_branch, max_branch=max_branch,
                early_stop_threshold=0.1, deep_expand_threshold=0.5, max_depth=max_depth,
            )
        else:
            generator = TreeSpeculativeGeneratorV2AdaptiveV3(
                target_model=self.target_model, draft_model=self.draft_model, tokenizer=self.tokenizer,
                tree_depth=base_depth, branch_factor=2, probability_threshold=0.05, max_tree_nodes=256,
                device=self.device, use_compile=False,
                high_conf_threshold=high_conf_threshold, low_conf_threshold=low_conf_threshold,
                min_branch=min_branch, max_branch=max_branch,
                early_stop_threshold=0.1, deep_expand_threshold=0.5, max_depth=max_depth,
                history_window=10, target_acceptance_rate=0.7, adjustment_rate=0.05, enable_auto_adjust=True,
            )
        
        all_ttft, all_tpot, all_throughput = [], [], []
        all_acceptance, all_path_lengths, all_rounds = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
            torch.cuda.synchronize()
            start_ttft = time.perf_counter()
            _ = self.target_model(input_ids=input_ids, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - start_ttft) * 1000
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, "
                      f"accept={stats.get('acceptance_rate', 0):.1%}, path={stats.get('avg_accepted_path_length', 0):.2f}")
        
        self._restore_eos(original_eos)
        
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.acceptance_rate = np.mean(all_acceptance)
        metrics.acceptance_std = np.std(all_acceptance)
        metrics.avg_path_length = np.mean(all_path_lengths)
        metrics.path_length_std = np.std(all_path_lengths)
        metrics.total_rounds = int(np.mean(all_rounds))
        metrics.tokens_per_round = max_new_tokens / metrics.total_rounds if metrics.total_rounds > 0 else 0
        metrics.total_tokens_generated = max_new_tokens
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, Speedup={metrics.speedup:.2f}x")
        
        self.all_results.append(metrics)
        return metrics
    
    def run_main_experiment(self, max_new_tokens: int = 1500):
        """Run main comparison experiment."""
        print("\n" + "#"*80)
        print("# MAIN EXPERIMENT: D=8, B=3, τ=0.10 Fixed Tree (Best Config)")
        print("#"*80)
        
        self.benchmark_baseline(max_new_tokens)
        self.benchmark_linear_spec(max_new_tokens, K=8)
        self.benchmark_fixed_tree(max_new_tokens)
        self.benchmark_adaptive(max_new_tokens, phase=1)
        self.benchmark_adaptive(max_new_tokens, phase=2)
        self.benchmark_adaptive(max_new_tokens, phase=3)
    
    def print_summary(self):
        """Print results summary."""
        print("\n" + "="*100)
        print("RESULTS SUMMARY")
        print("="*100)
        print(f"\nFixed Tree Configuration: D={FIXED_TREE_DEPTH}, B={FIXED_TREE_BRANCH}, τ={FIXED_TREE_THRESHOLD}")
        print("-"*100)
        print(f"{'Method':<40} {'Throughput':<15} {'Speedup':<10} {'Accept':<10} {'PathLen':<10}")
        print("-"*100)
        
        for r in self.all_results:
            throughput_str = f"{r.throughput_tps:.1f}±{r.throughput_std:.1f}"
            accept_str = f"{r.acceptance_rate:.1%}" if r.acceptance_rate > 0 else "N/A"
            path_str = f"{r.avg_path_length:.2f}" if r.avg_path_length > 0 else "N/A"
            print(f"{r.method:<40} {throughput_str:<15} {r.speedup:<10.2f}x {accept_str:<10} {path_str:<10}")
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        summary = {
            "experiment_info": {
                "study": "main_experiment_D8B3",
                "fixed_tree_config": {
                    "depth": FIXED_TREE_DEPTH,
                    "branch": FIXED_TREE_BRANCH,
                    "threshold": FIXED_TREE_THRESHOLD,
                },
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "dataset": "WikiText-2",
                "num_samples": self.num_samples,
                "warmup_runs": self.warmup_runs,
            },
            "all_results": [asdict(r) for r in self.all_results],
            "timestamp": datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Main Experiment with D=7, B=2, τ=0.10 Fixed Tree")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--output", type=str, default="results/adaptive/main_D8B3/results.json")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=800)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    
    args = parser.parse_args()
    
    benchmark = MainBenchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        num_samples=args.num_samples,
        warmup_runs=args.warmup_runs,
        max_prompt_length=args.max_prompt_length,
    )
    
    benchmark.run_main_experiment(max_new_tokens=args.max_new_tokens)
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()


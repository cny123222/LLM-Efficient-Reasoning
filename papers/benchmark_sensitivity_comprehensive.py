#!/usr/bin/env python3
"""
Comprehensive Sensitivity Analysis for Adaptive Tree Speculative Decoding

This script performs a thorough parameter sensitivity study with:
1. Threshold sensitivity: Different high/low threshold combinations with varying gaps
2. Branch factor sensitivity: Different min/max branch combinations
3. Depth sensitivity: Different base_depth and max_depth combinations
4. Cross-parameter analysis: Key combinations of the above

Output tokens: 1500
Input prompts length: 800
"""

import os
import sys
import json
import time
import gc
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SensitivityMetrics:
    """Metrics for sensitivity analysis."""
    # Config identification
    category: str  # threshold, branch, depth, cross
    config_name: str
    config_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    throughput_tps: float = 0.0
    throughput_std: float = 0.0
    speedup: float = 1.0
    
    # Latency
    ttft_ms: float = 0.0
    ttft_std: float = 0.0
    tpot_ms: float = 0.0
    tpot_std: float = 0.0
    
    # Acceptance metrics
    acceptance_rate: float = 0.0
    acceptance_std: float = 0.0
    avg_path_length: float = 0.0
    path_length_std: float = 0.0
    total_rounds: int = 0
    tokens_per_round: float = 0.0
    
    # Confidence distribution
    high_conf_ratio: float = 0.0
    medium_conf_ratio: float = 0.0
    low_conf_ratio: float = 0.0
    
    # Resource
    peak_memory_mb: float = 0.0


class ComprehensiveSensitivityBenchmark:
    """Comprehensive sensitivity analysis benchmark."""
    
    # =========================================================================
    # Configuration Definitions
    # =========================================================================
    
    # 1. Threshold configurations - varying gap sizes and positions
    THRESHOLD_CONFIGS = [
        # Format: (high_conf, low_conf, description)
        # Small gap (0.2) - narrow medium zone
        (0.7, 0.5, "gap=0.2, center=0.6"),
        (0.8, 0.6, "gap=0.2, center=0.7"),
        (0.9, 0.7, "gap=0.2, center=0.8"),
        
        # Medium gap (0.3) - balanced medium zone
        (0.7, 0.4, "gap=0.3, center=0.55"),
        (0.8, 0.5, "gap=0.3, center=0.65"),
        (0.9, 0.6, "gap=0.3, center=0.75"),
        
        # Large gap (0.4) - wide medium zone
        (0.7, 0.3, "gap=0.4, center=0.5"),
        (0.8, 0.4, "gap=0.4, center=0.6"),
        (0.9, 0.5, "gap=0.4, center=0.7"),
        
        # Very large gap (0.5) - very wide medium zone
        (0.8, 0.3, "gap=0.5, center=0.55"),
        (0.9, 0.4, "gap=0.5, center=0.65"),
        
        # Extreme configurations
        (0.95, 0.3, "high_only=aggressive"),  # Very high threshold
        (0.6, 0.2, "low_only=aggressive"),    # Very low threshold
    ]
    
    # 2. Branch factor configurations
    BRANCH_CONFIGS = [
        # Format: (min_branch, max_branch, description)
        (1, 2, "narrow_range"),
        (1, 3, "default_range"),
        (1, 4, "wide_range"),
        (1, 5, "very_wide_range"),
        (2, 3, "higher_min"),
        (2, 4, "higher_min_wide"),
        (2, 5, "higher_min_very_wide"),
        (3, 5, "high_branch_range"),
    ]
    
    # 3. Depth configurations
    DEPTH_CONFIGS = [
        # Format: (base_depth, max_depth, description)
        (4, 6, "shallow"),
        (5, 7, "medium_shallow"),
        (5, 8, "medium"),
        (6, 8, "medium_deep"),
        (6, 10, "deep"),
        (7, 10, "very_deep"),
        (8, 12, "ultra_deep"),
    ]
    
    # 4. Cross-parameter configurations (key combinations)
    CROSS_CONFIGS = [
        # Format: (high_conf, low_conf, min_branch, max_branch, base_depth, max_depth, description)
        # Conservative (high confidence, narrow branches, shallow depth)
        (0.9, 0.6, 1, 2, 5, 7, "conservative"),
        # Balanced
        (0.8, 0.4, 1, 3, 5, 8, "balanced"),
        # Aggressive (lower confidence, wide branches, deep)
        (0.7, 0.3, 2, 4, 6, 10, "aggressive"),
        # High throughput focus
        (0.85, 0.5, 1, 3, 6, 9, "high_throughput"),
        # High acceptance focus  
        (0.75, 0.35, 2, 4, 5, 8, "high_acceptance"),
    ]
    
    def __init__(
        self,
        target_model_path: str = "/mnt/disk1/models/pythia-2.8b",
        draft_model_path: str = "/mnt/disk1/models/pythia-70m",
        device: str = "cuda",
        num_samples: int = 10,
        warmup_runs: int = 2,
        max_prompt_length: int = 800,
        max_new_tokens: int = 1500,
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.num_samples = num_samples
        self.warmup_runs = warmup_runs
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        
        print("="*80)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS")
        print(f"Output tokens: {max_new_tokens}, Max prompt length: {max_prompt_length}")
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
        print(f"Loaded {len(self.prompts)} prompts")
        
        self.all_results: List[SensitivityMetrics] = []
        self.baseline_throughput: float = None
    
    def _load_wikitext_prompts(self) -> List[str]:
        """Load prompts from WikiText dataset."""
        print("Loading WikiText-2 dataset...")
        try:
            from modelscope.msdatasets import MsDataset
            dataset = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='validation')
            
            prompts = []
            current_text = ""
            min_length = 200
            
            for item in dataset:
                text = item.get('text', '')
                if not text or text.strip() == '' or text.startswith(' ='):
                    if current_text and len(current_text) >= min_length:
                        prompts.append(current_text.strip())
                        current_text = ""
                        if len(prompts) >= self.num_samples * 3:
                            break
                else:
                    current_text += " " + text
            
            valid_prompts = []
            for prompt in prompts:
                prompt = prompt.strip()
                if len(prompt) >= min_length:
                    if len(prompt) > self.max_prompt_length:
                        prompt = prompt[:self.max_prompt_length]
                    valid_prompts.append(prompt)
                    if len(valid_prompts) >= self.num_samples:
                        break
            
            return valid_prompts[:self.num_samples]
        except Exception as e:
            print(f"Warning: Failed to load WikiText: {e}")
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> List[str]:
        """Fallback prompts."""
        return [
            "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.",
        ] * self.num_samples
    
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
    
    # =========================================================================
    # Benchmark Methods
    # =========================================================================
    
    @torch.inference_mode()
    def benchmark_baseline(self) -> SensitivityMetrics:
        """Benchmark baseline autoregressive generation."""
        print(f"\n{'='*70}")
        print("Benchmarking: Baseline (AR)")
        print(f"{'='*70}")
        
        metrics = SensitivityMetrics(
            category="baseline",
            config_name="Baseline (AR)",
            config_params={"max_new_tokens": self.max_new_tokens}
        )
        
        original_eos = self._disable_eos()
        all_throughput, all_ttft, all_tpot = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt = self.prompts[run_idx % len(self.prompts)]
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
                
                for _ in range(self.max_new_tokens):
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    outputs = self.target_model(input_ids=next_token, past_key_values=past_kv, use_cache=True, return_dict=True)
                    past_kv = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                
                torch.cuda.synchronize()
                decode_time = (time.perf_counter() - start_decode) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            total_time = ttft + decode_time
            throughput = self.max_new_tokens / (total_time / 1000)
            tpot = decode_time / self.max_new_tokens
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.total_rounds = self.max_new_tokens
        metrics.tokens_per_round = 1.0
        metrics.peak_memory_mb = peak_memory
        metrics.speedup = 1.0
        
        self.baseline_throughput = metrics.throughput_tps
        
        print(f"\nBaseline: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_fixed_tree(self, tree_depth: int = 5, branch_factor: int = 2) -> SensitivityMetrics:
        """Benchmark fixed tree as reference."""
        config_name = f"Fixed Tree (D={tree_depth}, B={branch_factor})"
        print(f"\n{'='*70}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*70}")
        
        from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
        
        metrics = SensitivityMetrics(
            category="baseline",
            config_name=config_name,
            config_params={
                "tree_depth": tree_depth,
                "branch_factor": branch_factor,
            }
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            probability_threshold=0.05,
            max_tree_nodes=256,
            device=self.device,
            use_compile=False
        )
        
        all_throughput, all_ttft, all_tpot = [], [], []
        all_acceptance, all_path_lengths, all_rounds = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt = self.prompts[run_idx % len(self.prompts)]
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
                _ = generator.generate(prompt, max_new_tokens=self.max_new_tokens)
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, "
                      f"accept={stats.get('acceptance_rate', 0):.1%}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.acceptance_rate = np.mean(all_acceptance)
        metrics.acceptance_std = np.std(all_acceptance)
        metrics.avg_path_length = np.mean(all_path_lengths)
        metrics.path_length_std = np.std(all_path_lengths)
        metrics.total_rounds = int(np.mean(all_rounds))
        metrics.tokens_per_round = self.max_new_tokens / metrics.total_rounds if metrics.total_rounds > 0 else 0
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, "
              f"Speedup={metrics.speedup:.2f}x")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_adaptive(
        self,
        category: str,
        config_name: str,
        high_conf_threshold: float = 0.8,
        low_conf_threshold: float = 0.4,
        min_branch: int = 1,
        max_branch: int = 3,
        base_depth: int = 5,
        max_depth: int = 8,
    ) -> SensitivityMetrics:
        """Benchmark adaptive tree with specific configuration."""
        print(f"\n{'='*70}")
        print(f"Benchmarking: {config_name}")
        print(f"  high_conf={high_conf_threshold}, low_conf={low_conf_threshold}")
        print(f"  branch=[{min_branch}, {max_branch}], depth=[{base_depth}, {max_depth}]")
        print(f"{'='*70}")
        
        from spec_decode.core.tree_speculative_generator_adaptive import TreeSpeculativeGeneratorV2AdaptiveV3
        
        metrics = SensitivityMetrics(
            category=category,
            config_name=config_name,
            config_params={
                "high_conf_threshold": high_conf_threshold,
                "low_conf_threshold": low_conf_threshold,
                "min_branch": min_branch,
                "max_branch": max_branch,
                "base_depth": base_depth,
                "max_depth": max_depth,
            }
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2AdaptiveV3(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=base_depth,
            branch_factor=2,
            probability_threshold=0.05,
            max_tree_nodes=256,
            device=self.device,
            use_compile=False,
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold,
            min_branch=min_branch,
            max_branch=max_branch,
            early_stop_threshold=0.1,
            deep_expand_threshold=0.5,
            max_depth=max_depth,
            history_window=10,
            target_acceptance_rate=0.7,
            adjustment_rate=0.05,
            enable_auto_adjust=True,
        )
        
        all_throughput, all_ttft, all_tpot = [], [], []
        all_acceptance, all_path_lengths, all_rounds = [], [], []
        all_high_conf, all_medium_conf, all_low_conf = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt = self.prompts[run_idx % len(self.prompts)]
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
                _ = generator.generate(prompt, max_new_tokens=self.max_new_tokens)
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                
                adaptive_stats = stats.get('adaptive_stats', {})
                all_high_conf.append(adaptive_stats.get('high_conf_ratio', 0))
                all_medium_conf.append(adaptive_stats.get('medium_conf_ratio', 0))
                all_low_conf.append(adaptive_stats.get('low_conf_ratio', 0))
                
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, "
                      f"accept={stats.get('acceptance_rate', 0):.1%}, "
                      f"path={stats.get('avg_accepted_path_length', 0):.2f}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.acceptance_rate = np.mean(all_acceptance)
        metrics.acceptance_std = np.std(all_acceptance)
        metrics.avg_path_length = np.mean(all_path_lengths)
        metrics.path_length_std = np.std(all_path_lengths)
        metrics.total_rounds = int(np.mean(all_rounds))
        metrics.tokens_per_round = self.max_new_tokens / metrics.total_rounds if metrics.total_rounds > 0 else 0
        metrics.high_conf_ratio = np.mean(all_high_conf) if all_high_conf else 0
        metrics.medium_conf_ratio = np.mean(all_medium_conf) if all_medium_conf else 0
        metrics.low_conf_ratio = np.mean(all_low_conf) if all_low_conf else 0
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, "
              f"Speedup={metrics.speedup:.2f}x, Accept={metrics.acceptance_rate:.1%}")
        
        self.all_results.append(metrics)
        return metrics
    
    # =========================================================================
    # Experiment Suites
    # =========================================================================
    
    def run_baseline_experiments(self):
        """Run baseline methods for reference."""
        print("\n" + "#"*80)
        print("# BASELINE METHODS")
        print("#"*80)
        
        self.benchmark_baseline()
        self.benchmark_fixed_tree(tree_depth=5, branch_factor=2)
    
    def run_threshold_sensitivity(self):
        """Sensitivity to confidence thresholds."""
        print("\n" + "#"*80)
        print("# THRESHOLD SENSITIVITY ANALYSIS")
        print(f"# Testing {len(self.THRESHOLD_CONFIGS)} configurations")
        print("#"*80)
        
        for i, (high_conf, low_conf, desc) in enumerate(self.THRESHOLD_CONFIGS):
            gap = high_conf - low_conf
            config_name = f"thresh_h{high_conf}_l{low_conf}"
            print(f"\n[{i+1}/{len(self.THRESHOLD_CONFIGS)}] {desc}")
            
            self.benchmark_adaptive(
                category="threshold",
                config_name=config_name,
                high_conf_threshold=high_conf,
                low_conf_threshold=low_conf,
                # Use default values for other params
                min_branch=1,
                max_branch=3,
                base_depth=5,
                max_depth=8,
            )
    
    def run_branch_sensitivity(self):
        """Sensitivity to branch factor range."""
        print("\n" + "#"*80)
        print("# BRANCH FACTOR SENSITIVITY ANALYSIS")
        print(f"# Testing {len(self.BRANCH_CONFIGS)} configurations")
        print("#"*80)
        
        for i, (min_b, max_b, desc) in enumerate(self.BRANCH_CONFIGS):
            config_name = f"branch_{min_b}_{max_b}"
            print(f"\n[{i+1}/{len(self.BRANCH_CONFIGS)}] {desc}")
            
            self.benchmark_adaptive(
                category="branch",
                config_name=config_name,
                min_branch=min_b,
                max_branch=max_b,
                # Use default values for other params
                high_conf_threshold=0.8,
                low_conf_threshold=0.4,
                base_depth=5,
                max_depth=8,
            )
    
    def run_depth_sensitivity(self):
        """Sensitivity to tree depth."""
        print("\n" + "#"*80)
        print("# DEPTH SENSITIVITY ANALYSIS")
        print(f"# Testing {len(self.DEPTH_CONFIGS)} configurations")
        print("#"*80)
        
        for i, (base_d, max_d, desc) in enumerate(self.DEPTH_CONFIGS):
            config_name = f"depth_{base_d}_{max_d}"
            print(f"\n[{i+1}/{len(self.DEPTH_CONFIGS)}] {desc}")
            
            self.benchmark_adaptive(
                category="depth",
                config_name=config_name,
                base_depth=base_d,
                max_depth=max_d,
                # Use default values for other params
                high_conf_threshold=0.8,
                low_conf_threshold=0.4,
                min_branch=1,
                max_branch=3,
            )
    
    def run_cross_parameter_analysis(self):
        """Cross-parameter analysis with key combinations."""
        print("\n" + "#"*80)
        print("# CROSS-PARAMETER ANALYSIS")
        print(f"# Testing {len(self.CROSS_CONFIGS)} combinations")
        print("#"*80)
        
        for i, (high_conf, low_conf, min_b, max_b, base_d, max_d, desc) in enumerate(self.CROSS_CONFIGS):
            config_name = f"cross_{desc}"
            print(f"\n[{i+1}/{len(self.CROSS_CONFIGS)}] {desc}")
            
            self.benchmark_adaptive(
                category="cross",
                config_name=config_name,
                high_conf_threshold=high_conf,
                low_conf_threshold=low_conf,
                min_branch=min_b,
                max_branch=max_b,
                base_depth=base_d,
                max_depth=max_d,
            )
    
    def run_all(self):
        """Run complete sensitivity analysis."""
        total_configs = (
            2 +  # baselines
            len(self.THRESHOLD_CONFIGS) +
            len(self.BRANCH_CONFIGS) +
            len(self.DEPTH_CONFIGS) +
            len(self.CROSS_CONFIGS)
        )
        
        print("\n" + "="*80)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS")
        print(f"Total configurations to test: {total_configs}")
        print(f"  - Baseline methods: 2")
        print(f"  - Threshold configs: {len(self.THRESHOLD_CONFIGS)}")
        print(f"  - Branch configs: {len(self.BRANCH_CONFIGS)}")
        print(f"  - Depth configs: {len(self.DEPTH_CONFIGS)}")
        print(f"  - Cross-parameter configs: {len(self.CROSS_CONFIGS)}")
        print("="*80)
        
        self.run_baseline_experiments()
        self.run_threshold_sensitivity()
        self.run_branch_sensitivity()
        self.run_depth_sensitivity()
        self.run_cross_parameter_analysis()
    
    # =========================================================================
    # Results Export
    # =========================================================================
    
    def print_summary(self):
        """Print summary of results."""
        print("\n" + "="*120)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS SUMMARY")
        print("="*120)
        
        # Group by category
        categories = {}
        for r in self.all_results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)
        
        for cat_name, results in categories.items():
            print(f"\n{'─'*100}")
            print(f"Category: {cat_name.upper()} ({len(results)} configs)")
            print(f"{'─'*100}")
            print(f"{'Config':<30} {'Throughput':<15} {'Speedup':<10} {'Accept':<10} {'PathLen':<10} {'HighConf':<10}")
            print("-"*100)
            
            # Sort by throughput
            sorted_results = sorted(results, key=lambda x: x.throughput_tps, reverse=True)
            
            for r in sorted_results:
                throughput_str = f"{r.throughput_tps:.1f}±{r.throughput_std:.1f}"
                accept_str = f"{r.acceptance_rate:.1%}" if r.acceptance_rate > 0 else "N/A"
                path_str = f"{r.avg_path_length:.2f}" if r.avg_path_length > 0 else "N/A"
                high_conf_str = f"{r.high_conf_ratio:.1%}" if r.high_conf_ratio > 0 else "N/A"
                
                print(f"{r.config_name:<30} {throughput_str:<15} {r.speedup:<10.2f}x {accept_str:<10} {path_str:<10} {high_conf_str:<10}")
        
        # Find best configurations
        print("\n" + "="*80)
        print("BEST CONFIGURATIONS")
        print("="*80)
        
        adaptive_results = [r for r in self.all_results if r.category != "baseline"]
        if adaptive_results:
            best_throughput = max(adaptive_results, key=lambda x: x.throughput_tps)
            best_acceptance = max(adaptive_results, key=lambda x: x.acceptance_rate)
            best_path = max(adaptive_results, key=lambda x: x.avg_path_length)
            
            print(f"\nBest Throughput: {best_throughput.config_name}")
            print(f"  {best_throughput.throughput_tps:.1f} t/s, {best_throughput.speedup:.2f}x speedup")
            print(f"  Params: {best_throughput.config_params}")
            
            print(f"\nBest Acceptance Rate: {best_acceptance.config_name}")
            print(f"  {best_acceptance.acceptance_rate:.1%} acceptance, {best_acceptance.throughput_tps:.1f} t/s")
            print(f"  Params: {best_acceptance.config_params}")
            
            print(f"\nBest Path Length: {best_path.config_name}")
            print(f"  {best_path.avg_path_length:.2f} avg path, {best_path.throughput_tps:.1f} t/s")
            print(f"  Params: {best_path.config_params}")
    
    def save_results(self, output_path: str):
        """Save results to JSON and CSV."""
        summary = {
            "experiment_info": {
                "study": "comprehensive_sensitivity_analysis",
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "max_new_tokens": self.max_new_tokens,
                "max_prompt_length": self.max_prompt_length,
                "num_samples": self.num_samples,
                "warmup_runs": self.warmup_runs,
                "total_configs": len(self.all_results),
            },
            "config_definitions": {
                "threshold_configs": self.THRESHOLD_CONFIGS,
                "branch_configs": self.BRANCH_CONFIGS,
                "depth_configs": self.DEPTH_CONFIGS,
                "cross_configs": self.CROSS_CONFIGS,
            },
            "all_results": [asdict(r) for r in self.all_results],
            "timestamp": datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {output_path}")
        
        # Save CSV
        csv_path = output_path.replace('.json', '.csv')
        self._save_csv(csv_path)
    
    def _save_csv(self, output_path: str):
        """Save to CSV."""
        import csv
        
        fieldnames = [
            'category', 'config_name', 
            'throughput_tps', 'throughput_std', 'speedup',
            'ttft_ms', 'ttft_std', 'tpot_ms', 'tpot_std',
            'acceptance_rate', 'acceptance_std', 
            'avg_path_length', 'path_length_std',
            'total_rounds', 'tokens_per_round',
            'high_conf_ratio', 'medium_conf_ratio', 'low_conf_ratio',
            'peak_memory_mb',
            'high_conf_threshold', 'low_conf_threshold',
            'min_branch', 'max_branch', 'base_depth', 'max_depth',
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in self.all_results:
                row = {k: getattr(r, k, '') for k in fieldnames if hasattr(r, k)}
                # Add config params
                for k in ['high_conf_threshold', 'low_conf_threshold', 'min_branch', 'max_branch', 'base_depth', 'max_depth']:
                    row[k] = r.config_params.get(k, '')
                writer.writerow(row)
        
        print(f"CSV saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Sensitivity Analysis")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--output", type=str, default="results/adaptive/sensitivity/comprehensive_sensitivity.json")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=800)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--category", type=str, default="all",
                       choices=["all", "baseline", "threshold", "branch", "depth", "cross"],
                       help="Which category to run")
    
    args = parser.parse_args()
    
    benchmark = ComprehensiveSensitivityBenchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        num_samples=args.num_samples,
        warmup_runs=args.warmup_runs,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
    )
    
    if args.category == "all":
        benchmark.run_all()
    elif args.category == "baseline":
        benchmark.run_baseline_experiments()
    elif args.category == "threshold":
        benchmark.run_baseline_experiments()
        benchmark.run_threshold_sensitivity()
    elif args.category == "branch":
        benchmark.run_baseline_experiments()
        benchmark.run_branch_sensitivity()
    elif args.category == "depth":
        benchmark.run_baseline_experiments()
        benchmark.run_depth_sensitivity()
    elif args.category == "cross":
        benchmark.run_baseline_experiments()
        benchmark.run_cross_parameter_analysis()
    
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()


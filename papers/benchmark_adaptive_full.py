#!/usr/bin/env python3
"""
Benchmark: Full Adaptive Tree Speculative Decoding Comparison

This script provides a comprehensive comparison of all adaptive tree methods:
- Baseline: Autoregressive generation
- Fixed: TreeSpeculativeGeneratorV2 (fixed branch factor)
- Phase 1: TreeSpeculativeGeneratorV2Adaptive (confidence-based branching)
- Phase 2: TreeSpeculativeGeneratorV2AdaptiveV2 (+ dynamic depth)
- Phase 3: TreeSpeculativeGeneratorV2AdaptiveV3 (+ historical acceptance rate adjustment)

Dataset: WikiText-2 from ModelScope
"""

import os
import sys
import json
import time
import gc
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class FullMetrics:
    """Metrics for full adaptive comparison."""
    method: str
    phase: str
    config: Dict[str, Any]
    
    # Timing metrics
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput_tps: float = 0.0
    
    # Token metrics
    total_tokens_generated: int = 0
    
    # Acceptance metrics
    acceptance_rate: float = 0.0
    avg_path_length: float = 0.0
    total_rounds: int = 0
    
    # Adaptive stats
    high_conf_ratio: float = 0.0
    medium_conf_ratio: float = 0.0
    low_conf_ratio: float = 0.0
    
    # Phase 2 depth stats
    early_stops: int = 0
    deep_expansions: int = 0
    
    # Phase 3 adjustment stats
    total_adjustments: int = 0
    final_base_depth: int = 0
    final_high_conf_threshold: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Speedup
    speedup: float = 1.0


class FullAdaptiveBenchmark:
    """Full benchmark comparing all adaptive tree methods."""
    
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        device: str = "cuda",
        num_samples: int = 10,
        max_new_tokens: int = 500,
        warmup_runs: int = 2,
        min_prompt_length: int = 200,
        max_prompt_length: int = 800,
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.warmup_runs = warmup_runs
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        
        print("Loading models...")
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
        print(f"Loaded {len(self.prompts)} prompts from WikiText dataset")
        
        self.results: List[FullMetrics] = []
        self.baseline_throughput: Optional[float] = None
    
    def _load_wikitext_prompts(self) -> List[str]:
        """Load prompts from WikiText dataset via ModelScope."""
        print("Loading WikiText dataset from ModelScope...")
        
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
                        if len(prompts) >= self.num_samples * 2:
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
        fallback_prompts = [
            """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.""",
            """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.""",
            """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.""",
            """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.""",
            """The transformer is a deep learning model introduced in 2017 that utilizes the mechanism of self-attention, differentially weighting the significance of each part of the input data.""",
            """Large language models are neural network models that are trained on massive amounts of text data. These models can generate human-like text and perform various natural language tasks.""",
            """Speculative decoding is an inference optimization technique for large language models that uses a smaller draft model to generate candidate tokens, which are then verified by the larger target model.""",
            """The attention mechanism allows neural networks to focus on specific parts of the input when generating each part of the output. In the transformer architecture, self-attention enables the model to weigh the importance of different positions.""",
            """Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.""",
            """Reinforcement learning from human feedback is a machine learning technique that trains models to align with human preferences through collecting human feedback on model outputs.""",
        ]
        return fallback_prompts[:self.num_samples]
    
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
    def benchmark_baseline(self) -> FullMetrics:
        """Benchmark pure autoregressive generation as baseline."""
        print("\n" + "="*70)
        print("Benchmarking: Baseline (Autoregressive)")
        print("="*70)
        
        metrics = FullMetrics(method="Baseline (AR)", phase="baseline", config={})
        original_eos = self._disable_eos()
        
        all_throughput = []
        
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
                total_time = (time.perf_counter() - start_prefill) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            throughput = self.max_new_tokens / (total_time / 1000)
            
            if not is_warmup:
                all_throughput.append(throughput)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        
        self.baseline_throughput = metrics.throughput_tps
        metrics.speedup = 1.0
        
        print(f"\nBaseline Results: Throughput: {metrics.throughput_tps:.1f} t/s")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_fixed(self) -> FullMetrics:
        """Benchmark fixed tree (TreeSpeculativeGeneratorV2)."""
        method_name = "Fixed Tree (D=5, B=2)"
        
        print("\n" + "="*70)
        print(f"Benchmarking: {method_name}")
        print("="*70)
        
        from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
        
        metrics = FullMetrics(
            method=method_name,
            phase="fixed",
            config={"tree_depth": 5, "branch_factor": 2}
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=5,
            branch_factor=2,
            probability_threshold=0.05,
            max_tree_nodes=256,
            device=self.device,
            use_compile=False
        )
        
        all_throughput, all_acceptance, all_path_lengths = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
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
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: Throughput: {metrics.throughput_tps:.1f} t/s, Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_phase1(self) -> FullMetrics:
        """Benchmark Phase 1 (adaptive branching)."""
        method_name = "Phase 1: Adaptive Branch"
        
        print("\n" + "="*70)
        print(f"Benchmarking: {method_name}")
        print("="*70)
        
        from spec_decode.core.tree_speculative_generator_adaptive import TreeSpeculativeGeneratorV2Adaptive
        
        metrics = FullMetrics(
            method=method_name,
            phase="phase1",
            config={"base_depth": 5, "min_branch": 1, "max_branch": 4}
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2Adaptive(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=5,
            branch_factor=2,
            probability_threshold=0.05,
            max_tree_nodes=256,
            device=self.device,
            use_compile=False,
            high_conf_threshold=0.8,
            low_conf_threshold=0.3,
            min_branch=1,
            max_branch=4,
        )
        
        all_throughput, all_acceptance, all_path_lengths = [], [], []
        all_high_conf, all_medium_conf, all_low_conf = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
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
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                
                adaptive_stats = stats.get('adaptive_stats', {})
                all_high_conf.append(adaptive_stats.get('high_conf_ratio', 0))
                all_medium_conf.append(adaptive_stats.get('medium_conf_ratio', 0))
                all_low_conf.append(adaptive_stats.get('low_conf_ratio', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.high_conf_ratio = sum(all_high_conf) / len(all_high_conf) if all_high_conf else 0
        metrics.medium_conf_ratio = sum(all_medium_conf) / len(all_medium_conf) if all_medium_conf else 0
        metrics.low_conf_ratio = sum(all_low_conf) / len(all_low_conf) if all_low_conf else 0
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: Throughput: {metrics.throughput_tps:.1f} t/s, Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_phase2(self) -> FullMetrics:
        """Benchmark Phase 2 (adaptive branch + dynamic depth)."""
        method_name = "Phase 2: + Dynamic Depth"
        
        print("\n" + "="*70)
        print(f"Benchmarking: {method_name}")
        print("="*70)
        
        from spec_decode.core.tree_speculative_generator_adaptive import TreeSpeculativeGeneratorV2AdaptiveV2
        
        metrics = FullMetrics(
            method=method_name,
            phase="phase2",
            config={"base_depth": 5, "max_depth": 8}
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2AdaptiveV2(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=5,
            branch_factor=2,
            probability_threshold=0.05,
            max_tree_nodes=256,
            device=self.device,
            use_compile=False,
            high_conf_threshold=0.8,
            low_conf_threshold=0.3,
            min_branch=1,
            max_branch=4,
            early_stop_threshold=0.1,
            deep_expand_threshold=0.5,
            max_depth=8,
        )
        
        all_throughput, all_acceptance, all_path_lengths = [], [], []
        all_early_stops, all_deep_expansions = [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
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
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                
                depth_stats = stats.get('depth_stats', {})
                all_early_stops.append(depth_stats.get('early_stops', 0))
                all_deep_expansions.append(depth_stats.get('deep_expansions', 0))
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.early_stops = int(sum(all_early_stops) / len(all_early_stops)) if all_early_stops else 0
        metrics.deep_expansions = int(sum(all_deep_expansions) / len(all_deep_expansions)) if all_deep_expansions else 0
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: Throughput: {metrics.throughput_tps:.1f} t/s, Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_phase3(self) -> FullMetrics:
        """Benchmark Phase 3 (full adaptive with history-based adjustment)."""
        method_name = "Phase 3: + History Adjust"
        
        print("\n" + "="*70)
        print(f"Benchmarking: {method_name}")
        print("="*70)
        
        from spec_decode.core.tree_speculative_generator_adaptive import TreeSpeculativeGeneratorV2AdaptiveV3
        
        metrics = FullMetrics(
            method=method_name,
            phase="phase3",
            config={"base_depth": 5, "max_depth": 8, "history_window": 10}
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2AdaptiveV3(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=5,
            branch_factor=2,
            probability_threshold=0.05,
            max_tree_nodes=256,
            device=self.device,
            use_compile=False,
            high_conf_threshold=0.8,
            low_conf_threshold=0.3,
            min_branch=1,
            max_branch=4,
            early_stop_threshold=0.1,
            deep_expand_threshold=0.5,
            max_depth=8,
            history_window=10,
            target_acceptance_rate=0.7,
            adjustment_rate=0.05,
            enable_auto_adjust=True,
        )
        
        all_throughput, all_acceptance, all_path_lengths = [], [], []
        total_adjustments = 0
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
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
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                total_adjustments = stats.get('total_adjustments', 0)
                
                current_params = stats.get('current_params', {})
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, "
                      f"depth={current_params.get('current_base_depth', 5)}")
        
        self._restore_eos(original_eos)
        
        # Get final stats
        final_stats = generator.get_stats()
        final_params = final_stats.get('current_params', {})
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.total_adjustments = total_adjustments
        metrics.final_base_depth = final_params.get('current_base_depth', 5)
        metrics.final_high_conf_threshold = final_params.get('current_high_conf_threshold', 0.8)
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: Throughput: {metrics.throughput_tps:.1f} t/s, Speedup: {metrics.speedup:.2f}x")
        print(f"Final params: depth={metrics.final_base_depth}, adjustments={metrics.total_adjustments}")
        
        self.results.append(metrics)
        return metrics
    
    def run_full_study(self):
        """Run full comparison study."""
        print("\n" + "#"*80)
        print("# Full Adaptive Tree Speculative Decoding Comparison")
        print(f"# Dataset: WikiText-2 (ModelScope)")
        print(f"# Target Model: {self.target_model_path}")
        print(f"# Draft Model: {self.draft_model_path}")
        print(f"# Max New Tokens: {self.max_new_tokens}")
        print(f"# Samples: {len(self.prompts)}, Warmup: {self.warmup_runs}")
        print("#"*80)
        
        # Run all benchmarks
        self.benchmark_baseline()
        self.benchmark_fixed()
        self.benchmark_phase1()
        self.benchmark_phase2()
        self.benchmark_phase3()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            "experiment_info": {
                "study": "full_adaptive_comparison",
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "dataset": "WikiText-2 (ModelScope)",
                "max_new_tokens": self.max_new_tokens,
                "num_samples": len(self.prompts),
                "warmup_runs": self.warmup_runs,
            },
            "results": [asdict(m) for m in self.results],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Print summary table
        print("\n" + "="*110)
        print("FULL ADAPTIVE TREE COMPARISON RESULTS")
        print("="*110)
        print(f"\n{'Method':<30} {'Phase':<10} {'Throughput':<12} {'Speedup':<10} {'Accept%':<10} {'PathLen':<10}")
        print("-"*110)
        
        for m in self.results:
            accept_str = f"{m.acceptance_rate:.1%}" if m.acceptance_rate > 0 else "N/A"
            path_str = f"{m.avg_path_length:.2f}" if m.avg_path_length > 0 else "N/A"
            print(f"{m.method:<30} {m.phase:<10} {m.throughput_tps:>8.1f} t/s {m.speedup:>7.2f}x {accept_str:>10} {path_str:>10}")
        
        # Calculate improvements
        print("\n" + "="*110)
        print("IMPROVEMENT ANALYSIS (vs Baseline)")
        print("="*110)
        
        baseline_tps = self.results[0].throughput_tps if self.results else 0
        for m in self.results[1:]:
            improvement = ((m.throughput_tps / baseline_tps) - 1) * 100 if baseline_tps > 0 else 0
            print(f"{m.method:<30}: {m.speedup:.2f}x speedup ({improvement:+.1f}%)")
        
        return summary
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        summary = self._generate_summary()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Adaptive Tree Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--output", type=str, default="results/adaptive_full_comparison.json")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=800)
    
    args = parser.parse_args()
    
    benchmark = FullAdaptiveBenchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        max_prompt_length=args.max_prompt_length,
    )
    
    benchmark.run_full_study()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()



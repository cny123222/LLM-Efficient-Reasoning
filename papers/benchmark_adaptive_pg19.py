#!/usr/bin/env python3
"""
Benchmark Adaptive Tree Speculative Decoding on PG-19 Dataset

This script benchmarks adaptive tree speculative decoding on PG-19 dataset
and compares it with baseline autoregressive generation and fixed tree speculative decoding.

Configuration:
- Dataset: PG-19 (parquet file)
- Max prompt length: 1000 characters
- Max new tokens: 1000
- Comparison methods:
  1. Baseline (AR) - Pure autoregressive generation
  2. Tree Spec Decode (Fixed Tree) - Fixed tree depth and branch factor
  3. Adaptive Tree (Phase 3) - Adaptive tree with all optimizations

Metrics collected:
- Throughput (tokens/second)
- TTFT (Time To First Token, ms)
- TPOT (Time Per Output Token, ms)
- Acceptance Rate (%)
- Average Path Length
- Total Rounds
- Peak Memory (MB)
- Speedup vs Baseline
"""

import os
import sys
import json
import time
import gc
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Metrics:
    """Metrics for benchmark results."""
    # Identification
    method: str
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
    
    # Adaptive stats (for adaptive methods)
    high_conf_ratio: float = 0.0
    early_stops: int = 0
    deep_expansions: int = 0
    total_adjustments: int = 0
    
    # Resource metrics
    peak_memory_mb: float = 0.0
    
    # Variance for error bars
    throughput_std: float = 0.0
    acceptance_std: float = 0.0
    path_length_std: float = 0.0


class PG19Benchmark:
    """Benchmark suite for PG-19 dataset."""
    
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        pg19_path: str,
        device: str = "cuda",
        num_samples: int = 10,
        warmup_runs: int = 2,
        max_prompt_length: int = 1000,
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.pg19_path = pg19_path
        self.device = device
        self.num_samples = num_samples
        self.warmup_runs = warmup_runs
        self.max_prompt_length = max_prompt_length
        
        print("="*80)
        print("ADAPTIVE TREE SPECULATIVE DECODING - PG-19 BENCHMARK")
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
        
        self.prompts = self._load_pg19_prompts()
        print(f"Loaded {len(self.prompts)} prompts from PG-19 dataset")
        
        self.all_results: List[Metrics] = []
        self.baseline_throughput: Optional[float] = None
    
    def _load_pg19_prompts(self) -> List[str]:
        """Load prompts from PG-19 parquet file."""
        print(f"\nLoading PG-19 prompts from {self.pg19_path}...")
        print(f"  Target: {self.num_samples} prompts, max {self.max_prompt_length} chars each")
        
        try:
            df = pd.read_parquet(self.pg19_path)
            
            prompts = []
            min_length = 200  # Minimum prompt length
            
            for idx, row in df.iterrows():
                text = row.get('text', '')
                if not text or len(text) < min_length:
                    continue
                
                # Clean up and truncate
                text = text.strip()
                if len(text) > self.max_prompt_length:
                    text = text[:self.max_prompt_length]
                
                prompts.append(text)
                
                if len(prompts) >= self.num_samples:
                    break
            
            # If we don't have enough, duplicate
            while len(prompts) < self.num_samples:
                prompts.append(prompts[len(prompts) % max(1, len(prompts))])
            
            # Print prompt statistics
            lengths = [len(p) for p in prompts]
            print(f"  Loaded {len(prompts)} prompts")
            print(f"  Prompt lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.0f} chars")
            
            return prompts[:self.num_samples]
            
        except Exception as e:
            print(f"  Warning: Failed to load PG-19: {e}")
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> List[str]:
        """Fallback prompts if PG-19 loading fails."""
        fallback_prompts = [
            """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.""",
            
            """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.""",
            
            """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing is used in many applications including machine translation.""",
            
            """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, and medical image analysis.""",
            
            """The transformer is a deep learning model introduced in 2017 that utilizes the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing and computer vision. Like recurrent neural networks, transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. Unlike RNNs, transformers process the entire input all at once.""",
            
            """Large language models are neural network models that are trained on massive amounts of text data. These models can generate human-like text and perform various natural language tasks. The development of large language models has revolutionized the field of artificial intelligence, enabling new applications in chatbots, content generation, code completion, and many other areas. Recent advances include models with billions of parameters trained on internet-scale datasets.""",
            
            """Speculative decoding is an inference optimization technique for large language models that uses a smaller draft model to generate candidate tokens, which are then verified by the larger target model. This approach can significantly accelerate inference by enabling parallel verification of multiple tokens while maintaining the same output distribution as standard autoregressive decoding. The key insight is that small models can often predict what larger models will generate.""",
            
            """The attention mechanism allows neural networks to focus on specific parts of the input when generating each part of the output. In the transformer architecture, self-attention enables the model to weigh the importance of different positions in a sequence when encoding each position. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. This mechanism has become fundamental to modern NLP.""",
            
            """Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although practical ties between the fields are limited.""",
            
            """Reinforcement learning from human feedback is a machine learning technique that trains models to align with human preferences. The process typically involves collecting human feedback on model outputs, training a reward model to predict human preferences, and then fine-tuning the language model to maximize the predicted reward. This approach has been instrumental in improving the helpfulness and safety of conversational AI systems.""",
        ]
        
        # Truncate to max_prompt_length
        prompts = [p[:self.max_prompt_length] for p in fallback_prompts[:self.num_samples]]
        
        # Duplicate if needed
        while len(prompts) < self.num_samples:
            prompts.append(prompts[len(prompts) % len(prompts)])
        
        return prompts[:self.num_samples]
    
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
    def benchmark_baseline(self, max_new_tokens: int) -> Metrics:
        """Benchmark pure autoregressive generation."""
        method_name = "Baseline (AR)"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        metrics = Metrics(
            method=method_name,
            config={"max_new_tokens": max_new_tokens}
        )
        
        original_eos = self._disable_eos()
        
        all_ttft, all_tpot, all_throughput, all_total_time = [], [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
            
            with self._memory_tracking():
                # TTFT measurement
                torch.cuda.synchronize()
                start_prefill = time.perf_counter()
                outputs = self.target_model(input_ids=input_ids, use_cache=True, return_dict=True)
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - start_prefill) * 1000
                
                past_kv = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                # Decode phase
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
                all_total_time.append(total_time)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms")
        
        self._restore_eos(original_eos)
        
        # Calculate statistics
        metrics.ttft_ms = np.mean(all_ttft)
        metrics.ttft_std = np.std(all_ttft)
        metrics.tpot_ms = np.mean(all_tpot)
        metrics.tpot_std = np.std(all_tpot)
        metrics.throughput_tps = np.mean(all_throughput)
        metrics.throughput_std = np.std(all_throughput)
        metrics.total_time_ms = np.mean(all_total_time)
        metrics.total_tokens_generated = max_new_tokens
        metrics.total_rounds = max_new_tokens
        metrics.tokens_per_round = 1.0
        metrics.peak_memory_mb = peak_memory
        metrics.speedup = 1.0
        
        # Always record baseline throughput for speedup calculation
        if self.baseline_throughput is None:
            self.baseline_throughput = metrics.throughput_tps
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, "
              f"TTFT={metrics.ttft_ms:.1f}±{metrics.ttft_std:.1f}ms, "
              f"TPOT={metrics.tpot_ms:.2f}±{metrics.tpot_std:.2f}ms")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_tree_spec_decode(
        self,
        max_new_tokens: int,
        tree_depth: int = 5,
        branch_factor: int = 2,
    ) -> Metrics:
        """Benchmark fixed tree speculative decoding."""
        method_name = f"Tree Spec Decode (D={tree_depth}, B={branch_factor})"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
        
        metrics = Metrics(
            method=method_name,
            config={
                "max_new_tokens": max_new_tokens,
                "tree_depth": tree_depth,
                "branch_factor": branch_factor,
                "max_tree_nodes": 256,
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
        
        all_ttft, all_tpot, all_throughput = [], [], []
        all_acceptance, all_path_lengths, all_rounds = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT
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
        
        # Calculate statistics
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
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, "
              f"Speedup={metrics.speedup:.2f}x, Accept={metrics.acceptance_rate:.1%}")
        
        self.all_results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_adaptive_tree(
        self,
        max_new_tokens: int,
        base_depth: int = 5,
        max_depth: int = 9,
        high_conf_threshold: float = 0.9,
        low_conf_threshold: float = 0.4,
        min_branch: int = 1,
        max_branch: int = 3,
    ) -> Metrics:
        """Benchmark adaptive tree speculative decoding (Phase 3)."""
        method_name = "Adaptive Tree (Phase 3)"
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {method_name}")
        print(f"{'='*70}")
        
        from spec_decode.core.tree_speculative_generator_adaptive import TreeSpeculativeGeneratorV2AdaptiveV3
        
        metrics = Metrics(
            method=method_name,
            config={
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
        
        all_ttft, all_tpot, all_throughput = [], [], []
        all_acceptance, all_path_lengths, all_rounds = [], [], []
        all_high_conf, all_early_stops, all_deep_expansions = [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT
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
                
                # Adaptive stats
                adaptive_stats = stats.get('adaptive_stats', {})
                all_high_conf.append(adaptive_stats.get('high_conf_ratio', 0))
                
                # Phase 2+ stats
                depth_stats = stats.get('depth_stats', {})
                all_early_stops.append(depth_stats.get('early_stops', 0))
                all_deep_expansions.append(depth_stats.get('deep_expansions', 0))
                
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, "
                      f"accept={stats.get('acceptance_rate', 0):.1%}, path={stats.get('avg_accepted_path_length', 0):.2f}")
        
        self._restore_eos(original_eos)
        
        # Get final stats for Phase 3
        final_stats = generator.get_stats()
        
        # Calculate statistics
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
        
        # Adaptive stats
        metrics.high_conf_ratio = np.mean(all_high_conf) if all_high_conf else 0
        metrics.early_stops = int(np.mean(all_early_stops)) if all_early_stops else 0
        metrics.deep_expansions = int(np.mean(all_deep_expansions)) if all_deep_expansions else 0
        metrics.total_adjustments = final_stats.get('total_adjustments', 0)
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: {metrics.throughput_tps:.1f}±{metrics.throughput_std:.1f} t/s, "
              f"Speedup={metrics.speedup:.2f}x, Accept={metrics.acceptance_rate:.1%}")
        
        self.all_results.append(metrics)
        return metrics
    
    def run_comparison(self, max_new_tokens: int = 1000):
        """Run comparison of all methods."""
        print("\n" + "#"*80)
        print("# PG-19 DATASET COMPARISON")
        print("#"*80)
        
        # Baseline
        self.benchmark_baseline(max_new_tokens)
        
        # Tree Spec Decode (fixed tree)
        self.benchmark_tree_spec_decode(max_new_tokens, tree_depth=5, branch_factor=2)
        
        # Adaptive Tree (Phase 3)
        self.benchmark_adaptive_tree(max_new_tokens)
    
    def print_summary(self):
        """Print a summary of all results."""
        print("\n" + "="*120)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*120)
        print(f"{'Method':<35} {'Throughput':<15} {'Speedup':<10} {'TTFT':<12} {'TPOT':<12} {'Accept':<10} {'PathLen':<10}")
        print("-"*120)
        
        for r in self.all_results:
            throughput_str = f"{r.throughput_tps:.1f}±{r.throughput_std:.1f}"
            ttft_str = f"{r.ttft_ms:.1f}ms"
            tpot_str = f"{r.tpot_ms:.2f}ms"
            accept_str = f"{r.acceptance_rate:.1%}" if r.acceptance_rate > 0 else "N/A"
            path_str = f"{r.avg_path_length:.2f}" if r.avg_path_length > 0 else "N/A"
            
            print(f"{r.method:<35} {throughput_str:<15} {r.speedup:<10.2f}x {ttft_str:<12} {tpot_str:<12} {accept_str:<10} {path_str:<10}")
    
    def save_results(self, output_path: str):
        """Save all results to JSON file."""
        summary = {
            "experiment_info": {
                "study": "adaptive_tree_speculative_decoding_pg19",
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "dataset": "PG-19 (parquet)",
                "pg19_path": self.pg19_path,
                "num_samples": self.num_samples,
                "warmup_runs": self.warmup_runs,
                "max_prompt_length": self.max_prompt_length,
            },
            "all_results": [asdict(r) for r in self.all_results],
            "timestamp": datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save a CSV for easy plotting
        csv_path = output_path.replace('.json', '.csv')
        self._save_csv(csv_path)
    
    def _save_csv(self, output_path: str):
        """Save results to CSV for easy plotting."""
        import csv
        
        fieldnames = [
            'method', 'throughput_tps', 'throughput_std', 'speedup',
            'ttft_ms', 'ttft_std', 'tpot_ms', 'tpot_std',
            'acceptance_rate', 'acceptance_std', 'avg_path_length', 'path_length_std',
            'total_rounds', 'tokens_per_round', 'peak_memory_mb'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in self.all_results:
                row = {k: getattr(r, k, '') for k in fieldnames}
                writer.writerow(row)
        
        print(f"CSV saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Adaptive Tree Speculative Decoding on PG-19 Dataset")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                       help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                       help="Path to draft model")
    parser.add_argument("--pg19-path", type=str, default="/mnt/disk1/ljm/LLM-Efficient-Reasoning/data/pg19.parquet",
                       help="Path to PG-19 parquet file")
    parser.add_argument("--output", type=str, default="results/adaptive/pg19/pg19_benchmark.json",
                       help="Output JSON file path")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of samples to benchmark")
    parser.add_argument("--warmup-runs", type=int, default=2,
                       help="Number of warmup runs")
    parser.add_argument("--max-prompt-length", type=int, default=1000,
                       help="Maximum prompt length in characters")
    parser.add_argument("--max-new-tokens", type=int, default=1000,
                       help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    benchmark = PG19Benchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        pg19_path=args.pg19_path,
        num_samples=args.num_samples,
        warmup_runs=args.warmup_runs,
        max_prompt_length=args.max_prompt_length,
    )
    
    benchmark.run_comparison(max_new_tokens=args.max_new_tokens)
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()


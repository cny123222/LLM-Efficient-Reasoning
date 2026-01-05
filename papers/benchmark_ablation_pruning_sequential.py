#!/usr/bin/env python3
"""
Ablation Study: Tree Speculative Decoding Pruning Strategy

This script compares Tree V2 performance with and without probability-based pruning.
- With pruning: Uses original probability_threshold values (0.03, 0.05)
- Without pruning: Sets probability_threshold = 0.0 (disables pruning)

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
class AblationMetrics:
    """Metrics for ablation study."""
    method: str
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
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Speedup
    speedup: float = 1.0


class PruningAblationBenchmark:
    """Ablation study for Tree V2 pruning strategy using WikiText dataset."""
    
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
        
        # Load WikiText dataset
        self.prompts = self._load_wikitext_prompts()
        print(f"Loaded {len(self.prompts)} prompts from WikiText dataset")
        
        self.results: List[AblationMetrics] = []
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
            
            if len(valid_prompts) < self.num_samples:
                print(f"Warning: Only found {len(valid_prompts)} valid prompts")
                while len(valid_prompts) < self.num_samples and len(valid_prompts) > 0:
                    valid_prompts.append(valid_prompts[len(valid_prompts) % len(valid_prompts)])
            
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
    def _measure_prefill_ttft(self, prompt: str, model) -> float:
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(self.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000
    
    @torch.inference_mode()
    def benchmark_baseline(self) -> AblationMetrics:
        """Benchmark pure autoregressive generation as baseline."""
        print("\n" + "="*70)
        print("Benchmarking: Baseline (Autoregressive)")
        print("="*70)
        
        metrics = AblationMetrics(method="Baseline (AR)", config={})
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
                
                for _ in range(self.max_new_tokens):
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    outputs = self.target_model(input_ids=next_token, past_key_values=past_kv, use_cache=True, return_dict=True)
                    past_kv = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                
                torch.cuda.synchronize()
                decode_time = (time.perf_counter() - start_decode) * 1000
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            total_time = ttft + decode_time
            tpot = decode_time / self.max_new_tokens
            throughput = self.max_new_tokens / (total_time / 1000)
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s")
        
        self._restore_eos(original_eos)
        
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        
        self.baseline_throughput = metrics.throughput_tps
        metrics.speedup = 1.0
        
        print(f"\nBaseline Results: Throughput: {metrics.throughput_tps:.1f} t/s")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_tree_v2(
        self,
        tree_depth: int,
        branch_factor: int,
        probability_threshold: float,
        label: str = ""
    ) -> AblationMetrics:
        """Benchmark Tree V2 with specified configuration."""
        pruning_status = "无剪枝" if probability_threshold == 0.0 else f"有剪枝(t={probability_threshold})"
        method_name = f"Tree V2 (D={tree_depth}, B={branch_factor}) {pruning_status}"
        if label:
            method_name = f"{label}: {method_name}"
        
        print("\n" + "="*70)
        print(f"Benchmarking: {method_name}")
        print("="*70)
        
        from spec_decode.core.tree_speculative_generator_sequential import TreeSpeculativeGeneratorV2Sequential as TreeSpeculativeGeneratorV2
        
        metrics = AblationMetrics(
            method=method_name,
            config={
                "tree_depth": tree_depth,
                "branch_factor": branch_factor,
                "probability_threshold": probability_threshold,
                "pruning_enabled": probability_threshold > 0.0,
                "max_tree_nodes": 512
            }
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            probability_threshold=probability_threshold,
            max_tree_nodes=512,  # 增加节点预算以支持更深的树（D=7需要255节点）
            device=self.device,
            use_compile=False
        )
        
        all_throughput, all_acceptance, all_path_lengths, all_rounds, all_ttft, all_tpot = [], [], [], [], [], []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            ttft = self._measure_prefill_ttft(prompt, self.target_model)
            
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
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, accept={stats.get('acceptance_rate', 0):.2%}, path_len={stats.get('avg_accepted_path_length', 0):.1f}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.total_rounds = int(sum(all_rounds) / len(all_rounds))
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nResults: Throughput: {metrics.throughput_tps:.1f} t/s, Speedup: {metrics.speedup:.2f}x, Accept: {metrics.acceptance_rate:.2%}")
        
        self.results.append(metrics)
        return metrics
    
    def run_ablation_study(self):
        """Run the complete ablation study comparing pruning vs no-pruning."""
        print("\n" + "#"*80)
        print("# Ablation Study: Tree Speculative Decoding Pruning Strategy")
        print(f"# Dataset: WikiText-2 (ModelScope)")
        print(f"# Target Model: {self.target_model_path}")
        print(f"# Draft Model: {self.draft_model_path}")
        print(f"# Max New Tokens: {self.max_new_tokens}")
        print(f"# Samples: {len(self.prompts)}, Warmup: {self.warmup_runs}")
        print("#"*80)
        
        # 1. Baseline
        self.benchmark_baseline()
        
        # 2. Ablation configurations
        # Format: (depth, branch, original_threshold)
        ablation_configs = [
            (4, 2, 0.05),
            (4, 2, 0.03),
            (5, 2, 0.05),
            (6, 2, 0.05),
            (7, 2, 0.05),
        ]
        
        for depth, branch, orig_threshold in ablation_configs:
            # With pruning (original)
            self.benchmark_tree_v2(
                tree_depth=depth,
                branch_factor=branch,
                probability_threshold=orig_threshold,
                label="对照组"
            )
            
            # Without pruning (ablation)
            self.benchmark_tree_v2(
                tree_depth=depth,
                branch_factor=branch,
                probability_threshold=0.0,
                label="消融组"
            )
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of ablation results."""
        summary = {
            "experiment_info": {
                "type": "ablation_study",
                "focus": "pruning_strategy",
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "dataset": "WikiText-2 (ModelScope)",
                "max_new_tokens": self.max_new_tokens,
                "num_samples": len(self.prompts),
                "warmup_runs": self.warmup_runs,
                "max_prompt_length": self.max_prompt_length,
            },
            "results": [asdict(m) for m in self.results],
            "comparison": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Generate comparison pairs
        results_dict = {m.method: m for m in self.results}
        ablation_configs = [
            (4, 2, 0.05),
            (4, 2, 0.03),
            (5, 2, 0.05),
            (6, 2, 0.05),
            (7, 2, 0.05),
        ]
        
        for depth, branch, orig_threshold in ablation_configs:
            pruned_key = f"对照组: Tree V2 (D={depth}, B={branch}) 有剪枝(t={orig_threshold})"
            unpruned_key = f"消融组: Tree V2 (D={depth}, B={branch}) 无剪枝"
            
            if pruned_key in results_dict and unpruned_key in results_dict:
                pruned = results_dict[pruned_key]
                unpruned = results_dict[unpruned_key]
                
                comparison = {
                    "config": f"D={depth}, B={branch}",
                    "original_threshold": orig_threshold,
                    "with_pruning": {
                        "throughput": pruned.throughput_tps,
                        "speedup": pruned.speedup,
                        "acceptance_rate": pruned.acceptance_rate,
                        "avg_path_length": pruned.avg_path_length,
                    },
                    "without_pruning": {
                        "throughput": unpruned.throughput_tps,
                        "speedup": unpruned.speedup,
                        "acceptance_rate": unpruned.acceptance_rate,
                        "avg_path_length": unpruned.avg_path_length,
                    },
                    "pruning_impact": {
                        "throughput_diff": pruned.throughput_tps - unpruned.throughput_tps,
                        "throughput_ratio": pruned.throughput_tps / unpruned.throughput_tps if unpruned.throughput_tps > 0 else 0,
                        "acceptance_diff": pruned.acceptance_rate - unpruned.acceptance_rate,
                        "path_length_diff": pruned.avg_path_length - unpruned.avg_path_length,
                    }
                }
                summary["comparison"].append(comparison)
        
        # Print summary table
        print("\n" + "="*100)
        print("ABLATION STUDY RESULTS: Pruning vs No-Pruning")
        print("="*100)
        print(f"\n{'Method':<55} {'Throughput':<12} {'Speedup':<10} {'Accept%':<10} {'PathLen':<10}")
        print("-"*100)
        
        for m in self.results:
            accept_str = f"{m.acceptance_rate:.1%}" if m.acceptance_rate > 0 else "N/A"
            path_str = f"{m.avg_path_length:.2f}" if m.avg_path_length > 0 else "N/A"
            print(f"{m.method:<55} {m.throughput_tps:>8.1f} t/s {m.speedup:>7.2f}x {accept_str:>10} {path_str:>10}")
        
        print("\n" + "="*100)
        print("COMPARISON SUMMARY")
        print("="*100)
        for comp in summary["comparison"]:
            print(f"\n配置: {comp['config']} (原始阈值: {comp['original_threshold']})")
            print(f"  有剪枝: {comp['with_pruning']['throughput']:.1f} t/s, 接受率: {comp['with_pruning']['acceptance_rate']:.2%}")
            print(f"  无剪枝: {comp['without_pruning']['throughput']:.1f} t/s, 接受率: {comp['without_pruning']['acceptance_rate']:.2%}")
            print(f"  剪枝影响: 吞吐量变化 {comp['pruning_impact']['throughput_diff']:+.1f} t/s ({comp['pruning_impact']['throughput_ratio']:.2f}x)")
        
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
    
    parser = argparse.ArgumentParser(description="Ablation Study: Tree V2 Pruning Strategy")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--output", type=str, default="results/ablation_pruning_wikitext.json")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=800)
    
    args = parser.parse_args()
    
    benchmark = PruningAblationBenchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        max_prompt_length=args.max_prompt_length,
    )
    
    benchmark.run_ablation_study()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Speculative Decoding Methods
Using PG19 Dataset (Project Gutenberg books)

This script benchmarks multiple speculative decoding configurations measuring:
1. TTFT (Time To First Token) - Prefill latency
2. TPOT (Time Per Output Token) - Per-token generation latency
3. Throughput (tokens/second)
4. FLOPs estimation (based on model architecture)
5. Token acceptance rates comparison (Tree vs Linear)

Dataset: PG19 (Project Gutenberg books) from local parquet file

Tested configurations:
1. Tree V2 (D=8, B=3, t=0.03) - Best performer
2. HuggingFace Assisted Generation
3. Linear K=6
4. Streaming K=6 cache=1024
5. Linear K=7, K=8, K=5
6. Baseline (Autoregressive)
7. Streaming K=6 cache=512
"""

import os
import sys
import json
import time
import gc
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics."""
    method: str
    config: Dict[str, Any]
    
    # Timing metrics
    ttft_ms: float = 0.0           # Time To First Token (ms)
    tpot_ms: float = 0.0           # Time Per Output Token (ms)
    total_latency_ms: float = 0.0  # Total generation time (ms)
    throughput_tps: float = 0.0    # Throughput (tokens/second)
    
    # Token metrics
    total_tokens_generated: int = 0
    prompt_tokens: int = 0
    
    # Acceptance metrics (for spec decode)
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    tokens_per_round: float = 0.0
    total_rounds: int = 0
    
    # Tree-specific metrics
    avg_path_length: float = 0.0
    max_path_length: int = 0
    
    # FLOPs estimation
    prefill_flops: float = 0.0
    decode_flops: float = 0.0
    total_flops: float = 0.0
    flops_per_token: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Speedup
    speedup: float = 1.0


@dataclass
class ModelFLOPsConfig:
    """Model configuration for FLOPs estimation."""
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    
    def estimate_forward_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """Estimate FLOPs for a single forward pass."""
        d = self.hidden_size
        L = self.num_layers
        V = self.vocab_size
        ff = self.intermediate_size
        
        # Attention FLOPs per layer: 4 * seq_len * d^2 + 2 * seq_len^2 * d
        attention_flops = L * (4 * seq_len * d * d + 2 * seq_len * seq_len * d)
        
        # FFN FLOPs per layer: 2 * seq_len * d * ff
        ffn_flops = L * 2 * seq_len * d * ff
        
        # LM head: seq_len * d * V
        lm_head_flops = seq_len * d * V
        
        total = batch_size * (attention_flops + ffn_flops + lm_head_flops)
        return total


class FLOPsEstimator:
    """Estimate FLOPs for different generation strategies."""
    
    def __init__(self, target_config: ModelFLOPsConfig, draft_config: ModelFLOPsConfig):
        self.target = target_config
        self.draft = draft_config
    
    def estimate_prefill_flops(self, prompt_len: int) -> float:
        """Estimate FLOPs for prefill phase."""
        return self.target.estimate_forward_flops(prompt_len)
    
    def estimate_ar_decode_flops(self, num_tokens: int, prompt_len: int) -> float:
        """Estimate FLOPs for autoregressive decoding."""
        total = 0
        for i in range(num_tokens):
            seq_len = prompt_len + i + 1
            total += self.target.estimate_forward_flops(1)  # Single token forward
        return total
    
    def estimate_spec_decode_flops(
        self, num_tokens: int, K: int, num_rounds: int, prompt_len: int
    ) -> float:
        """Estimate FLOPs for speculative decoding."""
        # Draft model: K tokens per round
        draft_flops = num_rounds * K * self.draft.estimate_forward_flops(1)
        
        # Target model: verify K+1 tokens per round
        target_flops = num_rounds * self.target.estimate_forward_flops(K + 1)
        
        return draft_flops + target_flops
    
    def estimate_tree_decode_flops(
        self, num_tokens: int, depth: int, branch: int, 
        num_rounds: int, prompt_len: int
    ) -> float:
        """Estimate FLOPs for tree speculative decoding."""
        # Tree size estimation
        tree_size = sum(branch ** i for i in range(depth + 1))
        
        # Draft model: generate tree
        draft_flops = num_rounds * tree_size * self.draft.estimate_forward_flops(1)
        
        # Target model: verify tree
        target_flops = num_rounds * self.target.estimate_forward_flops(tree_size)
        
        return draft_flops + target_flops


class PG19Benchmark:
    """Comprehensive benchmark using PG19 dataset (Project Gutenberg books)."""
    
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        device: str = "cuda",
        num_samples: int = 5,
        max_new_tokens: int = 500,
        warmup_runs: int = 2,
        min_prompt_length: int = 200,  # Minimum prompt length in characters
        max_prompt_length: int = 800,  # Maximum prompt length in characters
        data_path: str = None,  # Path to pg19.parquet file
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.warmup_runs = warmup_runs
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        
        # Set default data path
        if data_path is None:
            self.data_path = os.path.join(project_root, "data", "pg19.parquet")
        else:
            self.data_path = data_path
        
        # Load models
        print("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models on single GPU to avoid device mismatch
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
        
        # Setup FLOPs estimator
        target_config = ModelFLOPsConfig(
            hidden_size=self.target_model.config.hidden_size,
            num_layers=self.target_model.config.num_hidden_layers,
            num_attention_heads=self.target_model.config.num_attention_heads,
            intermediate_size=self.target_model.config.intermediate_size,
            vocab_size=self.target_model.config.vocab_size
        )
        draft_config = ModelFLOPsConfig(
            hidden_size=self.draft_model.config.hidden_size,
            num_layers=self.draft_model.config.num_hidden_layers,
            num_attention_heads=self.draft_model.config.num_attention_heads,
            intermediate_size=self.draft_model.config.intermediate_size,
            vocab_size=self.draft_model.config.vocab_size
        )
        self.flops_estimator = FLOPsEstimator(target_config, draft_config)
        
        # Load PG19 dataset
        self.prompts = self._load_pg19_prompts()
        print(f"Loaded {len(self.prompts)} prompts from PG19 dataset")
        
        # Results storage
        self.results: List[BenchmarkMetrics] = []
        self.baseline_throughput: Optional[float] = None
    
    def _load_pg19_prompts(self) -> List[str]:
        """Load prompts from PG19 parquet dataset.
        
        Uses simple loading logic (same as tree_param_search_pg19.py):
        - Directly read text from parquet
        - Simple truncation to max_prompt_length
        - No complex header skipping
        """
        print(f"Loading PG19 dataset from {self.data_path}...")
        print(f"  Target: {self.num_samples} prompts, {self.min_prompt_length}-{self.max_prompt_length} chars each")
        
        try:
            import pandas as pd
            
            # Load parquet file using pandas (same as tree_param_search_pg19.py)
            df = pd.read_parquet(self.data_path)
            
            prompts = []
            
            # Simple iteration - same logic as tree_param_search_pg19.py
            for idx, row in df.iterrows():
                text = row.get('text', '')
                if not text or len(text) < self.min_prompt_length:
                    continue
                
                # Clean up and truncate (simple logic)
                text = text.strip()
                if len(text) > self.max_prompt_length:
                    text = text[:self.max_prompt_length]
                
                prompts.append(text)
                
                if len(prompts) >= self.num_samples:
                    break
            
            # If we don't have enough, duplicate
            if len(prompts) < self.num_samples:
                print(f"Warning: Only found {len(prompts)} valid prompts, needed {self.num_samples}")
                while len(prompts) < self.num_samples and len(prompts) > 0:
                    prompts.append(prompts[len(prompts) % max(1, len(prompts))])
            
            # Print prompt statistics
            if prompts:
                lengths = [len(p) for p in prompts]
                print(f"  Loaded {len(prompts)} prompts")
                print(f"  Prompt lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f} chars")
            
            return prompts[:self.num_samples]
            
        except ImportError:
            print("Warning: pandas not installed, using fallback prompts")
            return self._get_fallback_prompts()
        except Exception as e:
            print(f"Warning: Failed to load PG19 dataset: {e}")
            print("Using fallback prompts instead")
            return self._get_fallback_prompts()
    
    def _get_fallback_prompts(self) -> List[str]:
        """Fallback prompts if PG19 loading fails."""
        fallback_prompts = [
            """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.""",
            
            """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data.""",
            
            """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.""",
            
            """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision and natural language processing.""",
            
            """The transformer is a deep learning model introduced in 2017 that utilizes the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing and computer vision. Like recurrent neural networks, transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization.""",
            
            """Large language models are neural network models that are trained on massive amounts of text data. These models can generate human-like text and perform various natural language tasks. The development of large language models has revolutionized the field of artificial intelligence, enabling new applications in chatbots, content generation, code completion, and many other areas.""",
            
            """Speculative decoding is an inference optimization technique for large language models that uses a smaller draft model to generate candidate tokens, which are then verified by the larger target model. This approach can significantly accelerate inference by enabling parallel verification of multiple tokens while maintaining the same output distribution as standard autoregressive decoding.""",
            
            """The attention mechanism allows neural networks to focus on specific parts of the input when generating each part of the output. In the transformer architecture, self-attention enables the model to weigh the importance of different positions in a sequence when encoding each position. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.""",
            
            """Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning.""",
            
            """Reinforcement learning from human feedback is a machine learning technique that trains models to align with human preferences. The process typically involves collecting human feedback on model outputs, training a reward model to predict human preferences, and then fine-tuning the language model to maximize the predicted reward. This approach has been instrumental in improving the helpfulness and safety of conversational AI systems.""",
        ]
        return fallback_prompts[:self.num_samples]
    
    @contextmanager
    def _memory_tracking(self):
        """Context manager for GPU memory tracking."""
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            pass
    
    def _cleanup(self):
        """Clean up GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _disable_eos(self):
        """Disable EOS token to force long generation."""
        original_eos = self.tokenizer.eos_token_id
        self.tokenizer.eos_token_id = 999999
        return original_eos
    
    def _restore_eos(self, original_eos):
        """Restore original EOS token."""
        self.tokenizer.eos_token_id = original_eos
    
    @torch.inference_mode()
    def _measure_prefill_ttft(self, prompt: str, model) -> float:
        """Measure Time To First Token (TTFT) - the prefill latency."""
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).input_ids.to(self.device)
        
        torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=True, return_dict=True)
        
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - start_prefill) * 1000
        
        return ttft_ms
    
    @torch.inference_mode()
    def benchmark_baseline(self) -> BenchmarkMetrics:
        """Benchmark pure autoregressive generation.
        
        Uses HuggingFace's generate() method for consistent comparison
        with tree_param_search_pg19.py.
        """
        print("\n" + "="*60)
        print("Benchmarking: Baseline (Autoregressive)")
        print("="*60)
        
        metrics = BenchmarkMetrics(method="Baseline (AR)", config={})
        
        original_eos = self._disable_eos()
        
        all_ttft = []
        all_tpot = []
        all_throughput = []
        all_tokens = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).input_ids.to(self.device)
            
            prompt_len = input_ids.shape[1]
            
            with self._memory_tracking():
                # Measure TTFT (prefill only)
                torch.cuda.synchronize()
                start_prefill = time.perf_counter()
                
                _ = self.target_model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True
                )
                
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - start_prefill) * 1000  # ms
                
                # Use HuggingFace's generate() for consistent throughput measurement
                # (same as tree_param_search_pg19.py)
                torch.cuda.synchronize()
                start_gen = time.perf_counter()
                
                output_ids = self.target_model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_gen) * 1000  # ms
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            
            num_tokens = output_ids.shape[1] - input_ids.shape[1]
            throughput = num_tokens / (total_time / 1000)
            tpot = total_time / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                all_tokens.append(num_tokens)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms")
        
        self._restore_eos(original_eos)
        
        # Average metrics
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.total_tokens_generated = int(sum(all_tokens) / len(all_tokens))
        metrics.total_latency_ms = metrics.ttft_ms + metrics.tpot_ms * metrics.total_tokens_generated
        metrics.peak_memory_mb = peak_memory
        
        # FLOPs estimation
        avg_prompt_len = 512
        metrics.prefill_flops = self.flops_estimator.estimate_prefill_flops(avg_prompt_len)
        metrics.decode_flops = self.flops_estimator.estimate_ar_decode_flops(
            metrics.total_tokens_generated, avg_prompt_len
        )
        metrics.total_flops = metrics.prefill_flops + metrics.decode_flops
        metrics.flops_per_token = metrics.decode_flops / metrics.total_tokens_generated
        
        # Baseline is reference
        self.baseline_throughput = metrics.throughput_tps
        metrics.speedup = 1.0
        
        print(f"\nBaseline Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Total FLOPs: {metrics.total_flops:.2e}")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_hf_assisted(self) -> BenchmarkMetrics:
        """Benchmark HuggingFace's assisted generation."""
        print("\n" + "="*60)
        print("Benchmarking: HuggingFace Assisted Generation")
        print("="*60)
        
        metrics = BenchmarkMetrics(method="HF Assisted", config={"num_assistant_tokens": 5})
        
        original_eos = self._disable_eos()
        
        all_ttft = []
        all_tpot = []
        all_throughput = []
        all_tokens = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT
            ttft = self._measure_prefill_ttft(prompt, self.target_model)
            
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).input_ids.to(self.device)
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                output_ids = self.target_model.generate(
                    input_ids,
                    assistant_model=self.draft_model,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=999999,
                )
                
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            num_tokens = output_ids.shape[1] - input_ids.shape[1]
            throughput = num_tokens / (total_time / 1000)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_tokens.append(num_tokens)
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms, tokens={num_tokens}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.total_tokens_generated = int(sum(all_tokens) / len(all_tokens))
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nHF Assisted Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_linear_spec_decode(self, K: int = 6) -> BenchmarkMetrics:
        """Benchmark linear speculative decoding."""
        print("\n" + "="*60)
        print(f"Benchmarking: Linear Speculative Decoding (K={K})")
        print("="*60)
        
        from spec_decode.core.speculative_generator import SpeculativeGenerator
        
        metrics = BenchmarkMetrics(method=f"Linear K={K}", config={"K": K})
        
        original_eos = self._disable_eos()
        
        generator = SpeculativeGenerator(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            K=K,
            device=self.device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_tokens_per_round = []
        all_rounds = []
        all_ttft = []
        all_tpot = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT
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
                all_tokens_per_round.append(stats.get('tokens_per_round', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms, accept={stats.get('acceptance_rate', 0):.2%}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.tokens_per_round = sum(all_tokens_per_round) / len(all_tokens_per_round)
        metrics.total_rounds = int(sum(all_rounds) / len(all_rounds))
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        # FLOPs estimation
        avg_prompt_len = 512
        metrics.prefill_flops = self.flops_estimator.estimate_prefill_flops(avg_prompt_len)
        metrics.decode_flops = self.flops_estimator.estimate_spec_decode_flops(
            metrics.total_tokens_generated, K, metrics.total_rounds, avg_prompt_len
        )
        metrics.total_flops = metrics.prefill_flops + metrics.decode_flops
        metrics.flops_per_token = metrics.decode_flops / metrics.total_tokens_generated
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nLinear K={K} Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Acceptance Rate: {metrics.acceptance_rate:.2%}")
        print(f"  Tokens/Round: {metrics.tokens_per_round:.1f}")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_streaming_spec_decode(self, K: int = 6, max_cache_len: int = 1024) -> BenchmarkMetrics:
        """Benchmark streaming speculative decoding."""
        print("\n" + "="*60)
        print(f"Benchmarking: Streaming Speculative Decoding (K={K}, cache={max_cache_len})")
        print("="*60)
        
        from spec_decode.core.streaming_speculative_generator import StreamingSpeculativeGenerator
        
        metrics = BenchmarkMetrics(
            method=f"Streaming K={K} cache={max_cache_len}",
            config={"K": K, "max_cache_len": max_cache_len}
        )
        
        original_eos = self._disable_eos()
        
        generator = StreamingSpeculativeGenerator(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            K=K,
            max_cache_len=max_cache_len,
            device=self.device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_ttft = []
        all_tpot = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT
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
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nStreaming K={K} cache={max_cache_len} Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_tree_v2(
        self,
        tree_depth: int = 8,
        branch_factor: int = 3,
        probability_threshold: float = 0.03
    ) -> BenchmarkMetrics:
        """Benchmark Tree V2 speculative decoding."""
        print("\n" + "="*60)
        print(f"Benchmarking: Tree V2 (D={tree_depth}, B={branch_factor}, t={probability_threshold})")
        print("="*60)
        
        from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
        
        metrics = BenchmarkMetrics(
            method=f"Tree V2 (D={tree_depth}, B={branch_factor}, t={probability_threshold})",
            config={
                "tree_depth": tree_depth,
                "branch_factor": branch_factor,
                "probability_threshold": probability_threshold
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
            device=self.device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_path_lengths = []
        all_rounds = []
        all_ttft = []
        all_tpot = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT
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
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms, accept={stats.get('acceptance_rate', 0):.2%}, path_len={stats.get('avg_accepted_path_length', 0):.1f}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.total_rounds = int(sum(all_rounds) / len(all_rounds))
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        # FLOPs estimation
        avg_prompt_len = 512
        metrics.prefill_flops = self.flops_estimator.estimate_prefill_flops(avg_prompt_len)
        metrics.decode_flops = self.flops_estimator.estimate_tree_decode_flops(
            metrics.total_tokens_generated, tree_depth, branch_factor, 
            metrics.total_rounds, avg_prompt_len
        )
        metrics.total_flops = metrics.prefill_flops + metrics.decode_flops
        metrics.flops_per_token = metrics.decode_flops / metrics.total_tokens_generated
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nTree V2 Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Acceptance Rate: {metrics.acceptance_rate:.2%}")
        print(f"  Avg Path Length: {metrics.avg_path_length:.1f}")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    def compare_acceptance_rates(self) -> Dict[str, Any]:
        """Compare token acceptance rates between Tree and Linear methods."""
        print("\n" + "="*60)
        print("Acceptance Rate Comparison: Tree vs Linear")
        print("="*60)
        
        comparison = {
            "linear_methods": [],
            "tree_methods": [],
            "analysis": {}
        }
        
        for metrics in self.results:
            if metrics.method.startswith("Linear"):
                comparison["linear_methods"].append({
                    "method": metrics.method,
                    "acceptance_rate": metrics.acceptance_rate,
                    "tokens_per_round": metrics.tokens_per_round,
                    "total_rounds": metrics.total_rounds,
                })
                print(f"\n  {metrics.method}: acceptance={metrics.acceptance_rate:.2%}, tokens/round={metrics.tokens_per_round:.1f}")
            elif metrics.method.startswith("Tree"):
                comparison["tree_methods"].append({
                    "method": metrics.method,
                    "acceptance_rate": metrics.acceptance_rate,
                    "avg_path_length": metrics.avg_path_length,
                    "total_rounds": metrics.total_rounds,
                })
                print(f"\n  {metrics.method}: acceptance={metrics.acceptance_rate:.2%}, path_len={metrics.avg_path_length:.1f}")
        
        # Analysis
        if comparison["linear_methods"]:
            avg_linear_accept = sum(m["acceptance_rate"] for m in comparison["linear_methods"]) / len(comparison["linear_methods"])
            avg_linear_tokens = sum(m["tokens_per_round"] for m in comparison["linear_methods"]) / len(comparison["linear_methods"])
            comparison["analysis"]["avg_linear_acceptance"] = avg_linear_accept
            comparison["analysis"]["avg_linear_tokens_per_round"] = avg_linear_tokens
        
        if comparison["tree_methods"]:
            avg_tree_accept = sum(m["acceptance_rate"] for m in comparison["tree_methods"]) / len(comparison["tree_methods"])
            avg_tree_path = sum(m["avg_path_length"] for m in comparison["tree_methods"]) / len(comparison["tree_methods"])
            comparison["analysis"]["avg_tree_acceptance"] = avg_tree_accept
            comparison["analysis"]["avg_tree_path_length"] = avg_tree_path
        
        print(f"\nSummary:")
        if "avg_linear_acceptance" in comparison["analysis"]:
            print(f"  Avg Linear Acceptance: {comparison['analysis']['avg_linear_acceptance']:.2%}")
            print(f"  Avg Linear Tokens/Round: {comparison['analysis']['avg_linear_tokens_per_round']:.1f}")
        if "avg_tree_acceptance" in comparison["analysis"]:
            print(f"  Avg Tree Acceptance: {comparison['analysis']['avg_tree_acceptance']:.2%}")
            print(f"  Avg Tree Path Length: {comparison['analysis']['avg_tree_path_length']:.1f}")
        
        return comparison
    
    def run_all_benchmarks(self):
        """Run all benchmark configurations."""
        print("\n" + "#"*70)
        print("# Comprehensive Speculative Decoding Benchmark (PG19)")
        print(f"# Target Model: {self.target_model_path}")
        print(f"# Draft Model: {self.draft_model_path}")
        print(f"# Max New Tokens: {self.max_new_tokens}")
        print(f"# Samples: {len(self.prompts)}")
        print("#"*70)
        
        # Show prompts being used
        print("\nPrompts being used:")
        for i, prompt in enumerate(self.prompts):
            preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            print(f"  {i+1}. {preview}")
        
        # 1. Baseline first to establish reference
        self.benchmark_baseline()
        
        # 2. Tree V2 configurations (from parameter search results)
        tree_configs = [
            {"tree_depth": 4, "branch_factor": 2, "probability_threshold": 0.05},  # D=4 B=2 t=0.05
            {"tree_depth": 4, "branch_factor": 2, "probability_threshold": 0.03},  # D=4 B=2 t=0.03
            {"tree_depth": 5, "branch_factor": 2, "probability_threshold": 0.05},  # D=5 B=2 t=0.05
            {"tree_depth": 7, "branch_factor": 2, "probability_threshold": 0.05},  # D=7 B=2 t=0.05
            {"tree_depth": 6, "branch_factor": 2, "probability_threshold": 0.05},  # D=6 B=2 t=0.05
        ]
        for config in tree_configs:
            self.benchmark_tree_v2(**config)
        
        # 3. HuggingFace Assisted
        self.benchmark_hf_assisted()
        
        # 4. Linear variants (K roughly matches tree depth for comparison)
        for K in [4, 5, 6, 7]:
            self.benchmark_linear_spec_decode(K=K)
        
        # 5. Streaming variants
        self.benchmark_streaming_spec_decode(K=5, max_cache_len=1024)
        self.benchmark_streaming_spec_decode(K=6, max_cache_len=512)
        
        # Compare acceptance rates
        self.compare_acceptance_rates()
    
    def _generate_summary(self, acceptance_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            "benchmark_info": {
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "dataset": "PG19 (Project Gutenberg)",
                "data_path": self.data_path,
                "max_new_tokens": self.max_new_tokens,
                "num_samples": len(self.prompts),
                "warmup_runs": self.warmup_runs,
                "max_prompt_length": self.max_prompt_length,
            },
            "results": [asdict(m) for m in self.results],
            "acceptance_comparison": acceptance_comparison,
            "rankings": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Generate rankings
        sorted_by_throughput = sorted(self.results, key=lambda x: x.throughput_tps, reverse=True)
        summary["rankings"]["by_throughput"] = [
            {"rank": i+1, "method": m.method, "throughput": m.throughput_tps, "speedup": m.speedup}
            for i, m in enumerate(sorted_by_throughput)
        ]
        
        sorted_by_acceptance = sorted(
            [m for m in self.results if m.acceptance_rate > 0],
            key=lambda x: x.acceptance_rate, reverse=True
        )
        summary["rankings"]["by_acceptance_rate"] = [
            {"rank": i+1, "method": m.method, "acceptance_rate": m.acceptance_rate}
            for i, m in enumerate(sorted_by_acceptance)
        ]
        
        # Print summary table
        print("\n" + "="*110)
        print("FINAL RESULTS SUMMARY (PG19 Dataset)")
        print("="*110)
        print(f"\n{'Rank':<5} {'Method':<35} {'Throughput':<12} {'Speedup':<10} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'Accept%':<10}")
        print("-"*110)
        for i, m in enumerate(sorted_by_throughput):
            accept_str = f"{m.acceptance_rate:.1%}" if m.acceptance_rate > 0 else "N/A"
            print(f"{i+1:<5} {m.method:<35} {m.throughput_tps:>8.1f} t/s {m.speedup:>7.2f}x {m.ttft_ms:>8.1f} {m.tpot_ms:>8.2f} {accept_str:>10}")
        
        print("\n" + "="*80)
        print("FLOPs COMPARISON")
        print("="*80)
        print(f"\n{'Method':<35} {'Total FLOPs':<15} {'FLOPs/Token':<15} {'Memory(MB)':<12}")
        print("-"*80)
        for m in sorted_by_throughput:
            if m.total_flops > 0:
                print(f"{m.method:<35} {m.total_flops:>12.2e} {m.flops_per_token:>12.2e} {m.peak_memory_mb:>10.0f}")
        
        return summary
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        summary = self._generate_summary(self.compare_acceptance_rates())
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Speculative Decoding Benchmark (PG19)")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=800,
                        help="Maximum prompt length in characters (default: 800)")
    parser.add_argument("--min-prompt-length", type=int, default=200,
                        help="Minimum prompt length in characters (default: 200)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to pg19.parquet file (default: data/pg19.parquet)")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced samples")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 2
        args.warmup_runs = 1
        args.max_new_tokens = 200
    
    benchmark = PG19Benchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        max_prompt_length=args.max_prompt_length,
        min_prompt_length=args.min_prompt_length,
        data_path=args.data_path,
    )
    
    benchmark.run_all_benchmarks()
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/pg19_benchmark_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()


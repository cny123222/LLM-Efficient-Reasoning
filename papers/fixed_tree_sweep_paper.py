#!/usr/bin/env python3
"""
Fixed-tree Hyperparameter Sweep for Paper Experiments

This script systematically searches for the optimal Fixed Tree V2 configuration
using prompts from WikiText-2 dataset (ModelScope) under the paper protocol.

Paper Protocol Settings:
- Dataset: WikiText-2
- L_max (prompt length): 800 chars
- T (max_new_tokens): 1500
- N (num_prompts): 10
- warmup: 2

Parameters searched:
- D (depth): 3, 4, 5, 6, 7, 8
- B (branch factor): 2, 3, 4
- τ (threshold): 0.01, 0.02, 0.03, 0.05, 0.1

Total configurations: 6 × 3 × 5 = 90

Usage:
    python fixed_tree_sweep_paper.py \
        --target-model /mnt/disk1/models/pythia-2.8b \
        --draft-model /mnt/disk1/models/pythia-70m \
        --output-dir results/adaptive/fixed_tree_sweep \
        --num-prompts 10 \
        --prompt-length 800 \
        --num-runs 2

Output:
    - JSON results: fixed_tree_sweep_{timestamp}.json
    - Heatmap figure: tree_config_heatmap.pdf
    - Comparison figure: tree_config_comparison.pdf
"""

import os
import sys
import json
import time
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Optional seaborn for prettier heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

logging.set_verbosity_error()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_decode.core import TreeSpeculativeGeneratorV2


# =============================================================================
# Configuration - Updated for Paper Protocol
# =============================================================================
DEFAULT_DEPTHS = [3, 4, 5, 6, 7, 8]
DEFAULT_BRANCHES = [2, 3, 4]
DEFAULT_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.1]
DEFAULT_TOKEN_LENGTH = 1500  # Paper protocol: T=1500

# Paper protocol settings
DEFAULT_PROMPT_LENGTH = 800   # L_max = 800
DEFAULT_NUM_PROMPTS = 10      # N = 10
DEFAULT_NUM_RUNS = 2          # warmup = 2


# =============================================================================
# WikiText Dataset Loading
# =============================================================================
def load_wikitext_prompts(
    num_prompts: int = 10,
    min_length: int = 200,
    max_length: int = 800
) -> List[str]:
    """
    Load prompts from WikiText dataset via ModelScope.
    
    Args:
        num_prompts: Number of prompts to load
        min_length: Minimum prompt length in characters
        max_length: Maximum prompt length (will truncate)
        
    Returns:
        List of prompt strings
    """
    print(f"Loading WikiText prompts from ModelScope...")
    print(f"  Target: {num_prompts} prompts, {min_length}-{max_length} chars each")
    
    try:
        from modelscope.msdatasets import MsDataset
        
        # Load the wikitext dataset
        dataset = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='validation')
        
        prompts = []
        current_text = ""
        
        # Iterate through the dataset and collect good prompts
        for item in dataset:
            text = item.get('text', '')
            if not text or text.strip() == '' or text.startswith(' ='):
                # Skip empty lines and section headers, save accumulated text
                if current_text and len(current_text) >= min_length:
                    prompts.append(current_text.strip())
                    current_text = ""
                    if len(prompts) >= num_prompts * 3:  # Get extra for filtering
                        break
            else:
                current_text += " " + text
        
        # Add last accumulated text if valid
        if current_text and len(current_text) >= min_length:
            prompts.append(current_text.strip())
        
        # Filter and select prompts
        valid_prompts = []
        for prompt in prompts:
            prompt = prompt.strip()
            if len(prompt) >= min_length:
                # Truncate to max_length
                if len(prompt) > max_length:
                    prompt = prompt[:max_length]
                valid_prompts.append(prompt)
                if len(valid_prompts) >= num_prompts:
                    break
        
        if len(valid_prompts) < num_prompts:
            print(f"  Warning: Only found {len(valid_prompts)} valid prompts")
            # Duplicate if needed
            while len(valid_prompts) < num_prompts:
                valid_prompts.append(valid_prompts[len(valid_prompts) % len(valid_prompts) if valid_prompts else 0])
        
        # Print prompt statistics
        lengths = [len(p) for p in valid_prompts]
        print(f"  Loaded {len(valid_prompts)} prompts")
        print(f"  Prompt lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.0f} chars")
        
        return valid_prompts[:num_prompts]
        
    except ImportError:
        print("  Warning: modelscope not installed, using fallback prompts")
        return get_fallback_prompts(num_prompts, max_length)
    except Exception as e:
        print(f"  Warning: Failed to load WikiText: {e}")
        return get_fallback_prompts(num_prompts, max_length)


def get_fallback_prompts(num_prompts: int, max_length: int = 800) -> List[str]:
    """Fallback prompts if WikiText loading fails."""
    fallback_prompts = [
        """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.""",
        
        """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.""",
        
        """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing is used in many applications including machine translation.""",
        
        """Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, and medical image analysis.""",
        
        """The transformer is a deep learning model introduced in 2017 that utilizes the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing and computer vision. Like recurrent neural networks, transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. Unlike RNNs, transformers process the entire input all at once.""",
        
        """Large language models are neural network models that are trained on massive amounts of text data. These models can generate human-like text and perform various natural language tasks. The development of large language models has revolutionized the field of artificial intelligence, enabling new applications in chatbots, content generation, code completion, and many other areas. Recent advances include models with billions of parameters trained on internet-scale datasets.""",
        
        """Speculative decoding is an inference optimization technique for large language models that uses a smaller draft model to generate candidate tokens, which are then verified by the larger target model. This approach can significantly accelerate inference by enabling parallel verification of multiple tokens while maintaining the same output distribution as standard autoregressive decoding. The key insight is that small models can often predict what larger models will generate.""",
        
        """The attention mechanism allows neural networks to focus on specific parts of the input when generating each part of the output. In the transformer architecture, self-attention enables the model to weigh the importance of different positions in a sequence when encoding each position. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. This mechanism has become fundamental to modern NLP.""",
        
        """Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although practical ties between the two fields are limited.""",
        
        """Reinforcement learning from human feedback is a machine learning technique that trains models to align with human preferences. The process typically involves collecting human feedback on model outputs, training a reward model to predict human preferences, and then fine-tuning the language model to maximize the predicted reward. This approach has been instrumental in improving the helpfulness and safety of conversational AI systems.""",
    ]
    
    # Truncate to max_length
    prompts = [p[:max_length] for p in fallback_prompts[:num_prompts]]
    
    # Duplicate if needed
    while len(prompts) < num_prompts:
        prompts.append(prompts[len(prompts) % len(prompts)])
    
    return prompts[:num_prompts]


# =============================================================================
# Utility Functions
# =============================================================================
def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# Measurement Functions
# =============================================================================
def measure_baseline(
    target_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
    num_runs: int = 2
) -> Tuple[float, float, float]:
    """
    Measure baseline autoregressive throughput.
    
    Returns:
        throughput: tokens/sec
        ttft: time to first token (seconds)
        tpot: time per output token (seconds)
    """
    cleanup()
    
    # Disable EOS for forced long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    throughputs = []
    ttfts = []
    tpots = []
    
    try:
        for run_idx in range(num_runs):
            # Use different prompts for different runs
            prompt = prompts[run_idx % len(prompts)]
            
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.inference_mode():
                outputs = target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            generated_tokens = outputs.shape[1] - input_ids.shape[1]
            throughputs.append(generated_tokens / elapsed)
            
            # Approximate TTFT and TPOT for AR decoding
            ttft = elapsed / generated_tokens  # First token takes same time as others in AR
            tpot = elapsed / generated_tokens
            ttfts.append(ttft)
            tpots.append(tpot)
            
    finally:
        tokenizer.eos_token_id = original_eos
    
    return np.mean(throughputs), np.mean(ttfts), np.mean(tpots)


def measure_tree_v2(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    depth: int,
    branch: int,
    threshold: float,
    device: str,
    num_runs: int = 2
) -> Dict:
    """
    Measure Tree V2 performance with detailed metrics.
    
    Returns:
        Dictionary with throughput, speedup, path_length, acceptance_rate, ttft, tpot, total_rounds
    """
    cleanup()
    
    # Disable EOS for forced long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        generator = TreeSpeculativeGeneratorV2(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=depth,
            branch_factor=branch,
            probability_threshold=threshold,
            max_tree_nodes=128,
            device=device,
            use_compile=False
        )
        
        throughputs = []
        path_lengths = []
        acceptance_rates = []
        total_rounds_list = []
        ttfts = []
        tpots = []
        
        for run_idx in range(num_runs):
            # Use different prompts for different runs
            prompt = prompts[run_idx % len(prompts)]
            
            generator.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            total_tokens = stats['total_tokens']
            total_rounds = stats['total_rounds']
            
            throughputs.append(total_tokens / elapsed)
            path_lengths.append(stats.get('avg_accepted_path_length', 0))
            acceptance_rates.append(stats.get('acceptance_rate', 0))
            total_rounds_list.append(total_rounds)
            
            # Compute TTFT and TPOT
            # TTFT: time for first round (approximated)
            ttft = elapsed / total_rounds if total_rounds > 0 else elapsed
            # TPOT: average time per output token
            tpot = elapsed / total_tokens if total_tokens > 0 else 0
            ttfts.append(ttft)
            tpots.append(tpot)
        
        return {
            'throughput': np.mean(throughputs),
            'avg_path_length': np.mean(path_lengths),
            'acceptance_rate': np.mean(acceptance_rates),
            'total_rounds': np.mean(total_rounds_list),
            'ttft': np.mean(ttfts),
            'tpot': np.mean(tpots),
            'success': True
        }
        
    except Exception as e:
        print(f"    ERROR: {str(e)[:50]}")
        return {
            'throughput': 0.0,
            'avg_path_length': 0.0,
            'acceptance_rate': 0.0,
            'total_rounds': 0,
            'ttft': 0.0,
            'tpot': 0.0,
            'success': False,
            'error': str(e)
        }
    finally:
        tokenizer.eos_token_id = original_eos


# =============================================================================
# Search Functions
# =============================================================================
def search_tree_params(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
    depths: List[int] = DEFAULT_DEPTHS,
    branches: List[int] = DEFAULT_BRANCHES,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
    num_runs: int = 2
) -> Tuple[List[Dict], float]:
    """
    Systematically search Tree V2 parameters.
    
    Returns:
        Tuple of (results list, baseline_throughput)
    """
    total_configs = len(depths) * len(branches) * len(thresholds)
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"Estimated time: {format_time(total_configs * 180)}")  # ~3 min per config with T=1500
    
    # Measure baseline first
    print(f"\n{'='*70}")
    print(f"Measuring baseline (AR decoding) with T={max_new_tokens} tokens...")
    print(f"{'='*70}")
    
    baseline_tp, baseline_ttft, baseline_tpot = measure_baseline(
        target_model, tokenizer, prompts, max_new_tokens, device, num_runs
    )
    print(f"  Baseline throughput: {baseline_tp:.1f} t/s")
    print(f"  Baseline TTFT: {baseline_ttft*1000:.1f} ms")
    print(f"  Baseline TPOT: {baseline_tpot*1000:.2f} ms")
    
    results = []
    config_idx = 0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"Testing Tree V2 configurations with T={max_new_tokens} tokens")
    print(f"{'='*70}")
    
    for D in depths:
        for B in branches:
            for t in thresholds:
                config_idx += 1
                elapsed = time.time() - start_time
                eta = (elapsed / config_idx) * (total_configs - config_idx) if config_idx > 0 else 0
                
                print(f"  [{config_idx}/{total_configs}] D={D} B={B} τ={t:.2f}", end=" ")
                print(f"(ETA: {format_time(eta)})", end=" ")
                
                metrics = measure_tree_v2(
                    target_model, draft_model, tokenizer,
                    prompts, max_new_tokens, D, B, t, device, num_runs
                )
                
                speedup = metrics['throughput'] / baseline_tp if baseline_tp > 0 else 0
                
                result = {
                    'depth': D,
                    'branch': B,
                    'threshold': t,
                    'throughput': metrics['throughput'],
                    'speedup': speedup,
                    'avg_path_length': metrics['avg_path_length'],
                    'acceptance_rate': metrics['acceptance_rate'],
                    'total_rounds': metrics['total_rounds'],
                    'ttft': metrics['ttft'],
                    'tpot': metrics['tpot'],
                    'baseline_throughput': baseline_tp,
                    'success': metrics['success']
                }
                results.append(result)
                
                if metrics['success']:
                    print(f"-> {metrics['throughput']:.1f} t/s ({speedup:.2f}x)")
                else:
                    print(f"-> FAILED")
    
    return results, baseline_tp


def find_best_config(results: List[Dict]) -> Dict:
    """Find the best configuration from search results."""
    valid_results = [r for r in results if r.get('success', True) and r['throughput'] > 0]
    if not valid_results:
        return {}
    
    # Find best by speedup
    best = max(valid_results, key=lambda x: x['speedup'])
    return best


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_heatmap(results: List[Dict], output_path: str):
    """Generate heatmap visualization of parameter search results."""
    
    # Find the best branch factor
    best_branch = 2
    best_speedup = 0
    for b in DEFAULT_BRANCHES:
        filtered = [r for r in results if r['branch'] == b and r.get('success', True)]
        if filtered:
            max_speedup = max(r['speedup'] for r in filtered)
            if max_speedup > best_speedup:
                best_speedup = max_speedup
                best_branch = b
    
    # Use the best branch factor for heatmap
    filtered = [r for r in results if r['branch'] == best_branch and r.get('success', True)]
    
    if not filtered:
        print("No valid results for heatmap")
        return
    
    # Create matrix for heatmap (D x threshold)
    depths = sorted(set(r['depth'] for r in filtered))
    thresholds = sorted(set(r['threshold'] for r in filtered))
    
    matrix = np.zeros((len(depths), len(thresholds)))
    for r in filtered:
        i = depths.index(r['depth'])
        j = thresholds.index(r['threshold'])
        matrix[i, j] = r['speedup']
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(
            matrix, ax=ax, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=[f'{t:.2f}' for t in thresholds],
            yticklabels=depths,
            cbar_kws={'label': 'Speedup over AR baseline'},
            annot_kws={'fontsize': 10}
        )
    else:
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths)
        
        # Add text annotations
        for i in range(len(depths)):
            for j in range(len(thresholds)):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Speedup over AR baseline')
    
    ax.set_xlabel('Pruning Threshold (τ)', fontsize=12)
    ax.set_ylabel('Tree Depth (D)', fontsize=12)
    ax.set_title(f'Fixed Tree V2 Speedup Heatmap (B={best_branch}, T=1500, WikiText-2)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap saved to: {output_path}")


def plot_comparison(results: List[Dict], output_path: str):
    """Generate comparison bar chart for different configurations."""
    
    # Find top 10 configurations by speedup
    valid_results = [r for r in results if r.get('success', True) and r['throughput'] > 0]
    valid_results.sort(key=lambda x: x['speedup'], reverse=True)
    top_results = valid_results[:10]
    
    if not top_results:
        print("No valid results for comparison")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Labels for each config
    labels = [f"D={r['depth']},B={r['branch']},τ={r['threshold']:.2f}" for r in top_results]
    
    # Subplot 1: Throughput comparison
    throughputs = [r['throughput'] for r in top_results]
    baseline = top_results[0]['baseline_throughput'] if top_results else 0
    
    bars1 = ax1.bar(range(len(top_results)), throughputs, color='steelblue', label='Tree V2')
    ax1.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'AR Baseline ({baseline:.1f} t/s)')
    ax1.set_xticks(range(len(top_results)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=11)
    ax1.set_title('Top 10 Configurations by Throughput', fontsize=12)
    ax1.legend()
    
    # Add speedup labels on bars
    for i, bar in enumerate(bars1):
        speedup = top_results[i]['speedup']
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Acceptance rate and path length
    x = np.arange(len(top_results))
    width = 0.35
    
    acceptance_rates = [r['acceptance_rate'] * 100 for r in top_results]  # Convert to percentage
    path_lengths = [r['avg_path_length'] for r in top_results]
    
    bars2a = ax2.bar(x - width/2, acceptance_rates, width, label='Accept Rate (%)', color='forestgreen')
    ax2_twin = ax2.twinx()
    bars2b = ax2_twin.bar(x + width/2, path_lengths, width, label='Avg Path Length', color='orange')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=11, color='forestgreen')
    ax2_twin.set_ylabel('Avg Path Length', fontsize=11, color='orange')
    ax2.set_title('Verification Efficiency Metrics', fontsize=12)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.suptitle('Fixed Tree V2 Configuration Comparison (WikiText-2, T=1500)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison chart saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fixed Tree V2 Hyperparameter Sweep (Paper Protocol)")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                       help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                       help="Path to draft model")
    parser.add_argument("--output-dir", type=str, default="results/adaptive/fixed_tree_sweep",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                       help=f"Number of runs per configuration (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS,
                       help=f"Number of prompts to use from WikiText (default: {DEFAULT_NUM_PROMPTS})")
    parser.add_argument("--prompt-length", type=int, default=DEFAULT_PROMPT_LENGTH,
                       help=f"Maximum prompt length in characters (default: {DEFAULT_PROMPT_LENGTH})")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_TOKEN_LENGTH,
                       help=f"Number of tokens to generate (default: {DEFAULT_TOKEN_LENGTH})")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer configurations for testing")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("Fixed Tree V2 Hyperparameter Sweep (Paper Protocol)")
    print("=" * 70)
    print(f"\nPaper Protocol Settings:")
    print(f"  Dataset: WikiText-2")
    print(f"  L_max (prompt length): {args.prompt_length} chars")
    print(f"  T (max_new_tokens): {args.max_new_tokens}")
    print(f"  N (num_prompts): {args.num_prompts}")
    print(f"  warmup/runs: {args.num_runs}")
    print(f"\nModel Configuration:")
    print(f"  Target Model: {args.target_model}")
    print(f"  Draft Model: {args.draft_model}")
    print(f"  Device: {args.device}")
    
    # Load WikiText prompts
    prompts = load_wikitext_prompts(
        num_prompts=args.num_prompts,
        min_length=100,
        max_length=args.prompt_length
    )
    
    # Show prompts
    print(f"\nPrompts loaded ({len(prompts)}):")
    for i, p in enumerate(prompts):
        preview = p[:80] + "..." if len(p) > 80 else p
        print(f"  {i+1}. [{len(p)} chars] {preview}")
    
    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    target_model.eval()
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    draft_model.eval()
    
    print(f"  Memory usage: {get_memory_mb():.0f} MB")
    
    # Configure search parameters
    if args.quick:
        depths = [4, 5, 6]
        branches = [2, 3]
        thresholds = [0.02, 0.05]
        print("\n[QUICK MODE] Using reduced configuration grid")
    else:
        depths = DEFAULT_DEPTHS
        branches = DEFAULT_BRANCHES
        thresholds = DEFAULT_THRESHOLDS
    
    print(f"\nSearch Grid:")
    print(f"  Depths (D): {depths}")
    print(f"  Branches (B): {branches}")
    print(f"  Thresholds (τ): {thresholds}")
    print(f"  Total configurations: {len(depths) * len(branches) * len(thresholds)}")
    
    # Run search
    print("\nStarting parameter search...")
    results, baseline_throughput = search_tree_params(
        target_model, draft_model, tokenizer, prompts, args.device,
        args.max_new_tokens, depths, branches, thresholds, args.num_runs
    )
    
    # Find best configuration
    best = find_best_config(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SEARCH RESULTS SUMMARY")
    print("=" * 70)
    
    # Group by branch factor
    for B in sorted(set(r['branch'] for r in results)):
        branch_results = [r for r in results if r['branch'] == B and r.get('success', True)]
        if branch_results:
            best_for_branch = max(branch_results, key=lambda x: x['speedup'])
            print(f"\n  Branch factor B={B}:")
            print(f"    Best config: D={best_for_branch['depth']}, τ={best_for_branch['threshold']:.2f}")
            print(f"    Throughput: {best_for_branch['throughput']:.1f} t/s ({best_for_branch['speedup']:.2f}x)")
            print(f"    Accept rate: {best_for_branch['acceptance_rate']:.2%}")
            print(f"    Avg path length: {best_for_branch['avg_path_length']:.2f}")
    
    if best:
        print(f"\n{'='*70}")
        print(f"OVERALL BEST CONFIGURATION:")
        print(f"{'='*70}")
        print(f"  Depth (D): {best['depth']}")
        print(f"  Branch (B): {best['branch']}")
        print(f"  Threshold (τ): {best['threshold']}")
        print(f"  Throughput: {best['throughput']:.1f} t/s")
        print(f"  Speedup: {best['speedup']:.2f}x over AR baseline")
        print(f"  Acceptance Rate: {best['acceptance_rate']:.2%}")
        print(f"  Avg Path Length: {best['avg_path_length']:.2f}")
        print(f"  Total Rounds: {best['total_rounds']:.0f}")
        print(f"  TPOT: {best['tpot']*1000:.2f} ms")
    
    # Save results
    prompt_info = [{"length": len(p), "preview": p[:100]} for p in prompts]
    
    output_data = {
        'config': {
            'target_model': args.target_model,
            'draft_model': args.draft_model,
            'dataset': 'WikiText-2 (ModelScope)',
            'protocol': 'Paper Protocol',
            'depths': depths,
            'branches': branches,
            'thresholds': thresholds,
            'max_new_tokens': args.max_new_tokens,
            'num_runs': args.num_runs,
            'num_prompts': args.num_prompts,
            'prompt_length': args.prompt_length,
            'timestamp': timestamp
        },
        'baseline': {
            'throughput': baseline_throughput
        },
        'prompts': prompt_info,
        'results': results,
        'best_overall': best
    }
    
    json_path = os.path.join(args.output_dir, f"fixed_tree_sweep_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate visualizations
    heatmap_path = os.path.join(args.output_dir, "tree_config_heatmap.pdf")
    plot_heatmap(results, heatmap_path)
    
    comparison_path = os.path.join(args.output_dir, "tree_config_comparison.pdf")
    plot_comparison(results, comparison_path)
    
    print("\n" + "=" * 70)
    print("Parameter sweep complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - JSON: {json_path}")
    print(f"  - Heatmap: {heatmap_path}")
    print(f"  - Comparison: {comparison_path}")


if __name__ == "__main__":
    main()


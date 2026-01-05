#!/usr/bin/env python3
"""
Tree-based Speculative Decoding Parameter Search using WikiText Dataset

This script systematically searches for the optimal Tree V2 configuration
using prompts from WikiText-2 dataset (ModelScope).

Parameters searched:
- D (depth): 3, 4, 5, 6, 7, 8
- B (branch factor): 2, 3, 4
- t (threshold): 0.01, 0.02, 0.03, 0.05, 0.1
- tokens: 100, 200, 300, 500, 1000

Usage:
    python tree_param_search_wikitext.py \
        --target-model /mnt/disk1/models/pythia-2.8b \
        --draft-model /mnt/disk1/models/pythia-70m \
        --output-dir results \
        --num-prompts 5 \
        --prompt-length 500
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
# Configuration
# =============================================================================
DEFAULT_DEPTHS = [3, 4, 5, 6, 7, 8]
DEFAULT_BRANCHES = [2, 3, 4]
DEFAULT_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.1]
DEFAULT_TOKEN_LENGTHS = [100, 200, 300, 500, 1000]


# =============================================================================
# WikiText Dataset Loading
# =============================================================================
def load_wikitext_prompts(
    num_prompts: int = 5,
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
) -> float:
    """Measure baseline autoregressive throughput."""
    cleanup()
    
    # Disable EOS for forced long generation
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    throughputs = []
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
    finally:
        tokenizer.eos_token_id = original_eos
    
    return np.mean(throughputs)


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
) -> Tuple[float, float, float]:
    """
    Measure Tree V2 performance.
    
    Returns:
        throughput: tokens/sec
        avg_path_length: average accepted path length
        acceptance_rate: acceptance rate
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
            throughputs.append(stats['total_tokens'] / elapsed)
            path_lengths.append(stats.get('avg_accepted_path_length', 0))
            acceptance_rates.append(stats.get('acceptance_rate', 0))
        
        return (
            np.mean(throughputs),
            np.mean(path_lengths),
            np.mean(acceptance_rates)
        )
    except Exception as e:
        print(f"    ERROR: {str(e)[:50]}")
        return 0.0, 0.0, 0.0
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
    depths: List[int] = DEFAULT_DEPTHS,
    branches: List[int] = DEFAULT_BRANCHES,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
    token_lengths: List[int] = DEFAULT_TOKEN_LENGTHS,
    num_runs: int = 2
) -> List[Dict]:
    """
    Systematically search Tree V2 parameters.
    
    Returns:
        List of result dictionaries with all configurations tested.
    """
    total_configs = len(depths) * len(branches) * len(thresholds) * len(token_lengths)
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"Estimated time: {format_time(total_configs * 5)}")  # ~5s per config with WikiText
    
    results = []
    baselines = {}  # Cache baselines by token length
    config_idx = 0
    start_time = time.time()
    
    for tokens in token_lengths:
        print(f"\n{'='*70}")
        print(f"Testing with {tokens} tokens")
        print(f"{'='*70}")
        
        # Measure baseline for this token length
        if tokens not in baselines:
            print(f"  Measuring baseline...")
            baselines[tokens] = measure_baseline(
                target_model, tokenizer, prompts, tokens, device, num_runs
            )
            print(f"  Baseline: {baselines[tokens]:.1f} t/s")
        
        baseline_tp = baselines[tokens]
        
        for D in depths:
            for B in branches:
                for t in thresholds:
                    config_idx += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / config_idx) * (total_configs - config_idx)
                    
                    print(f"  [{config_idx}/{total_configs}] D={D} B={B} t={t:.2f}", end=" ")
                    print(f"(ETA: {format_time(eta)})", end=" ")
                    
                    tp, path_len, acc_rate = measure_tree_v2(
                        target_model, draft_model, tokenizer,
                        prompts, tokens, D, B, t, device, num_runs
                    )
                    
                    speedup = tp / baseline_tp if baseline_tp > 0 else 0
                    
                    result = {
                        'tokens': tokens,
                        'depth': D,
                        'branch': B,
                        'threshold': t,
                        'throughput': tp,
                        'speedup': speedup,
                        'avg_path_length': path_len,
                        'acceptance_rate': acc_rate,
                        'baseline_throughput': baseline_tp
                    }
                    results.append(result)
                    
                    if tp > 0:
                        print(f"-> {tp:.1f} t/s ({speedup:.2f}x)")
                    else:
                        print(f"-> FAILED")
    
    return results


def find_best_config(results: List[Dict]) -> Dict:
    """Find the best configuration from search results."""
    valid_results = [r for r in results if r['throughput'] > 0]
    if not valid_results:
        return {}
    
    # Find best by speedup
    best = max(valid_results, key=lambda x: x['speedup'])
    return best


def find_best_per_token_length(results: List[Dict]) -> Dict[int, Dict]:
    """Find best configuration for each token length."""
    best_per_length = {}
    
    for result in results:
        tokens = result['tokens']
        if tokens not in best_per_length or result['speedup'] > best_per_length[tokens]['speedup']:
            best_per_length[tokens] = result
    
    return best_per_length


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_heatmaps(results: List[Dict], output_path: str):
    """Generate heatmap visualizations of parameter search results."""
    # Group results by token length
    token_lengths = sorted(set(r['tokens'] for r in results))
    
    fig, axes = plt.subplots(1, len(token_lengths), figsize=(5*len(token_lengths), 5))
    if len(token_lengths) == 1:
        axes = [axes]
    
    for idx, tokens in enumerate(token_lengths):
        ax = axes[idx]
        
        # Filter results for this token length and find best branch factor
        best_branch = 2
        best_speedup = 0
        for b in [2, 3, 4]:
            filtered = [r for r in results if r['tokens'] == tokens and r['branch'] == b]
            if filtered:
                max_speedup = max(r['speedup'] for r in filtered)
                if max_speedup > best_speedup:
                    best_speedup = max_speedup
                    best_branch = b
        
        filtered = [r for r in results if r['tokens'] == tokens and r['branch'] == best_branch]
        
        if not filtered:
            continue
        
        # Create matrix for heatmap (D x threshold)
        depths = sorted(set(r['depth'] for r in filtered))
        thresholds = sorted(set(r['threshold'] for r in filtered))
        
        matrix = np.zeros((len(depths), len(thresholds)))
        for r in filtered:
            i = depths.index(r['depth'])
            j = thresholds.index(r['threshold'])
            matrix[i, j] = r['speedup']
        
        # Plot heatmap
        if HAS_SEABORN:
            sns.heatmap(
                matrix, ax=ax, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'{t:.2f}' for t in thresholds],
                yticklabels=depths,
                cbar_kws={'label': 'Speedup'}
            )
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels(depths)
            
            # Add text annotations
            for i in range(len(depths)):
                for j in range(len(thresholds)):
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Speedup')
        
        ax.set_xlabel('Threshold (t)')
        ax.set_ylabel('Depth (D)')
        ax.set_title(f'{tokens} tokens (B={best_branch})')
    
    plt.suptitle('Tree V2 Parameter Search (WikiText): Speedup vs D and threshold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap saved to: {output_path}")


def plot_speedup_by_tokens(results: List[Dict], output_path: str):
    """Plot best speedup for each token length."""
    best_per_length = find_best_per_token_length(results)
    
    token_lengths = sorted(best_per_length.keys())
    speedups = [best_per_length[t]['speedup'] for t in token_lengths]
    configs = [f"D={best_per_length[t]['depth']} B={best_per_length[t]['branch']} t={best_per_length[t]['threshold']:.2f}" 
               for t in token_lengths]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(token_lengths)), speedups, color='steelblue')
    ax.set_xticks(range(len(token_lengths)))
    ax.set_xticklabels([str(t) for t in token_lengths])
    ax.set_xlabel('Token Length')
    ax.set_ylabel('Best Speedup')
    ax.set_title('Best Tree V2 Speedup by Token Length (WikiText Dataset)')
    
    # Add config labels on bars
    for i, (bar, config) in enumerate(zip(bars, configs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                config, ha='center', va='bottom', fontsize=8, rotation=15)
    
    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Speedup chart saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Tree V2 Parameter Search (WikiText Dataset)")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                       help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                       help="Path to draft model")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--num-runs", type=int, default=2,
                       help="Number of runs per configuration")
    parser.add_argument("--num-prompts", type=int, default=5,
                       help="Number of prompts to use from WikiText")
    parser.add_argument("--prompt-length", type=int, default=500,
                       help="Maximum prompt length in characters")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer configurations")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("papers/figures", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("Tree V2 Parameter Search (WikiText Dataset)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target Model: {args.target_model}")
    print(f"  Draft Model: {args.draft_model}")
    print(f"  Device: {args.device}")
    print(f"  Num Runs: {args.num_runs}")
    print(f"  Num Prompts: {args.num_prompts}")
    print(f"  Prompt Length: {args.prompt_length} chars")
    
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
        depths = [4, 5, 6, 7]
        branches = [2, 3]
        thresholds = [0.02, 0.03, 0.05]
        token_lengths = [200, 500]
    else:
        depths = DEFAULT_DEPTHS
        branches = DEFAULT_BRANCHES
        thresholds = DEFAULT_THRESHOLDS
        token_lengths = DEFAULT_TOKEN_LENGTHS
    
    # Run search
    print("\nStarting parameter search...")
    results = search_tree_params(
        target_model, draft_model, tokenizer, prompts, args.device,
        depths, branches, thresholds, token_lengths,
        args.num_runs
    )
    
    # Find best configuration
    best = find_best_config(results)
    best_per_length = find_best_per_token_length(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SEARCH RESULTS SUMMARY (WikiText Dataset)")
    print("=" * 70)
    
    print("\nBest configuration per token length:")
    for tokens in sorted(best_per_length.keys()):
        r = best_per_length[tokens]
        print(f"  {tokens:4d} tokens: D={r['depth']} B={r['branch']} t={r['threshold']:.2f} "
              f"-> {r['throughput']:.1f} t/s ({r['speedup']:.2f}x)")
    
    if best:
        print(f"\nOverall best configuration:")
        print(f"  Tokens: {best['tokens']}")
        print(f"  Depth (D): {best['depth']}")
        print(f"  Branch (B): {best['branch']}")
        print(f"  Threshold (t): {best['threshold']}")
        print(f"  Throughput: {best['throughput']:.1f} t/s")
        print(f"  Speedup: {best['speedup']:.2f}x")
        print(f"  Avg Path Length: {best['avg_path_length']:.1f}")
        print(f"  Acceptance Rate: {best['acceptance_rate']:.2%}")
    
    # Save results
    prompt_info = [{"length": len(p), "preview": p[:100]} for p in prompts]
    
    output_data = {
        'config': {
            'target_model': args.target_model,
            'draft_model': args.draft_model,
            'dataset': 'WikiText-2 (ModelScope)',
            'depths': depths,
            'branches': branches,
            'thresholds': thresholds,
            'token_lengths': token_lengths,
            'num_runs': args.num_runs,
            'num_prompts': args.num_prompts,
            'prompt_length': args.prompt_length,
            'timestamp': timestamp
        },
        'prompts': prompt_info,
        'results': results,
        'best_overall': best,
        'best_per_token_length': {str(k): v for k, v in best_per_length.items()}
    }
    
    json_path = os.path.join(args.output_dir, f"tree_param_search_wikitext_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate visualizations
    heatmap_path = f"papers/figures/tree_param_heatmap_wikitext_{timestamp}.png"
    plot_heatmaps(results, heatmap_path)
    
    speedup_path = f"papers/figures/tree_param_speedup_wikitext_{timestamp}.png"
    plot_speedup_by_tokens(results, speedup_path)
    
    print("\n" + "=" * 70)
    print("Parameter search complete!")
    print("=" * 70)
    print(f"\nTo run benchmark with best config:")
    if best:
        print(f"  python papers/benchmark_wikitext.py \\")
        print(f"      --target-model {args.target_model} \\")
        print(f"      --draft-model {args.draft_model} \\")
        print(f"      --num-samples {args.num_prompts} \\")
        print(f"      --max-new-tokens {best['tokens']}")


if __name__ == "__main__":
    main()





#!/usr/bin/env python3
"""
Ablation Study for Head-Aware Attention Mask

This script runs multiple experiments to understand:
1. How different window sizes affect PPL
2. Whether head-aware windowing outperforms uniform at equivalent context sizes
3. The importance of sink tokens

Usage:
    python scripts/ablation_head_aware_mask.py \
        --model EleutherAI/pythia-2.8b \
        --classifications results/attention_analysis_pythia-2.8b/head_classifications.json \
        --max-tokens 2000 \
        --output results/ablation_head_aware_mask.json
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_tokenizer(model_id: str, device: torch.device):
    print(f"Loading model: {model_id}")
    
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_sample_text(num_chars: int = 50000) -> str:
    data_path = project_root / "data" / "pg19.parquet"
    if data_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            return df.iloc[0]['text'][:num_chars]
        except:
            pass
    return ("The quick brown fox jumps over the lazy dog. " * 1000)[:num_chars]


def run_experiment(model, tokenizer, text, max_tokens, device, config):
    """Run a single experiment with given configuration."""
    from kvcompress.evaluate import evaluate_with_head_aware_mask
    from kvcompress.methods.head_aware_compress import HeadAwareMaskGenerator
    
    name = config['name']
    print(f"\n  Running: {name}...")
    
    if config['type'] == 'uniform':
        mask_gen = HeadAwareMaskGenerator.create_uniform(
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
            sink_size=config['sink_size'],
            window_size=config['window_size'],
        )
    elif config['type'] == 'head_aware':
        mask_gen = HeadAwareMaskGenerator.from_classifications(
            config['classifications_path'],
            window_size_override=config.get('window_override'),
            sink_size=config.get('sink_size', 4),
        )
    elif config['type'] == 'head_aware_custom':
        # Custom window sizes for different head types
        mask_gen = HeadAwareMaskGenerator.from_classifications(
            config['classifications_path'],
            window_size_override=config['window_override'],
            sink_size=config.get('sink_size', 4),
        )
    else:
        raise ValueError(f"Unknown config type: {config['type']}")
    
    result = evaluate_with_head_aware_mask(
        model=model,
        tokenizer=tokenizer,
        text=text,
        mask_generator=mask_gen,
        max_tokens=max_tokens,
        device=device,
        show_progress=True,
    )
    
    result['name'] = name
    result['config'] = config
    
    clear_gpu_memory()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Ablation study for head-aware mask")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--classifications", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    text = load_sample_text(args.max_tokens * 10)
    
    print("\n" + "="*70)
    print("ABLATION STUDY: HEAD-AWARE ATTENTION MASK")
    print("="*70)
    
    # Define experiments
    experiments = [
        # Uniform baselines with different window sizes
        {"name": "uniform_512", "type": "uniform", "sink_size": 4, "window_size": 508},
        {"name": "uniform_256", "type": "uniform", "sink_size": 4, "window_size": 252},
        {"name": "uniform_128", "type": "uniform", "sink_size": 4, "window_size": 124},
        {"name": "uniform_64", "type": "uniform", "sink_size": 4, "window_size": 60},
        
        # Head-aware with default settings
        {"name": "head_aware_default", "type": "head_aware", 
         "classifications_path": args.classifications},
        
        # Head-aware with different positional window sizes
        {"name": "head_aware_pos4", "type": "head_aware_custom",
         "classifications_path": args.classifications,
         "window_override": {"positional": 4, "mixed": 64, "gathering": -1}},
        
        {"name": "head_aware_pos16", "type": "head_aware_custom",
         "classifications_path": args.classifications,
         "window_override": {"positional": 16, "mixed": 64, "gathering": -1}},
        
        {"name": "head_aware_pos32", "type": "head_aware_custom",
         "classifications_path": args.classifications,
         "window_override": {"positional": 32, "mixed": 64, "gathering": -1}},
        
        # Head-aware with different mixed window sizes
        {"name": "head_aware_mix32", "type": "head_aware_custom",
         "classifications_path": args.classifications,
         "window_override": {"positional": 8, "mixed": 32, "gathering": -1}},
        
        {"name": "head_aware_mix128", "type": "head_aware_custom",
         "classifications_path": args.classifications,
         "window_override": {"positional": 8, "mixed": 128, "gathering": -1}},
        
        # Test without sink tokens
        {"name": "head_aware_no_sink", "type": "head_aware_custom",
         "classifications_path": args.classifications,
         "sink_size": 0,
         "window_override": {"positional": 12, "mixed": 68, "gathering": -1}},
        
        # All heads use small window (extreme compression)
        {"name": "uniform_32", "type": "uniform", "sink_size": 4, "window_size": 28},
        {"name": "uniform_16", "type": "uniform", "sink_size": 4, "window_size": 12},
    ]
    
    results = []
    
    for exp in experiments:
        try:
            result = run_experiment(model, tokenizer, text, args.max_tokens, device, exp)
            results.append(result)
            
            print(f"    PPL: {result['perplexity']:.2f}, "
                  f"Acc: {result['accuracy']:.2%}, "
                  f"Eff.Ctx: {result['effective_context']:.1f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"name": exp['name'], "error": str(e)})
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Name':<25} {'PPL':>8} {'Acc':>8} {'Eff.Ctx':>10}")
    print("-"*55)
    
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<25} {'ERROR':>8}")
        else:
            print(f"{r['name']:<25} {r['perplexity']:>8.2f} {r['accuracy']:>7.2%} "
                  f"{r['effective_context']:>10.1f}")
    
    # Compute insights
    print("\n" + "-"*70)
    print("KEY INSIGHTS:")
    
    # Find best head-aware vs best uniform at similar context
    head_aware_results = [r for r in results if 'head_aware' in r.get('name', '') and 'error' not in r]
    uniform_results = [r for r in results if 'uniform' in r.get('name', '') and 'error' not in r]
    
    if head_aware_results and uniform_results:
        best_ha = min(head_aware_results, key=lambda x: x['perplexity'])
        
        # Find uniform with similar effective context
        ha_ctx = best_ha['effective_context']
        closest_uniform = min(uniform_results, 
                             key=lambda x: abs(x['effective_context'] - ha_ctx))
        
        print(f"\nBest Head-Aware: {best_ha['name']}")
        print(f"  PPL={best_ha['perplexity']:.2f}, Ctx={best_ha['effective_context']:.0f}")
        
        print(f"\nClosest Uniform: {closest_uniform['name']}")
        print(f"  PPL={closest_uniform['perplexity']:.2f}, Ctx={closest_uniform['effective_context']:.0f}")
        
        ppl_diff = (best_ha['perplexity'] - closest_uniform['perplexity']) / closest_uniform['perplexity'] * 100
        print(f"\nHead-Aware vs Uniform (similar context): PPL {ppl_diff:+.2f}%")
    
    # Save results
    if args.output:
        output_data = {
            'config': {
                'model': args.model,
                'max_tokens': args.max_tokens,
                'classifications': args.classifications,
            },
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()


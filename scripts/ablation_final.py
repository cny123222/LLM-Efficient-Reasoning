#!/usr/bin/env python3
"""
Final Ablation Study for Head-Aware Attention Mask

This streamlined script focuses on:
1. Control groups (Window-only, StreamingLLM)
2. Confidence-based approaches with threshold sweep
3. Inverse experiments (large positional windows)
4. Small gathering window experiments

Usage:
    python scripts/ablation_final.py \
        --model EleutherAI/pythia-2.8b \
        --classifications results/attention_analysis_pythia-2.8b/head_classifications.json \
        --max-tokens 1000 \
        --output results/ablation_study/ablation_final.json
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
    elif config['type'] == 'confidence_based':
        mask_gen = HeadAwareMaskGenerator.from_classifications_with_confidence(
            config['classifications_path'],
            confidence_threshold=config.get('confidence_threshold', 0.6),
            base_window=config.get('base_window', 128),
            high_conf_windows=config.get('high_conf_windows'),
            sink_size=config.get('sink_size', 4),
        )
    elif config['type'] == 'inverse':
        mask_gen = HeadAwareMaskGenerator.from_classifications_inverse(
            config['classifications_path'],
            base_window=config.get('base_window', 64),
            gathering_window=config.get('gathering_window', 256),
            gathering_confidence_threshold=config.get('gathering_confidence_threshold', 0.4),
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
    result['group'] = config.get('group', 'unknown')
    
    clear_gpu_memory()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Final ablation study for head-aware mask")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--classifications", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    text = load_sample_text(args.max_tokens * 10)
    
    print("\n" + "="*70)
    print("FINAL ABLATION STUDY: HEAD-AWARE ATTENTION MASK")
    print("="*70)
    
    experiments = []
    
    # ========== Group A: Window only (no sink) - Control ==========
    # Shows the importance of sink tokens
    for window in [512, 256, 128, 64]:
        experiments.append({
            "name": f"A_window_{window}",
            "group": "A_window_only",
            "type": "uniform",
            "sink_size": 0,
            "window_size": window,
        })
    
    # ========== Group B: StreamingLLM (sink=4 + window) - Main Baseline ==========
    for total in [512, 256, 128, 64]:
        window = total - 4
        experiments.append({
            "name": f"B_streaming_{total}",
            "group": "B_streaming",
            "type": "uniform",
            "sink_size": 4,
            "window_size": window,
        })
    
    # ========== Group G: Confidence-Based Threshold Sweep ==========
    # Test different confidence thresholds with base_window=128
    for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
        experiments.append({
            "name": f"G_conf{threshold}_base128",
            "group": "G_confidence_sweep",
            "type": "confidence_based",
            "classifications_path": args.classifications,
            "confidence_threshold": threshold,
            "base_window": 128,
            "high_conf_windows": {"positional": 16, "mixed": 64, "gathering": 256},
            "sink_size": 4,
        })
    
    # Test with base_window=64
    for threshold in [0.5, 0.6, 0.7]:
        experiments.append({
            "name": f"G_conf{threshold}_base64",
            "group": "G_confidence_sweep",
            "type": "confidence_based",
            "classifications_path": args.classifications,
            "confidence_threshold": threshold,
            "base_window": 64,
            "high_conf_windows": {"positional": 8, "mixed": 64, "gathering": 128},
            "sink_size": 4,
        })
    
    # ========== Group I: Inverse Experiments ==========
    # Give positional heads LARGER windows (opposite of traditional approach)
    # Hypothesis: Maybe positional heads need MORE context, not less
    for pos_window in [128, 256, 512]:
        experiments.append({
            "name": f"I_pos{pos_window}_mix64_g128",
            "group": "I_inverse_positional",
            "type": "head_aware",
            "classifications_path": args.classifications,
            "window_override": {"positional": pos_window, "mixed": 64, "gathering": 128},
            "sink_size": 4,
        })
    
    # Inverse with larger mixed window
    experiments.append({
        "name": "I_pos256_mix128_g128",
        "group": "I_inverse_positional",
        "type": "head_aware",
        "classifications_path": args.classifications,
        "window_override": {"positional": 256, "mixed": 128, "gathering": 128},
        "sink_size": 4,
    })
    
    # ========== Group J: Small Gathering Window ==========
    # Test if gathering heads truly need large context
    for g_window in [64, 128, 256]:
        experiments.append({
            "name": f"J_pos16_mix64_g{g_window}",
            "group": "J_small_gathering",
            "type": "head_aware",
            "classifications_path": args.classifications,
            "window_override": {"positional": 16, "mixed": 64, "gathering": g_window},
            "sink_size": 4,
        })
    
    # With larger positional to compensate
    experiments.append({
        "name": "J_pos64_mix64_g128",
        "group": "J_small_gathering",
        "type": "head_aware",
        "classifications_path": args.classifications,
        "window_override": {"positional": 64, "mixed": 64, "gathering": 128},
        "sink_size": 4,
    })
    
    # ========== Group K: Uniform-like with type awareness ==========
    # Give all heads similar windows but still use type info
    for window in [64, 128]:
        experiments.append({
            "name": f"K_uniform{window}_aware",
            "group": "K_uniform_aware",
            "type": "head_aware",
            "classifications_path": args.classifications,
            "window_override": {"positional": window, "mixed": window, "gathering": window},
            "sink_size": 4,
        })
    
    results = []
    
    current_group = None
    for exp in experiments:
        group = exp.get('group', 'unknown')
        if group != current_group:
            current_group = group
            print(f"\n{'='*70}")
            print(f"GROUP: {group}")
            print(f"{'='*70}")
        
        try:
            result = run_experiment(model, tokenizer, text, args.max_tokens, device, exp)
            results.append(result)
            
            print(f"    PPL: {result['perplexity']:.2f}, "
                  f"Acc: {result['accuracy']:.2%}, "
                  f"Eff.Ctx: {result['effective_context']:.1f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": exp['name'], "group": exp.get('group'), "error": str(e)})
    
    # Print summary by group
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    
    groups = {}
    for r in results:
        g = r.get('group', 'unknown')
        if g not in groups:
            groups[g] = []
        groups[g].append(r)
    
    for group_name in sorted(groups.keys()):
        group_results = groups[group_name]
        print(f"\n--- {group_name} ---")
        print(f"{'Name':<25} {'PPL':>8} {'Acc':>8} {'Eff.Ctx':>10}")
        print("-"*55)
        
        for r in sorted(group_results, key=lambda x: x.get('effective_context', 0)):
            if 'error' in r:
                print(f"{r['name']:<25} {'ERROR':>8}")
            else:
                print(f"{r['name']:<25} {r['perplexity']:>8.2f} {r['accuracy']:>7.2%} "
                      f"{r['effective_context']:>10.1f}")
    
    # Key comparisons
    print("\n" + "="*70)
    print("KEY COMPARISONS")
    print("="*70)
    
    by_name = {r['name']: r for r in results if 'error' not in r}
    streaming_results = [r for r in results if r.get('group') == 'B_streaming' and 'error' not in r]
    
    # Find best streaming at ctx~128
    streaming_128 = by_name.get('B_streaming_128')
    if streaming_128:
        print(f"\nBaseline (B_streaming_128): PPL={streaming_128['perplexity']:.2f}, Ctx={streaming_128['effective_context']:.0f}")
        
        # Compare all confidence-based
        conf_results = [r for r in results if r.get('group') == 'G_confidence_sweep' and 'error' not in r]
        if conf_results:
            print("\nConfidence-based vs StreamingLLM_128:")
            for r in sorted(conf_results, key=lambda x: x['perplexity']):
                ppl_diff = (r['perplexity'] - streaming_128['perplexity']) / streaming_128['perplexity'] * 100
                better = "better" if ppl_diff < 0 else "worse"
                print(f"  {r['name']}: PPL={r['perplexity']:.2f}, Ctx={r['effective_context']:.0f} ({ppl_diff:+.1f}% {better})")
        
        # Compare inverse experiments
        inverse_results = [r for r in results if r.get('group') == 'I_inverse_positional' and 'error' not in r]
        if inverse_results:
            print("\nInverse (large positional) vs StreamingLLM:")
            for r in sorted(inverse_results, key=lambda x: x['perplexity']):
                closest_s = min(streaming_results, key=lambda x: abs(x['effective_context'] - r['effective_context']))
                ppl_diff = (r['perplexity'] - closest_s['perplexity']) / closest_s['perplexity'] * 100
                print(f"  {r['name']}: PPL={r['perplexity']:.2f}, Ctx={r['effective_context']:.0f}")
                print(f"    vs {closest_s['name']}: {ppl_diff:+.1f}%")
        
        # Compare small gathering
        small_g_results = [r for r in results if r.get('group') == 'J_small_gathering' and 'error' not in r]
        if small_g_results:
            print("\nSmall gathering window experiments:")
            for r in sorted(small_g_results, key=lambda x: x['perplexity']):
                closest_s = min(streaming_results, key=lambda x: abs(x['effective_context'] - r['effective_context']))
                ppl_diff = (r['perplexity'] - closest_s['perplexity']) / closest_s['perplexity'] * 100
                print(f"  {r['name']}: PPL={r['perplexity']:.2f}, Ctx={r['effective_context']:.0f} ({ppl_diff:+.1f}%)")
    
    # Best overall
    valid_results = [r for r in results if 'error' not in r and r.get('group') != 'A_window_only']
    if valid_results:
        best = min(valid_results, key=lambda x: x['perplexity'])
        print(f"\n*** BEST RESULT: {best['name']} ***")
        print(f"    PPL: {best['perplexity']:.2f}, Acc: {best['accuracy']:.2%}, Ctx: {best['effective_context']:.0f}")
    
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
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*70)
    print("FINAL ABLATION STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()


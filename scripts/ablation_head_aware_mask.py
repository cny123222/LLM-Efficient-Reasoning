#!/usr/bin/env python3
"""
Ablation Study for Head-Aware Attention Mask

This script runs multiple experiments to understand:
1. How different window sizes affect PPL (uniform vs head-aware)
2. The importance of sink tokens
3. Whether head-aware windowing outperforms uniform at equivalent context sizes

Experiment Groups:
- Group A: Uniform window (sink + window) at various sizes
- Group B: Sink-only (no window) at various sizes  
- Group C: Head-aware with different positional window sizes
- Group D: Head-aware with different mixed window sizes
- Group E: Head-aware without sink tokens

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
    
    # Define experiments in logical order
    experiments = []
    
    # ========== Group A: Window only (no sink) ==========
    # Pure sliding window without sink tokens
    for window in [512, 256, 128, 64]:
        experiments.append({
            "name": f"A_window_{window}",
            "group": "A_window_only",
            "type": "uniform",
            "sink_size": 0,
            "window_size": window,
        })
    
    # ========== Group B: StreamingLLM style (sink=4 + window) ==========
    # Standard StreamingLLM: 4 sink tokens + sliding window
    for total in [512, 256, 128, 64]:
        window = total - 4
        experiments.append({
            "name": f"B_streaming_{total}",
            "group": "B_streaming",
            "type": "uniform",
            "sink_size": 4,
            "window_size": window,
        })
    
    # ========== Group C: Head-aware with varying positional window ==========
    # Test how positional head window size affects PPL
    # Keep mixed=64, gathering=full
    # for pos_window in [4, 8, 16, 32, 64]:
        # experiments.append({
        #     "name": f"C_ha_pos{pos_window}",
        #     "group": "C_head_aware_pos",
        #     "type": "head_aware",
        #     "classifications_path": args.classifications,
        #     "window_override": {"positional": pos_window, "mixed": 64, "gathering": -1},
        #     "sink_size": 4,
        # })
    
    # ========== Group C: Head-aware pos varying + gathering limited to 512 ==========
    # Same as Group C, but gathering heads limited to sink(4) + window(508) = 512
    # Tests whether gathering heads truly need full context
    for pos_window in [8, 16, 32, 64]:
        for mix_window in [32, 64, 128]:
            experiments.append({
                "name": f"C_ha_pos{pos_window}_mix{mix_window}_g512",
                "group": "C_head_aware_pos_mix_g512",
                "type": "head_aware",
                "classifications_path": args.classifications,
                "window_override": {"positional": pos_window, "mixed": mix_window, "gathering": 508},
                "sink_size": 4,
            })
    
    # ========== Group D: Head-aware with varying mixed window ==========
    # Test how mixed head window size affects PPL
    # Keep positional=8, gathering=full
    for pos_window in [8, 16, 32, 64]:
        for mix_window in [32, 64, 128]:
            experiments.append({
                "name": f"D_ha_pos{pos_window}_mix{mix_window}",
                "group": "D_head_aware_pos_mix",
                "type": "head_aware",
                "classifications_path": args.classifications,
                "window_override": {"positional": pos_window, "mixed": mix_window, "gathering": -1},
                "sink_size": 4,
            })
    
    # ========== Group E: Head-aware without sink ==========
    # Test the importance of sink tokens in head-aware setting
    experiments.append({
        "name": "E_ha_no_sink",
        "group": "E_head_aware_no_sink",
        "type": "head_aware",
        "classifications_path": args.classifications,
        "window_override": {"positional": 12, "mixed": 68, "gathering": -1},
        "sink_size": 0,
    })
    
    # ========== Group F: Head-aware with per-head recommended windows ==========
    # Uses the recommended_window from head_classifications.json for each head
    # This is the "optimal" per-head configuration from attention analysis:
    # - Positional heads: use analyzed recommended_window (mostly 8, some 16)
    # - Mixed heads: default to 64 (since their recommended_window is -1)
    # - Gathering heads: full context (-1)
    experiments.append({
        "name": "F_ha_default",
        "group": "F_head_aware_default",
        "type": "head_aware",
        "classifications_path": args.classifications,
        "sink_size": 4,
        # No window_override - uses per-head recommended_window from JSON
    })
    
    # F2: Same as F but with gathering limited to 512
    experiments.append({
        "name": "F2_ha_default_g512",
        "group": "F_head_aware_default",
        "type": "head_aware",
        "classifications_path": args.classifications,
        "window_override": {"gathering": 508},  # Only override gathering
        "sink_size": 4,
    })
    
    # ========== Group G: Confidence-based window sizing ==========
    # Only trust high-confidence classifications, use baseline for uncertain heads
    # G1: threshold=0.6, base=128 (strict - only trust very confident)
    experiments.append({
        "name": "G1_conf0.6_base128",
        "group": "G_confidence_based",
        "type": "confidence_based",
        "classifications_path": args.classifications,
        "confidence_threshold": 0.6,
        "base_window": 128,
        "high_conf_windows": {"positional": 16, "mixed": 64, "gathering": 512},
        "sink_size": 4,
    })
    
    # G2: threshold=0.5, base=128 (medium)
    experiments.append({
        "name": "G2_conf0.5_base128",
        "group": "G_confidence_based",
        "type": "confidence_based",
        "classifications_path": args.classifications,
        "confidence_threshold": 0.5,
        "base_window": 128,
        "high_conf_windows": {"positional": 16, "mixed": 64, "gathering": 512},
        "sink_size": 4,
    })
    
    # G3: threshold=0.4, base=64 (lenient, smaller baseline)
    experiments.append({
        "name": "G3_conf0.4_base64",
        "group": "G_confidence_based",
        "type": "confidence_based",
        "classifications_path": args.classifications,
        "confidence_threshold": 0.4,
        "base_window": 64,
        "high_conf_windows": {"positional": 8, "mixed": 64, "gathering": 256},
        "sink_size": 4,
    })
    
    # G4: threshold=0.6, base=64 (strict threshold, small baseline)
    experiments.append({
        "name": "G4_conf0.6_base64",
        "group": "G_confidence_based",
        "type": "confidence_based",
        "classifications_path": args.classifications,
        "confidence_threshold": 0.6,
        "base_window": 64,
        "high_conf_windows": {"positional": 8, "mixed": 64, "gathering": 256},
        "sink_size": 4,
    })
    
    # ========== Group H: Inverse approach (uniform + expand gathering) ==========
    # Give everyone baseline, only expand confident gathering heads
    # H1: base=64, gathering=256, threshold=0.4
    experiments.append({
        "name": "H1_base64_g256",
        "group": "H_inverse",
        "type": "inverse",
        "classifications_path": args.classifications,
        "base_window": 64,
        "gathering_window": 256,
        "gathering_confidence_threshold": 0.4,
        "sink_size": 4,
    })
    
    # H2: base=64, gathering=512, threshold=0.3
    experiments.append({
        "name": "H2_base64_g512",
        "group": "H_inverse",
        "type": "inverse",
        "classifications_path": args.classifications,
        "base_window": 64,
        "gathering_window": 512,
        "gathering_confidence_threshold": 0.3,
        "sink_size": 4,
    })
    
    # H3: base=128, gathering=256, threshold=0.4
    experiments.append({
        "name": "H3_base128_g256",
        "group": "H_inverse",
        "type": "inverse",
        "classifications_path": args.classifications,
        "base_window": 128,
        "gathering_window": 256,
        "gathering_confidence_threshold": 0.4,
        "sink_size": 4,
    })
    
    # H4: base=128, gathering=512, threshold=0.3
    experiments.append({
        "name": "H4_base128_g512",
        "group": "H_inverse",
        "type": "inverse",
        "classifications_path": args.classifications,
        "base_window": 128,
        "gathering_window": 512,
        "gathering_confidence_threshold": 0.3,
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
    print("ABLATION RESULTS SUMMARY (by group)")
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
        print(f"{'Name':<20} {'PPL':>8} {'Acc':>8} {'Eff.Ctx':>10}")
        print("-"*50)
        
        for r in group_results:
            if 'error' in r:
                print(f"{r['name']:<20} {'ERROR':>8}")
            else:
                print(f"{r['name']:<20} {r['perplexity']:>8.2f} {r['accuracy']:>7.2%} "
                      f"{r['effective_context']:>10.1f}")
    
    # Key comparisons
    print("\n" + "="*70)
    print("KEY COMPARISONS")
    print("="*70)
    
    # Get results by name for easy lookup
    by_name = {r['name']: r for r in results if 'error' not in r}
    
    # Comparison 1: Window-only vs StreamingLLM (Effect of Sink Tokens)
    print("\n1. Window-only vs StreamingLLM (Effect of Sink Tokens):")
    window_results = [r for r in results if r.get('group') == 'A_window_only' and 'error' not in r]
    streaming_results = [r for r in results if r.get('group') == 'B_streaming' and 'error' not in r]
    if window_results and streaming_results:
        for w_res in sorted(window_results, key=lambda x: -x['effective_context'])[:3]:
            # Find streaming with similar total context
            closest_s = min(streaming_results, key=lambda x: abs(x['effective_context'] - w_res['effective_context']))
            ppl_diff = (w_res['perplexity'] - closest_s['perplexity']) / closest_s['perplexity'] * 100
            print(f"   Window {w_res['effective_context']:.0f}: PPL={w_res['perplexity']:.2f} vs "
                  f"Streaming {closest_s['effective_context']:.0f}: PPL={closest_s['perplexity']:.2f} "
                  f"({ppl_diff:+.2f}%)")
    
    # Comparison 2: Head-aware (g512) vs StreamingLLM at similar context
    print("\n2. Head-Aware (gathering=512) vs StreamingLLM:")
    ha_g512_results = [r for r in results if r.get('group') == 'C_head_aware_pos_mix_g512' and 'error' not in r]
    if ha_g512_results and streaming_results:
        # Pick a representative head-aware config
        for ha_res in sorted(ha_g512_results, key=lambda x: x['effective_context'])[:3]:
            closest_s = min(streaming_results, key=lambda x: abs(x['effective_context'] - ha_res['effective_context']))
            ppl_diff = (ha_res['perplexity'] - closest_s['perplexity']) / closest_s['perplexity'] * 100
            better = 'head-aware better' if ppl_diff < 0 else 'streaming better'
            print(f"   {ha_res['name']}: PPL={ha_res['perplexity']:.2f}, Ctx={ha_res['effective_context']:.0f}")
            print(f"      vs {closest_s['name']}: PPL={closest_s['perplexity']:.2f}, Ctx={closest_s['effective_context']:.0f} ({ppl_diff:+.2f}%, {better})")
    
    # Comparison 3: Head-aware (gathering=full) vs (gathering=512)
    print("\n3. Gathering Full vs Gathering=512 (Do gathering heads need full context?):")
    ha_full_results = [r for r in results if r.get('group') == 'D_head_aware_pos_mix' and 'error' not in r]
    if ha_g512_results and ha_full_results:
        # Compare same pos/mix configs
        for ha_g512 in ha_g512_results[:4]:
            # Extract pos and mix from name like "C_ha_pos8_mix64_g512"
            parts = ha_g512['name'].replace('C_ha_pos', '').replace('_g512', '').split('_mix')
            if len(parts) == 2:
                pos, mix = parts[0], parts[1]
                target_name = f"D_ha_pos{pos}_mix{mix}"
                ha_full = by_name.get(target_name)
                if ha_full:
                    ppl_diff = (ha_g512['perplexity'] - ha_full['perplexity']) / ha_full['perplexity'] * 100
                    ctx_diff = ha_g512['effective_context'] - ha_full['effective_context']
                    print(f"   pos={pos}, mix={mix}:")
                    print(f"      g=512: PPL={ha_g512['perplexity']:.2f}, Ctx={ha_g512['effective_context']:.0f}")
                    print(f"      g=full: PPL={ha_full['perplexity']:.2f}, Ctx={ha_full['effective_context']:.0f}")
                    print(f"      PPL change: {ppl_diff:+.2f}%, Context saved: {-ctx_diff:.0f}")
    
    # Comparison 4: Effect of positional window size (within Group C)
    print("\n4. Positional Window Sensitivity (gathering=512, mixed=64):")
    pos_sensitivity = [(r['name'], r['perplexity'], r['effective_context']) 
                       for r in ha_g512_results if '_mix64_' in r['name']]
    if pos_sensitivity:
        for name, ppl, ctx in sorted(pos_sensitivity, key=lambda x: x[2]):
            print(f"   {name}: PPL={ppl:.2f}, Ctx={ctx:.0f}")
    
    # Comparison 5: Effect of mixed window size (within Group C)
    print("\n5. Mixed Window Sensitivity (gathering=512, pos=16):")
    mix_sensitivity = [(r['name'], r['perplexity'], r['effective_context']) 
                       for r in ha_g512_results if '_pos16_' in r['name']]
    if mix_sensitivity:
        for name, ppl, ctx in sorted(mix_sensitivity, key=lambda x: x[2]):
            print(f"   {name}: PPL={ppl:.2f}, Ctx={ctx:.0f}")
    
    # Comparison 6: Effect of sink tokens in head-aware
    print("\n6. Effect of Sink Tokens (Head-Aware with vs without sink):")
    ha_no_sink = by_name.get('E_ha_no_sink')
    # Find a comparable head-aware config with sink
    ha_with_sink = by_name.get('D_ha_pos16_mix64')
    if ha_with_sink and ha_no_sink:
        print(f"   With sink (4):    PPL={ha_with_sink['perplexity']:.2f}, Ctx={ha_with_sink['effective_context']:.0f}")
        print(f"   Without sink:     PPL={ha_no_sink['perplexity']:.2f}, Ctx={ha_no_sink['effective_context']:.0f}")
        ppl_diff = (ha_no_sink['perplexity'] - ha_with_sink['perplexity']) / ha_with_sink['perplexity'] * 100
        print(f"   PPL change without sink: {ppl_diff:+.2f}%")
    
    # Comparison 7: F group - Per-head recommended windows vs uniform overrides
    print("\n7. Per-Head Recommended Windows (F group) vs Uniform Overrides:")
    f_default = by_name.get('F_ha_default')
    f_g512 = by_name.get('F2_ha_default_g512')
    if f_default:
        print(f"   F_ha_default (per-head windows): PPL={f_default['perplexity']:.2f}, Ctx={f_default['effective_context']:.0f}")
        # Compare with similar C/D configs
        if streaming_results:
            closest_s = min(streaming_results, key=lambda x: abs(x['effective_context'] - f_default['effective_context']))
            ppl_diff = (f_default['perplexity'] - closest_s['perplexity']) / closest_s['perplexity'] * 100
            print(f"   vs Streaming {closest_s['effective_context']:.0f}: PPL={closest_s['perplexity']:.2f} ({ppl_diff:+.2f}%)")
    if f_g512:
        print(f"   F2_ha_default_g512 (gathering=512): PPL={f_g512['perplexity']:.2f}, Ctx={f_g512['effective_context']:.0f}")
        if f_default:
            ppl_diff = (f_g512['perplexity'] - f_default['perplexity']) / f_default['perplexity'] * 100
            ctx_saved = f_default['effective_context'] - f_g512['effective_context']
            print(f"   PPL change: {ppl_diff:+.2f}%, Context saved: {ctx_saved:.0f}")
    
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
    print("ABLATION STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

# Head-Aware KV Cache Compression: Experiment Report

## Executive Summary

This report documents our investigation into head-aware KV cache compression strategies for transformer language models. We explored whether applying different attention window sizes to different attention heads (based on their functional classification) could outperform uniform compression strategies like StreamingLLM.

**Key Finding**: On Pythia-2.8b, uniform StreamingLLM-style compression (sink tokens + sliding window) **consistently outperforms** head-aware strategies at equivalent effective context sizes. The best head-aware approach achieved only marginal improvement over baseline uniform compression.

## Background

### Motivation
Attention heads in transformer models exhibit different behaviors:
- **Gathering heads**: Aggregate information from across the entire sequence
- **Positional heads**: Focus on recent/local positions
- **Mixed heads**: Show mixed attention patterns

The hypothesis was that by giving gathering heads more context and positional heads less context, we could achieve better PPL/accuracy at the same average context size.

### Model and Classification
- **Model**: EleutherAI/pythia-2.8b (32 layers, 32 heads per layer = 1024 total heads)
- **Classification Results**:
  - Gathering: 88 heads (8.6%)
  - Mixed: 566 heads (55.3%)
  - Positional: 370 heads (36.1%)

### Critical Issue: Classification Confidence
Analysis of classification confidence revealed a significant problem:

| Head Type | Count | Confidence Range | Avg Confidence |
|-----------|-------|------------------|----------------|
| Positional | 370 | 0.025 - 1.000 | 0.509 |
| Mixed | 566 | **0.500 - 0.500** | 0.500 |
| Gathering | 88 | 0.011 - 0.590 | 0.255 |

**55% of heads (mixed) have confidence exactly 0.500**, meaning the classifier is essentially guessing. Only 34 heads (3.3%) have confidence ≥ 0.6.

## Experiment Design

### Control Groups
- **A (Window-only)**: Pure sliding window without sink tokens
- **B (StreamingLLM)**: Sink tokens (4) + sliding window

### Experimental Groups
- **C/D**: Traditional head-aware (positional=small, gathering=large)
- **E**: Head-aware without sink tokens
- **F**: Per-head recommended windows from analysis
- **G**: Confidence-based (only trust high-confidence classifications)
- **H**: Inverse approach (uniform baseline + expand confident gathering)

## Results Summary

### Key Results Table

| Configuration | PPL | Accuracy | Eff. Context | vs B_streaming_128 |
|--------------|-----|----------|--------------|-------------------|
| **B_streaming_512** | **8.81** | 52.45% | 512 | -7.5% |
| **B_streaming_256** | **9.14** | 51.95% | 256 | -4.0% |
| **B_streaming_128** | **9.52** | 52.05% | 128 | baseline |
| **B_streaming_64** | 10.57 | 51.05% | 64 | +11.0% |
| G1_conf0.6_base128 | 9.57 | 51.85% | 128.3 | +0.5% |
| H3_base128_g256 | 9.57 | 52.05% | 134 | +0.5% |
| C_ha_pos64_mix128_g512 | 9.75 | 52.35% | 141.5 | +2.4% |
| F_ha_default | 11.74 | 49.75% | 130 | +23.3% |
| A_window_128 (no sink) | 26.28 | 40.94% | 128 | +176% |
| E_ha_no_sink | 151.69 | 24.42% | 130 | +1493% |

### Key Observations

1. **Sink Tokens are Critical**
   - Without sink tokens, PPL explodes (26-152 vs 9-11)
   - This confirms the StreamingLLM finding about attention sinks

2. **Head-Aware Never Beats StreamingLLM**
   - Best head-aware (G1_conf0.6): PPL=9.57, Ctx=128.3
   - StreamingLLM baseline: PPL=9.52, Ctx=128
   - Difference: only +0.5% worse

3. **Traditional Head-Aware is Harmful**
   - F_ha_default (per-head windows): PPL=11.74 (+23% worse)
   - C/D groups consistently underperform StreamingLLM

4. **Confidence-Based Approach Helps**
   - Only trusting 34 high-confidence heads brings PPL close to baseline
   - Essentially becomes "StreamingLLM + minor tweaks"

5. **Classification Unreliability**
   - 55% of heads (mixed) have confidence=0.5 (random)
   - Forcing differentiation on unreliable classifications hurts performance

## Analysis

### Why Head-Aware Fails

1. **Unreliable Classifications**: Over half the heads cannot be confidently classified. Treating them differently based on uncertain labels introduces noise.

2. **Context Distribution Mismatch**: Head-aware creates uneven context distribution:
   - Some heads see 512 tokens (gathering)
   - Some see only 8 tokens (positional)
   - This imbalance may disrupt information flow between layers

3. **Positional Heads May Need More Context**: Our analysis assumes positional heads need small windows, but they may actually benefit from seeing more context to establish position properly.

4. **Model Architecture**: Pythia models may have evolved during training to rely on uniform attention patterns, making head-specific compression counterproductive.

### What Works

1. **Sink Tokens**: Absolutely essential (4 tokens sufficient)
2. **Uniform Window**: Simple sliding window works best
3. **Conservative Approach**: When uncertain, use baseline (confidence-based G1)

## Conclusions

### For Pythia-2.8b
- **Recommendation**: Use StreamingLLM (sink=4, window=124 for ctx=128)
- Head-aware compression provides no benefit and often hurts performance
- The head classification methodology may need fundamental revision

### Potential Future Directions

1. **Improve Classification Method**
   - Current entropy-based classification yields low confidence
   - Consider learned classifiers or attention pattern clustering

2. **Try Larger Models**
   - Head specialization may be more pronounced in larger models
   - Test on Pythia-6.9b, Pythia-12b, or other architectures

3. **Different Architectures**
   - Models with explicit head specialization (MoE heads) might benefit more
   - Test on models with different attention patterns (Llama, Mistral)

4. **Dynamic Head-Aware**
   - Instead of static windows, adapt based on actual attention patterns during inference

## Appendix: Experiment Configurations

### StreamingLLM Baseline (Group B)
```python
sink_size = 4
window_size = total_context - 4  # e.g., 124 for ctx=128
```

### Confidence-Based (Group G)
```python
confidence_threshold = 0.6
base_window = 128  # for low-confidence heads
high_conf_windows = {
    "positional": 16,
    "mixed": 64, 
    "gathering": 512
}
```

### Traditional Head-Aware (Groups C/D)
```python
window_override = {
    "positional": 8-64,   # small
    "mixed": 32-128,      # medium
    "gathering": 508/-1   # large/full
}
```

## References

- StreamingLLM: Efficient Streaming Language Models with Attention Sinks
- Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling

---
*Report generated from ablation experiments on Pythia-2.8b*
*Last updated: December 2024*


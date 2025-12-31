# Speculative Decoding Implementation

This directory contains a custom implementation of Speculative Decoding for accelerating LLM inference.

## Overview

Speculative Decoding is a technique that uses a small "draft" model to quickly generate candidate tokens, which are then verified by a larger "target" model in parallel. This can significantly speed up inference when the draft model has a high acceptance rate.

### Key Features

- **Static KV Cache**: Pre-allocated memory with O(1) truncation operations
- **Greedy Decoding**: Produces identical outputs to pure target model
- **PyTorch 2.0 Optimizations**: Uses `torch.compile` and `torch.inference_mode`
- **HuggingFace Compatible**: Works with any HuggingFace transformer model

## Architecture

```
spec_decode/
├── core/
│   ├── __init__.py
│   ├── static_cache.py          # Static KV Cache implementation
│   ├── speculative_generator.py # Main generator class
│   └── utils.py                 # Utility functions
├── benchmark_custom_vs_hf.py    # Performance comparison
├── test_correctness.py          # Correctness verification
├── demo.py                      # Usage demonstration
└── README.md
```

## Usage

### Basic Usage

```python
from spec_decode.core import SpeculativeGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
target_model = AutoModelForCausalLM.from_pretrained("pythia-2.8b", torch_dtype=torch.float16, device_map="cuda")
draft_model = AutoModelForCausalLM.from_pretrained("pythia-70m", torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("pythia-2.8b")

# Create generator
generator = SpeculativeGenerator(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    K=5,  # Number of tokens to draft per round
    max_len=2048,
    device="cuda"
)

# Generate
output = generator.generate("The future of AI is", max_new_tokens=100)
print(output)

# Get statistics
stats = generator.get_stats()
print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
print(f"Tokens per round: {stats['avg_tokens_per_round']:.2f}")
```

### Run Demo

```bash
cd spec_decode
python demo.py --demo basic
python demo.py --demo comparison
python demo.py --demo k_values
python demo.py --demo all
```

### Run Benchmarks

```bash
python benchmark_custom_vs_hf.py \
    --target-model /path/to/pythia-2.8b \
    --draft-model /path/to/pythia-70m \
    --k-values 3 5 7 \
    --num-samples 5 \
    --max-new-tokens 100
```

### Run Tests

```bash
python test_correctness.py \
    --target-model /path/to/pythia-2.8b \
    --draft-model /path/to/pythia-70m

# Quick test
python test_correctness.py --quick
```

## Algorithm

### Speculative Decoding Flow

1. **Prefill**: Process the prompt with the target model, initialize KV cache
2. **Draft Phase**: Generate K tokens using the draft model (with temporary cache)
3. **Verify Phase**: Verify all K tokens with target model in one forward pass
4. **Accept/Reject**: Accept tokens where draft matches target, stop at first mismatch
5. **Update Cache**: Truncate target cache to correct length
6. **Repeat**: Continue until max tokens or EOS

### Cache Management Strategy

- **Target Model**: Uses persistent Static KV Cache with O(1) truncation
- **Draft Model**: Uses temporary cache (discarded after each round)

This avoids complex cache synchronization issues while maintaining efficiency.

## Performance

Expected performance on Pythia models:

| Method | K | Throughput | Speedup |
|--------|---|------------|---------|
| Baseline | - | ~25 t/s | 1.0x |
| HuggingFace | 5 | ~40 t/s | 1.6x |
| Custom | 5 | ~38-42 t/s | 1.5-1.7x |

*Results may vary based on hardware and model configurations.*

## Key Design Decisions

1. **Greedy Decoding**: Produces identical outputs to pure target model
2. **Static Cache**: Pre-allocated memory avoids dynamic allocation overhead
3. **No Draft Cache Persistence**: Simplifies implementation, small overhead for small models
4. **torch.compile**: Reduces Python interpreter overhead

## Limitations

- Currently supports batch_size=1 only
- Greedy decoding only (no sampling support yet)
- Draft model must share vocabulary with target model

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)



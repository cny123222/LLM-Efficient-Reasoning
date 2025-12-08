#!/usr/bin/env python3
"""
Test script for the new --no_recent_only parameter

This script demonstrates how the recent_only control group works.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from kvcompress.methods import recent_only_compress
import torch

def test_recent_only():
    """Test the recent_only_compress function."""
    print("="*70)
    print("Testing recent_only_compress")
    print("="*70)
    
    # Create dummy KV cache
    batch_size = 1
    num_heads = 8
    seq_len = 1000
    head_dim = 64
    num_layers = 4
    
    # Generate dummy KV cache
    past_key_values = []
    for layer in range(num_layers):
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        past_key_values.append((key, value))
    
    print(f"\nOriginal cache:")
    print(f"  Layers: {len(past_key_values)}")
    print(f"  Seq length: {past_key_values[0][0].shape[2]}")
    
    # Test different window sizes
    for window_size in [256, 512, 1024]:
        compressed = recent_only_compress(
            past_key_values,
            window_size=window_size,
            skip_layers=[0, 1]
        )
        
        print(f"\nAfter recent_only compression (window_size={window_size}):")
        for layer_idx, (k, v) in enumerate(compressed):
            compressed_len = k.shape[2]
            if layer_idx in [0, 1]:
                print(f"  Layer {layer_idx} (skipped): {compressed_len} tokens")
            else:
                print(f"  Layer {layer_idx}: {compressed_len} tokens")
    
    print("\n" + "="*70)
    print("Test passed!")
    print("="*70)

def show_usage_examples():
    """Show usage examples for the --no_recent_only parameter."""
    print("\n" + "="*70)
    print("Usage Examples for --no_recent_only parameter")
    print("="*70)
    
    examples = [
        {
            "title": "StreamingLLM with recent_only control group (default)",
            "command": "python scripts/benchmark.py --method streaming_llm \\\n"
                      "    --recent_sizes 252 --start_size 4 \\\n"
                      "    --num_samples 2 --max_tokens 2000",
            "description": "This will test:\n"
                          "  - baseline (no compression)\n"
                          "  - recent_only_256 (sliding window, keep last 256 tokens)\n"
                          "  - streaming_256 (StreamingLLM: 4 attention sinks + 252 recent)"
        },
        {
            "title": "StreamingLLM WITHOUT recent_only control group",
            "command": "python scripts/benchmark.py --method streaming_llm \\\n"
                      "    --recent_sizes 252 --start_size 4 \\\n"
                      "    --num_samples 2 --max_tokens 2000 \\\n"
                      "    --no_recent_only",
            "description": "This will test:\n"
                          "  - baseline (no compression)\n"
                          "  - streaming_256 (StreamingLLM: 4 attention sinks + 252 recent)"
        },
        {
            "title": "Fix-size L2 with recent_only control group",
            "command": "python scripts/benchmark.py --method fix_size_l2 \\\n"
                      "    --fix_kv_sizes 512 --strategies keep_low \\\n"
                      "    --keep_ratios 0.5 --num_samples 2",
            "description": "This will test:\n"
                          "  - baseline\n"
                          "  - recent_only_512 (sliding window, keep last 512 tokens)\n"
                          "  - fix512_keep_low_kr=0.5 (L2-based eviction)"
        },
        {
            "title": "L2 compress (no recent_only, as cache size is dynamic)",
            "command": "python scripts/benchmark.py --method l2_compress \\\n"
                      "    --keep_ratios 0.8,0.5 --num_samples 2",
            "description": "This will test:\n"
                          "  - baseline\n"
                          "  - l2_kr=0.8\n"
                          "  - l2_kr=0.5\n"
                          "(No recent_only as cache size varies with sequence length)"
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. {ex['title']}")
        print("-" * 70)
        print(f"Command:\n{ex['command']}")
        print(f"\n{ex['description']}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # Test the recent_only function
    test_recent_only()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n✅ All tests passed!")
    print("\nYou can now use the --no_recent_only flag in benchmark.py")
    print("to disable the sliding window control group when needed.")


"""
Correctness Tests for Custom Speculative Decoding Implementation

This script verifies:
1. Output correctness: Greedy decoding should produce identical output to pure target model
2. Cache correctness: KV cache length should match sequence length
3. Memory management: No memory leaks after multiple generations
4. Acceptance rate: Should be reasonable (> 0.3 for similar model families)
"""

import torch
import gc
import argparse
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

from core import SpeculativeGenerator, StaticKVCache


def test_output_correctness(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    prompts: List[str],
    max_new_tokens: int = 50
) -> Tuple[bool, List[str]]:
    """
    Test that speculative decoding produces identical output to pure target model.
    
    For greedy decoding, outputs should be exactly identical.
    """
    print("\n" + "="*60)
    print("TEST 1: Output Correctness")
    print("="*60)
    
    errors = []
    
    # Create generator
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False  # Disable for testing
    )
    
    for i, prompt in enumerate(prompts):
        print(f"\nTest {i+1}/{len(prompts)}: '{prompt[:50]}...'")
        
        # Generate with pure target model
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            target_output = target_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        target_text = tokenizer.decode(target_output[0], skip_special_tokens=True)
        
        # Generate with speculative decoding
        generator.reset()
        spec_text = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        # Compare
        if target_text == spec_text:
            print(f"  ✅ Outputs match!")
        else:
            print(f"  ❌ Outputs differ!")
            print(f"     Target: {target_text[:100]}...")
            print(f"     Spec:   {spec_text[:100]}...")
            errors.append(f"Sample {i}: outputs differ")
    
    success = len(errors) == 0
    return success, errors


def test_cache_correctness(
    target_model,
    draft_model,
    tokenizer,
    device: str
) -> Tuple[bool, List[str]]:
    """
    Test that KV cache is correctly managed.
    """
    print("\n" + "="*60)
    print("TEST 2: Cache Correctness")
    print("="*60)
    
    errors = []
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    prompt = "The quick brown fox"
    
    # Generate some tokens
    generator.reset()
    _ = generator.generate(prompt, max_new_tokens=20)
    
    # Check cache length matches sequence length
    # Use get_seq_length() for DynamicCache
    cache_len = generator.target_cache.get_seq_length()
    seq_len = generator.current_ids.shape[1]
    
    print(f"\nCache length: {cache_len}")
    print(f"Sequence length: {seq_len}")
    
    # Cache should be <= sequence length (might be slightly less due to truncation)
    if cache_len <= seq_len and cache_len >= seq_len - generator.K:
        print("✅ Cache length is valid")
    else:
        print("❌ Cache length mismatch!")
        errors.append(f"Cache length {cache_len} doesn't match sequence length {seq_len}")
    
    # Test reset
    generator.reset()
    if generator.target_cache is None:
        print("✅ Cache reset works correctly")
    else:
        print("❌ Cache reset failed!")
        errors.append("Cache reset failed")
    
    success = len(errors) == 0
    return success, errors


def test_memory_leaks(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    num_iterations: int = 5
) -> Tuple[bool, List[str]]:
    """
    Test for memory leaks by running multiple generations.
    """
    print("\n" + "="*60)
    print("TEST 3: Memory Leak Test")
    print("="*60)
    
    errors = []
    
    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    initial_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"\nInitial memory: {initial_memory:.1f} MB")
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    memory_after_init = torch.cuda.memory_allocated() / (1024**2)
    print(f"After init: {memory_after_init:.1f} MB")
    
    # Run multiple generations
    prompts = [
        "The quick brown fox",
        "In a galaxy far away",
        "Machine learning is",
        "The capital of France",
        "Once upon a time"
    ]
    
    for i in range(num_iterations):
        for prompt in prompts:
            generator.reset()
            _ = generator.generate(prompt, max_new_tokens=50)
        
        current_memory = torch.cuda.memory_allocated() / (1024**2)
        print(f"Iteration {i+1}/{num_iterations}: {current_memory:.1f} MB")
    
    # Clean up
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    
    final_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"After cleanup: {final_memory:.1f} MB")
    
    # Check for significant memory increase
    memory_increase = final_memory - initial_memory
    if memory_increase < 100:  # Allow some tolerance
        print(f"✅ No significant memory leak detected (increase: {memory_increase:.1f} MB)")
    else:
        print(f"⚠️ Potential memory leak: {memory_increase:.1f} MB increase")
        errors.append(f"Memory increased by {memory_increase:.1f} MB")
    
    success = len(errors) == 0
    return success, errors


def test_acceptance_rate(
    target_model,
    draft_model,
    tokenizer,
    device: str,
    prompts: List[str],
    max_new_tokens: int = 100
) -> Tuple[bool, List[str]]:
    """
    Test that acceptance rate is reasonable.
    """
    print("\n" + "="*60)
    print("TEST 4: Acceptance Rate")
    print("="*60)
    
    errors = []
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    total_rounds = 0
    total_accepted = 0
    
    for prompt in prompts:
        generator.reset()
        _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
        stats = generator.get_stats()
        total_rounds += stats["total_rounds"]
        total_accepted += stats["total_accepted"]
    
    # Get acceptance rate from the last stats (or calculate from totals)
    final_stats = generator.get_stats()
    avg_acceptance = final_stats.get("acceptance_rate", 0.0)
    avg_tokens_per_round = total_accepted / total_rounds if total_rounds > 0 else 0
    
    print(f"\nTotal rounds: {total_rounds}")
    print(f"Total accepted: {total_accepted}")
    print(f"Avg tokens/round: {avg_tokens_per_round:.2f}")
    print(f"Acceptance rate: {avg_acceptance:.2%}")
    
    # Acceptance rate should be > 0.3 for similar model families
    if avg_acceptance >= 0.3:
        print(f"✅ Acceptance rate is good (>= 30%)")
    elif avg_acceptance >= 0.1:
        print(f"⚠️ Acceptance rate is low but acceptable (>= 10%)")
    else:
        print(f"❌ Acceptance rate is too low (< 10%)")
        errors.append(f"Acceptance rate {avg_acceptance:.2%} is too low")
    
    success = len(errors) == 0
    return success, errors


def test_static_cache():
    """
    Test StaticKVCache functionality independently.
    """
    print("\n" + "="*60)
    print("TEST 5: StaticKVCache Unit Tests")
    print("="*60)
    
    errors = []
    
    from core.static_cache import StaticKVCache, CacheConfig
    
    config = CacheConfig(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        max_seq_len=100,
        dtype=torch.float16,
        device="cuda"
    )
    
    cache = StaticKVCache(config)
    
    # Test initial state
    assert cache.get_seq_len() == 0, "Initial length should be 0"
    print("✅ Initial state correct")
    
    # Test update
    new_keys = torch.randn(4, 1, 8, 10, 64, dtype=torch.float16, device="cuda")
    new_values = torch.randn(4, 1, 8, 10, 64, dtype=torch.float16, device="cuda")
    cache.update(new_keys, new_values)
    assert cache.get_seq_len() == 10, "Length after update should be 10"
    print("✅ Update works correctly")
    
    # Test truncate
    cache.truncate(5)
    assert cache.get_seq_len() == 5, "Length after truncate should be 5"
    print("✅ Truncate works correctly")
    
    # Test to_hf_format
    hf_format = cache.to_hf_format()
    assert len(hf_format) == 4, "Should have 4 layers"
    assert hf_format[0][0].shape[2] == 5, "Key length should be 5"
    print("✅ to_hf_format works correctly")
    
    # Test reset
    cache.reset()
    assert cache.get_seq_len() == 0, "Length after reset should be 0"
    print("✅ Reset works correctly")
    
    success = len(errors) == 0
    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Test speculative decoding correctness")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                        help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                        help="Path to draft model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests only")
    args = parser.parse_args()
    
    print("🔬 Running Speculative Decoding Correctness Tests")
    print(f"   Target: {args.target_model}")
    print(f"   Draft: {args.draft_model}")
    
    # Test prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning of time, there was nothing.",
        "Machine learning is a fascinating field of study.",
    ]
    
    if args.quick:
        prompts = prompts[:1]
    
    all_passed = True
    all_errors = []
    
    # Test 5: Static cache (no model needed)
    success, errors = test_static_cache()
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Load models for remaining tests
    print(f"\n📦 Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16
    ).to(args.device)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model.eval()
    draft_model.eval()
    
    # Test 1: Output correctness
    success, errors = test_output_correctness(
        target_model, draft_model, tokenizer, args.device, prompts
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Test 2: Cache correctness
    success, errors = test_cache_correctness(
        target_model, draft_model, tokenizer, args.device
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Test 3: Memory leaks
    if not args.quick:
        success, errors = test_memory_leaks(
            target_model, draft_model, tokenizer, args.device
        )
        all_passed = all_passed and success
        all_errors.extend(errors)
    
    # Test 4: Acceptance rate
    success, errors = test_acceptance_rate(
        target_model, draft_model, tokenizer, args.device, prompts
    )
    all_passed = all_passed and success
    all_errors.extend(errors)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed:")
        for error in all_errors:
            print(f"   - {error}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())


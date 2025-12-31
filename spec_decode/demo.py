"""
Demo Script for Custom Speculative Decoding

This script demonstrates how to use the custom speculative decoding implementation.
"""

import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

from core import SpeculativeGenerator


def demo_basic_generation():
    """Basic demonstration of speculative decoding."""
    print("\n" + "="*60)
    print("DEMO: Basic Speculative Decoding")
    print("="*60)
    
    # Default model paths (adjust as needed)
    target_model_id = "/mnt/disk1/models/pythia-2.8b"
    draft_model_id = "/mnt/disk1/models/pythia-70m"
    device = "cuda:0"
    
    print(f"\n📦 Loading models...")
    print(f"   Target: {target_model_id}")
    print(f"   Draft: {draft_model_id}")
    
    # Load models
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.float16
    ).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create generator
    print(f"\n🔧 Creating Speculative Generator (K=5)...")
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False  # Disable for demo (faster startup)
    )
    
    print(f"   Target cache memory: {generator.target_cache.get_memory_usage_mb():.1f} MB")
    
    # Generate
    prompt = "The future of artificial intelligence is"
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"\n🚀 Generating...")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    output = generator.generate(prompt, max_new_tokens=100, verbose=False)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"\n📖 Generated text:")
    print("-"*60)
    print(output)
    print("-"*60)
    
    # Statistics
    stats = generator.get_stats()
    print(f"\n📊 Statistics:")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Total rounds: {stats['total_rounds']}")
    print(f"   Avg tokens/round: {stats['avg_tokens_per_round']:.2f}")
    print(f"   Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {stats['total_tokens']/elapsed:.1f} tokens/s")


def demo_comparison_with_baseline():
    """Compare speculative decoding with baseline."""
    print("\n" + "="*60)
    print("DEMO: Comparison with Baseline")
    print("="*60)
    
    target_model_id = "/mnt/disk1/models/pythia-2.8b"
    draft_model_id = "/mnt/disk1/models/pythia-70m"
    device = "cuda:0"
    
    print(f"\n📦 Loading models...")
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generator = SpeculativeGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_len=2048,
        device=device,
        use_compile=False
    )
    
    prompt = "Machine learning is revolutionizing the way we"
    max_new_tokens = 100
    
    # Baseline (pure target model)
    print(f"\n🏃 Running Baseline...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.inference_mode():
        baseline_output = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    baseline_time = time.perf_counter() - start
    baseline_tokens = baseline_output.shape[1] - inputs.input_ids.shape[1]
    baseline_throughput = baseline_tokens / baseline_time
    
    print(f"   Tokens: {baseline_tokens}")
    print(f"   Time: {baseline_time:.2f}s")
    print(f"   Throughput: {baseline_throughput:.1f} tokens/s")
    
    # Speculative decoding
    print(f"\n🚀 Running Speculative Decoding (K=5)...")
    generator.reset()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    spec_output = generator.generate(prompt, max_new_tokens=max_new_tokens)
    
    torch.cuda.synchronize()
    spec_time = time.perf_counter() - start
    spec_tokens = generator.get_stats()['total_tokens']
    spec_throughput = spec_tokens / spec_time
    
    print(f"   Tokens: {spec_tokens}")
    print(f"   Time: {spec_time:.2f}s")
    print(f"   Throughput: {spec_throughput:.1f} tokens/s")
    print(f"   Acceptance rate: {generator.get_acceptance_rate():.2%}")
    
    # Comparison
    speedup = spec_throughput / baseline_throughput
    print(f"\n📊 Speedup: {speedup:.2f}x")
    
    # Verify outputs match
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    if baseline_text == spec_output:
        print("✅ Outputs match!")
    else:
        print("⚠️ Outputs differ slightly")


def demo_different_k_values():
    """Demonstrate effect of different K values."""
    print("\n" + "="*60)
    print("DEMO: Effect of Different K Values")
    print("="*60)
    
    target_model_id = "/mnt/disk1/models/pythia-2.8b"
    draft_model_id = "/mnt/disk1/models/pythia-70m"
    device = "cuda:0"
    
    print(f"\n📦 Loading models...")
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "The history of computer science begins with"
    max_new_tokens = 100
    
    print(f"\n📝 Prompt: '{prompt}'")
    print(f"\n{'K':<5} {'Throughput':<15} {'Accept Rate':<15} {'Tokens/Round':<15}")
    print("-"*50)
    
    for K in [3, 5, 7, 10]:
        generator = SpeculativeGenerator(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=K,
            max_len=2048,
            device=device,
            use_compile=False
        )
        
        # Warmup
        generator.generate("Warmup", max_new_tokens=10)
        
        # Benchmark
        generator.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        stats = generator.get_stats()
        throughput = stats['total_tokens'] / elapsed
        
        print(f"{K:<5} {throughput:<15.1f} {stats['acceptance_rate']:<15.2%} {stats['avg_tokens_per_round']:<15.2f}")
        
        del generator
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Demo")
    parser.add_argument("--demo", type=str, default="basic",
                        choices=["basic", "comparison", "k_values", "all"],
                        help="Which demo to run")
    args = parser.parse_args()
    
    print("🎯 Speculative Decoding Demo")
    
    if args.demo == "basic" or args.demo == "all":
        demo_basic_generation()
    
    if args.demo == "comparison" or args.demo == "all":
        demo_comparison_with_baseline()
    
    if args.demo == "k_values" or args.demo == "all":
        demo_different_k_values()
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()


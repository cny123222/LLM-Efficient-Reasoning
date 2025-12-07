"""
Evaluation Module for KnormPress

This module implements PPL and Accuracy evaluation with KV cache compression.
The implementation exactly follows the original paper's eval_lm_quick.py logic.

Key implementation points from the paper:
1. Process one token at a time (autoregressive)
2. Use CrossEntropyLoss(reduction="none") for per-token NLL
3. Apply compression after EVERY forward pass (if keep_ratio < 1.0)
4. PPL = exp(mean(nlls))
5. Accuracy = num_correct / num_tokens
"""

from typing import Dict, List, Optional, Union
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import DynamicCache

from .compress import l2_compress, fix_size_l2_compress, to_dynamic_cache, get_compress_fn


def evaluate_with_compression(
    model,
    tokenizer,
    text: str,
    keep_ratio: float = 1.0,
    prune_after: int = 100,  # Start compressing early for meaningful evaluation
    skip_layers: List[int] = [0, 1],
    max_tokens: int = 3000,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate PPL and Accuracy with KV cache compression.
    
    This function exactly replicates the evaluation logic from the original
    paper's eval_lm_quick.py (lines 92-132).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0)
        prune_after: Only compress if cache length > this value
        skip_layers: Layer indices to skip compression
        max_tokens: Maximum number of tokens to evaluate
        device: Device to use (auto-detected if None)
        show_progress: Whether to show progress bar
    
    Returns:
        Dict containing:
        - perplexity: The perplexity score
        - accuracy: Next token prediction accuracy
        - num_tokens: Number of tokens evaluated
        - final_cache_size: Final KV cache size after compression
    
    Example:
        >>> results = evaluate_with_compression(
        ...     model, tokenizer, text,
        ...     keep_ratio=0.8,
        ...     prune_after=1000
        ... )
        >>> print(f"PPL: {results['perplexity']:.2f}")
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "num_tokens": 0,
            "final_cache_size": 0
        }
    
    # Loss function for per-token NLL (matching paper)
    loss_fn = CrossEntropyLoss(reduction="none")
    
    # Initialize
    past_key_values = None
    nlls = []
    num_correct = []
    
    model.eval()
    
    # Create progress bar if requested
    token_range = range(seq_len - 1)
    if show_progress:
        token_range = tqdm(token_range, desc="Evaluating")
    
    with torch.inference_mode():
        for idx in token_range:
            # Current token (single token input, as in original paper)
            current_token = input_ids[:, idx:idx+1]
            
            # Target token (next token)
            target = input_ids[:, idx+1:idx+2].view(-1)
            
            # Forward pass
            outputs = model(
                current_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Get logits for next token prediction
            logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
            
            # Calculate NLL for this token (matching paper)
            nll = loss_fn(logits, target)
            nlls.append(nll.item())
            
            # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)
            num_correct.append((predicted == target).int().item())
            
            # Update progress bar
            if show_progress:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                current_acc = sum(num_correct) / len(num_correct)
                token_range.set_description(
                    f"PPL: {current_ppl:.2f}, Acc: {current_acc:.2%}"
                )
            
            # Get KV cache from outputs
            past_key_values = outputs.past_key_values
            
            # Apply compression if keep_ratio < 1.0
            if keep_ratio < 1.0 and past_key_values is not None:
                # Convert to list format for compression
                if hasattr(past_key_values, 'to_legacy_cache'):
                    kv_list = past_key_values.to_legacy_cache()
                else:
                    kv_list = list(past_key_values)
                
                # Get size before compression (layer 0, which is in skip_layers)
                size_before_l0 = kv_list[0][0].size(2) if kv_list else 0
                # Get size before for a compressed layer (layer 2)
                size_before_l2 = kv_list[2][0].size(2) if len(kv_list) > 2 else 0
                
                # Apply L2 compression
                compressed_kv = l2_compress(
                    kv_list,
                    keep_ratio=keep_ratio,
                    prune_after=prune_after,
                    skip_layers=skip_layers,
                )
                
                # Get size after compression
                size_after_l0 = compressed_kv[0][0].size(2) if compressed_kv else 0
                size_after_l2 = compressed_kv[2][0].size(2) if len(compressed_kv) > 2 else 0
                
                # Print compression info at key points
                if size_before_l2 != size_after_l2 and (idx == 101 or idx == 500 or idx == 1000):
                    print(f"    [Token {idx}] L0(skip): {size_before_l0}->{size_after_l0} | "
                          f"L2(compress): {size_before_l2}->{size_after_l2} "
                          f"(kept {size_after_l2/size_before_l2:.1%})")
                
                # Convert back to DynamicCache for next iteration
                past_key_values = to_dynamic_cache(compressed_kv)
    
    # Calculate final metrics (matching paper)
    perplexity = torch.exp(torch.tensor(nlls).mean()).item()
    accuracy = sum(num_correct) / len(num_correct)
    
    # Get final cache size
    final_cache_size = 0
    if past_key_values is not None:
        if hasattr(past_key_values, 'to_legacy_cache'):
            kv_list = past_key_values.to_legacy_cache()
        else:
            kv_list = list(past_key_values)
        if kv_list:
            final_cache_size = kv_list[0][0].size(2)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": len(nlls),
        "final_cache_size": final_cache_size
    }


def evaluate_baseline(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 3000,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> Dict[str, float]:
    """
    Evaluate baseline PPL and Accuracy without compression.
    
    This is a convenience wrapper for evaluate_with_compression with keep_ratio=1.0.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        max_tokens: Maximum number of tokens to evaluate
        device: Device to use
        show_progress: Whether to show progress bar
    
    Returns:
        Dict with perplexity, accuracy, num_tokens, final_cache_size
    """
    return evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        keep_ratio=1.0,  # No compression
        prune_after=999999,  # Never trigger
        skip_layers=[],
        max_tokens=max_tokens,
        device=device,
        show_progress=show_progress,
    )


def compare_compression_levels(
    model,
    tokenizer,
    text: str,
    keep_ratios: List[float] = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3],
    prune_after: int = 1000,
    skip_layers: List[int] = [0, 1],
    max_tokens: int = 3000,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Compare PPL and Accuracy across different compression levels.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        keep_ratios: List of keep_ratio values to test
        prune_after: Only compress if cache length > this value
        skip_layers: Layer indices to skip compression
        max_tokens: Maximum number of tokens to evaluate
        device: Device to use
    
    Returns:
        List of result dicts, one per keep_ratio
    """
    results = []
    
    for keep_ratio in keep_ratios:
        print(f"\nEvaluating keep_ratio={keep_ratio:.1f}...")
        
        result = evaluate_with_compression(
            model=model,
            tokenizer=tokenizer,
            text=text,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            skip_layers=skip_layers,
            max_tokens=max_tokens,
            device=device,
            show_progress=True,
        )
        
        result['keep_ratio'] = keep_ratio
        results.append(result)
        
        print(f"  PPL: {result['perplexity']:.2f}")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Tokens evaluated: {result['num_tokens']}")
        print(f"  Final cache size: {result['final_cache_size']}")
    
    return results


def evaluate_fix_size_compression(
    model,
    tokenizer,
    text: str,
    fix_kv_size: int = 512,
    keep_ratio: float = 0.0,
    strategy: str = "keep_low",
    skip_layers: List[int] = [0, 1],
    max_tokens: int = 3000,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate PPL and Accuracy with fixed-size KV cache compression.
    
    This uses the fix_size_l2_compress method which maintains a fixed
    maximum cache size by evicting tokens.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        fix_kv_size: Maximum KV cache size to maintain
        keep_ratio: Fraction of fix_kv_size to protect (recent tokens)
        strategy: Eviction strategy ("keep_low", "keep_high", "random")
        skip_layers: Layer indices to skip compression
        max_tokens: Maximum number of tokens to evaluate
        device: Device to use
        show_progress: Whether to show progress bar
    
    Returns:
        Dict containing perplexity, accuracy, num_tokens, final_cache_size
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "num_tokens": 0,
            "final_cache_size": 0
        }
    
    loss_fn = CrossEntropyLoss(reduction="none")
    
    past_key_values = None
    nlls = []
    num_correct = []
    
    model.eval()
    
    token_range = range(seq_len - 1)
    if show_progress:
        token_range = tqdm(token_range, desc=f"Evaluating (fix_size={fix_kv_size}, strategy={strategy})")
    
    with torch.inference_mode():
        for idx in token_range:
            current_token = input_ids[:, idx:idx+1]
            target = input_ids[:, idx+1:idx+2].view(-1)
            
            outputs = model(
                current_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
            
            nll = loss_fn(logits, target)
            nlls.append(nll.item())
            
            predicted = torch.argmax(logits, dim=-1)
            num_correct.append((predicted == target).int().item())
            
            if show_progress:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                current_acc = sum(num_correct) / len(num_correct)
                token_range.set_description(
                    f"PPL: {current_ppl:.2f}, Acc: {current_acc:.2%}"
                )
            
            past_key_values = outputs.past_key_values
            
            # Apply fixed-size compression
            if past_key_values is not None:
                if hasattr(past_key_values, 'to_legacy_cache'):
                    kv_list = past_key_values.to_legacy_cache()
                else:
                    kv_list = list(past_key_values)
                
                # Find a layer that will be compressed (not in skip_layers)
                check_layer = 0
                for l in range(len(kv_list)):
                    if l not in skip_layers:
                        check_layer = l
                        break
                
                # Check if compression needed (check non-skip layer)
                current_size = kv_list[check_layer][0].size(2) if kv_list else 0
                
                if current_size > fix_kv_size:
                    # Get sizes before compression
                    size_before_skip = kv_list[0][0].size(2) if kv_list else 0
                    size_before_compress = kv_list[check_layer][0].size(2) if len(kv_list) > check_layer else 0
                    
                    compressed_kv = fix_size_l2_compress(
                        kv_list,
                        fix_kv_size=fix_kv_size,
                        keep_ratio=keep_ratio,
                        strategy=strategy,
                        skip_layers=skip_layers,
                    )
                    
                    size_after_skip = compressed_kv[0][0].size(2) if compressed_kv else 0
                    size_after_compress = compressed_kv[check_layer][0].size(2) if len(compressed_kv) > check_layer else 0
                    
                    # Print compression info at key points
                    if idx == fix_kv_size + 1 or idx == 500 or idx == 1000:
                        print(f"\n    [Token {idx}] L0(skip): {size_before_skip}->{size_after_skip} | "
                              f"L{check_layer}(compress): {size_before_compress}->{size_after_compress}")
                    
                    past_key_values = to_dynamic_cache(compressed_kv)
    
    perplexity = torch.exp(torch.tensor(nlls).mean()).item()
    accuracy = sum(num_correct) / len(num_correct)
    
    final_cache_size = 0
    if past_key_values is not None:
        if hasattr(past_key_values, 'to_legacy_cache'):
            kv_list = past_key_values.to_legacy_cache()
        else:
            kv_list = list(past_key_values)
        if kv_list:
            final_cache_size = kv_list[0][0].size(2)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": len(nlls),
        "final_cache_size": final_cache_size,
        "fix_kv_size": fix_kv_size,
        "strategy": strategy,
    }


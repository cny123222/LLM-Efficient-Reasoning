"""
Recent-Only (Sliding Window) KV Cache Compression

This module implements a simple sliding window strategy that only keeps
the most recent N tokens in the KV cache. This serves as a baseline
comparison for more sophisticated compression methods.
"""

from typing import List, Tuple, Union
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def recent_only_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    window_size: int = 512,
    skip_layers: List[int] = [0, 1],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sliding window compression: only keep the most recent tokens.
    
    This is the simplest KV cache compression strategy, serving as a baseline
    to compare against more sophisticated methods like:
    - StreamingLLM (recent + attention sinks)
    - L2-based eviction (recent + important tokens)
    
    Algorithm:
    1. If cache size <= window_size, no eviction needed
    2. Otherwise, keep only the last window_size tokens
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        window_size: Number of recent tokens to keep
        skip_layers: Layer indices to skip compression
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with at most window_size tokens per layer
    
    Example:
        >>> # Keep only the last 512 tokens
        >>> compressed = recent_only_compress(
        ...     past_key_values,
        ...     window_size=512
        ... )
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # No eviction needed if within limit
        if seq_len <= window_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        # Keep only the last window_size tokens
        final_keys = keys[:, :, -window_size:, :]
        final_values = values[:, :, -window_size:, :]
        
        past_key_values[layer_idx] = (final_keys, final_values)
    
    return past_key_values


__all__ = ['recent_only_compress']


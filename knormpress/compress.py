"""
KV Cache Compression Methods

This module implements multiple KV cache compression strategies:
1. l2_compress: Original KnormPress ratio-based compression
2. fix_size_l2_compress: Fixed-size KV cache with eviction strategies

Reference: "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
"""

from typing import List, Tuple, Union, Literal
from math import ceil
import torch
from transformers import DynamicCache


# ============================================================================
# Original L2 Compress (Ratio-based)
# ============================================================================

def l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    keep_ratio: float = 1.0,
    prune_after: int = 1000,
    skip_layers: List[int] = [0, 1],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compress KV cache by keeping tokens with the lowest L2 norms.
    
    This is the original KnormPress algorithm that compresses by ratio.
    
    Args:
        past_key_values: KV cache
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0)
        prune_after: Only compress if sequence length > this value
        skip_layers: Layer indices to skip compression
    
    Returns:
        Compressed KV cache as list of (key, value) tuples
    """
    # Convert DynamicCache to list format if needed
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    past_key_values = list(past_key_values)
    
    if keep_ratio >= 1.0:
        return past_key_values
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        if seq_len <= prune_after:
            continue
        
        if layer_idx in skip_layers:
            continue
        
        tokens_to_keep = ceil(keep_ratio * seq_len)
        
        if tokens_to_keep >= seq_len:
            continue
        
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Compute L2 norm
        token_norms = torch.norm(keys, p=2, dim=-1)
        
        # Sort by norm (ascending = low norm = important)
        sorted_indices = token_norms.argsort(dim=-1)
        
        # Select top tokens
        indices_to_keep = sorted_indices[:, :, :tokens_to_keep]
        
        # Maintain temporal order
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand and gather
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep, head_dim
        )
        
        compressed_keys = torch.gather(keys, dim=2, index=indices_expanded)
        compressed_values = torch.gather(values, dim=2, index=indices_expanded)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


# ============================================================================
# Fixed-Size L2 Compress (with eviction strategies)
# ============================================================================

def fix_size_l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    fix_kv_size: int = 1024,
    keep_ratio: float = 0.0,
    strategy: Literal["keep_low", "keep_high", "random"] = "keep_low",
    skip_layers: List[int] = [0, 1],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fixed-size KV cache compression with eviction strategies.
    
    This method maintains a fixed maximum KV cache size by evicting tokens
    when the cache exceeds fix_kv_size. Recent tokens (controlled by keep_ratio)
    are never evicted; eviction only happens in the older portion.
    
    Algorithm:
    1. If cache size <= fix_kv_size, no eviction needed
    2. Calculate protected_length = fix_kv_size * keep_ratio (recent tokens to keep)
    3. Eviction zone = tokens[0 : seq_len - protected_length]
    4. From eviction zone, keep (fix_kv_size - protected_length) tokens based on strategy
    5. Combine kept eviction zone tokens + protected recent tokens
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        fix_kv_size: Maximum number of tokens to keep in cache
        keep_ratio: Fraction of fix_kv_size to protect (most recent tokens)
                   Default 0.0 means all tokens can be evicted
                   Example: keep_ratio=0.2, fix_kv_size=1000 -> last 200 tokens protected
        strategy: Eviction strategy
                  - "keep_low": Keep tokens with low L2 norm (important tokens)
                  - "keep_high": Keep tokens with high L2 norm
                  - "random": Random eviction
        skip_layers: Layer indices to skip compression
    
    Returns:
        Compressed KV cache with at most fix_kv_size tokens per layer
    
    Example:
        >>> # Keep max 512 tokens, protect last 20% (102 tokens)
        >>> compressed = fix_size_l2_compress(
        ...     past_key_values,
        ...     fix_kv_size=512,
        ...     keep_ratio=0.2,
        ...     strategy="keep_low"
        ... )
    """
    # Convert DynamicCache to list format if needed
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    past_key_values = list(past_key_values)
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # No eviction needed if within limit
        if seq_len <= fix_kv_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Calculate protected zone (recent tokens that won't be evicted)
        protected_length = int(fix_kv_size * keep_ratio)
        protected_length = min(protected_length, seq_len)  # Can't protect more than we have
        
        # Eviction zone: everything except the protected recent tokens
        eviction_zone_end = seq_len - protected_length
        
        # How many tokens to keep from eviction zone
        tokens_to_keep_from_eviction = fix_kv_size - protected_length
        
        if tokens_to_keep_from_eviction <= 0:
            # Only keep protected recent tokens
            final_keys = keys[:, :, -protected_length:, :]
            final_values = values[:, :, -protected_length:, :]
            past_key_values[layer_idx] = (final_keys, final_values)
            continue
        
        if eviction_zone_end <= tokens_to_keep_from_eviction:
            # No need to evict, keep all from eviction zone
            continue
        
        # Get eviction zone keys for computing norms
        eviction_keys = keys[:, :, :eviction_zone_end, :]
        eviction_values = values[:, :, :eviction_zone_end, :]
        
        # Select tokens to keep from eviction zone based on strategy
        if strategy == "keep_low":
            # Keep tokens with lowest L2 norm (most important)
            token_norms = torch.norm(eviction_keys, p=2, dim=-1)
            # argsort ascending: lowest norms first
            sorted_indices = token_norms.argsort(dim=-1)
            indices_to_keep = sorted_indices[:, :, :tokens_to_keep_from_eviction]
            
        elif strategy == "keep_high":
            # Keep tokens with highest L2 norm
            token_norms = torch.norm(eviction_keys, p=2, dim=-1)
            # argsort descending: highest norms first
            sorted_indices = token_norms.argsort(dim=-1, descending=True)
            indices_to_keep = sorted_indices[:, :, :tokens_to_keep_from_eviction]
            
        elif strategy == "random":
            # Random selection
            indices_to_keep = torch.stack([
                torch.stack([
                    torch.randperm(eviction_zone_end, device=keys.device)[:tokens_to_keep_from_eviction]
                    for _ in range(num_heads)
                ])
                for _ in range(batch_size)
            ])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # CRITICAL: Sort indices to maintain temporal order
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand indices for gather
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep_from_eviction, head_dim
        )
        
        # Gather selected tokens from eviction zone
        kept_eviction_keys = torch.gather(eviction_keys, dim=2, index=indices_expanded)
        kept_eviction_values = torch.gather(eviction_values, dim=2, index=indices_expanded)
        
        # Get protected recent tokens
        if protected_length > 0:
            protected_keys = keys[:, :, -protected_length:, :]
            protected_values = values[:, :, -protected_length:, :]
            
            # Concatenate: kept eviction tokens + protected recent tokens
            final_keys = torch.cat([kept_eviction_keys, protected_keys], dim=2)
            final_values = torch.cat([kept_eviction_values, protected_values], dim=2)
        else:
            final_keys = kept_eviction_keys
            final_values = kept_eviction_values
        
        past_key_values[layer_idx] = (final_keys, final_values)
    
    return past_key_values


# ============================================================================
# Utility Functions
# ============================================================================

def to_dynamic_cache(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
) -> DynamicCache:
    """Convert list of (key, value) tuples to DynamicCache object."""
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(past_key_values):
        cache.update(keys, values, layer_idx)
    return cache


def get_cache_size_mb(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> float:
    """Calculate KV cache size in megabytes."""
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    total_size = 0
    for keys, values in past_key_values:
        total_size += keys.element_size() * keys.nelement()
        total_size += values.element_size() * values.nelement()
    return total_size / (1024 ** 2)


def get_cache_info(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> dict:
    """Get detailed information about KV cache."""
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    if not past_key_values:
        return {"num_layers": 0, "seq_lengths": [], "total_size_mb": 0}
    
    seq_lengths = [keys.size(2) for keys, values in past_key_values]
    
    return {
        "num_layers": len(past_key_values),
        "seq_lengths": seq_lengths,
        "min_seq_len": min(seq_lengths),
        "max_seq_len": max(seq_lengths),
        "avg_seq_len": sum(seq_lengths) / len(seq_lengths),
        "total_size_mb": get_cache_size_mb(past_key_values)
    }


# ============================================================================
# Compression Factory
# ============================================================================

def get_compress_fn(method: str = "l2_compress"):
    """
    Get compression function by name.
    
    Args:
        method: Compression method name
                - "l2_compress": Original KnormPress ratio-based
                - "fix_size_l2": Fixed-size with L2-based eviction
    
    Returns:
        Compression function
    """
    methods = {
        "l2_compress": l2_compress,
        "fix_size_l2": fix_size_l2_compress,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    return methods[method]

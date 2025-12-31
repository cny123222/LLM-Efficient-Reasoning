"""
Static KV Cache Implementation for Speculative Decoding

This module provides a pre-allocated KV cache that avoids dynamic memory allocation
and enables O(1) truncation operations by only moving pointers.
"""

import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass

try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False


@dataclass
class CacheConfig:
    """Configuration for Static KV Cache"""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class StaticKVCache:
    """
    Pre-allocated Static KV Cache for efficient speculative decoding.
    
    Key Features:
    - Pre-allocates fixed-size GPU memory to avoid dynamic allocation
    - truncate() only moves a pointer, O(1) operation with zero memory copy
    - Compatible with HuggingFace's past_key_values format
    
    Memory Layout:
    - keys: [num_layers, batch_size, num_heads, max_seq_len, head_dim]
    - values: [num_layers, batch_size, num_heads, max_seq_len, head_dim]
    
    Note: This implementation assumes batch_size=1 for simplicity.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize the static KV cache with pre-allocated memory.
        
        Args:
            config: CacheConfig with model parameters
        """
        self.config = config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.dtype = config.dtype
        self.device = config.device
        
        # Current valid length (pointer)
        self.current_len = 0
        
        # Pre-allocate memory for keys and values
        # Shape: [num_layers, batch=1, num_heads, max_seq_len, head_dim]
        self.keys = torch.zeros(
            (self.num_layers, 1, self.num_heads, self.max_seq_len, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.values = torch.zeros_like(self.keys)
        
        # Track memory usage for debugging
        self._memory_allocated = self.keys.numel() * 2 * self.keys.element_size()
    
    @classmethod
    def from_model_config(cls, model_config, max_seq_len: int, device: str = "cuda"):
        """
        Create StaticKVCache from a HuggingFace model config.
        
        Args:
            model_config: HuggingFace model configuration
            max_seq_len: Maximum sequence length to pre-allocate
            device: Device to allocate memory on
            
        Returns:
            StaticKVCache instance
        """
        # Handle different model architectures
        # GPT-NeoX (Pythia) uses these attribute names
        if hasattr(model_config, 'num_hidden_layers'):
            num_layers = model_config.num_hidden_layers
        else:
            num_layers = model_config.n_layer
            
        if hasattr(model_config, 'num_attention_heads'):
            num_heads = model_config.num_attention_heads
        else:
            num_heads = model_config.n_head
            
        if hasattr(model_config, 'hidden_size'):
            head_dim = model_config.hidden_size // num_heads
        else:
            head_dim = model_config.n_embd // num_heads
        
        # Determine dtype from model config if available
        dtype = torch.float16
        if hasattr(model_config, 'torch_dtype') and model_config.torch_dtype is not None:
            dtype = model_config.torch_dtype
            
        config = CacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device
        )
        
        return cls(config)
    
    def reset(self):
        """Reset the cache to empty state (only moves pointer, no memory operation)."""
        self.current_len = 0
    
    def truncate(self, new_len: int):
        """
        Truncate cache to specified length.
        
        This is an O(1) operation - only moves the pointer, no data copying.
        Old data beyond new_len will be overwritten on next update.
        
        Args:
            new_len: New length to truncate to (must be <= current_len)
        """
        assert 0 <= new_len <= self.max_seq_len, \
            f"new_len {new_len} must be in [0, {self.max_seq_len}]"
        self.current_len = new_len
    
    def update(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Append new key-value pairs to the cache.
        
        Args:
            new_keys: New keys to append, shape [num_layers, batch, heads, seq_len, head_dim]
            new_values: New values to append, same shape as new_keys
        """
        seq_len = new_keys.shape[3]
        end_pos = self.current_len + seq_len
        
        assert end_pos <= self.max_seq_len, \
            f"Cache overflow: trying to add {seq_len} tokens but only {self.max_seq_len - self.current_len} slots available"
        
        # Direct assignment (in-place, no new memory allocation)
        self.keys[:, :, :, self.current_len:end_pos, :] = new_keys
        self.values[:, :, :, self.current_len:end_pos, :] = new_values
        
        self.current_len = end_pos
    
    def update_from_hf(self, past_key_values):
        """
        Update cache from HuggingFace past_key_values format.
        
        Supports both DynamicCache (new) and tuple format (legacy).
        Each key/value has shape [batch, heads, seq_len, head_dim]
        
        Args:
            past_key_values: HuggingFace format past_key_values (DynamicCache or tuple)
        """
        if past_key_values is None:
            return
        
        # Handle DynamicCache (newer transformers)
        if HAS_DYNAMIC_CACHE and isinstance(past_key_values, DynamicCache):
            seq_len = past_key_values.get_seq_length()
            
            if self.current_len == 0:
                # Full update
                for layer_idx in range(len(past_key_values)):
                    key, value = past_key_values[layer_idx]
                    self.keys[layer_idx, :, :, :seq_len, :] = key
                    self.values[layer_idx, :, :, :seq_len, :] = value
                self.current_len = seq_len
            else:
                # Incremental update
                new_tokens = seq_len - self.current_len
                if new_tokens > 0:
                    for layer_idx in range(len(past_key_values)):
                        key, value = past_key_values[layer_idx]
                        self.keys[layer_idx, :, :, self.current_len:seq_len, :] = key[:, :, self.current_len:, :]
                        self.values[layer_idx, :, :, self.current_len:seq_len, :] = value[:, :, self.current_len:, :]
                    self.current_len = seq_len
        else:
            # Handle tuple format (legacy)
            seq_len = past_key_values[0][0].shape[2]
            
            if self.current_len == 0:
                # Full update - copy everything
                for layer_idx, (key, value) in enumerate(past_key_values):
                    self.keys[layer_idx, :, :, :seq_len, :] = key
                    self.values[layer_idx, :, :, :seq_len, :] = value
                self.current_len = seq_len
            else:
                # Incremental update - only copy new tokens
                new_tokens = seq_len - self.current_len
                if new_tokens > 0:
                    for layer_idx, (key, value) in enumerate(past_key_values):
                        self.keys[layer_idx, :, :, self.current_len:seq_len, :] = key[:, :, self.current_len:, :]
                        self.values[layer_idx, :, :, self.current_len:seq_len, :] = value[:, :, self.current_len:, :]
                    self.current_len = seq_len
    
    def to_hf_format(self):
        """
        Convert cache to HuggingFace past_key_values format.
        
        IMPORTANT: Returns CLONES of the cache data, not views!
        This is necessary because HuggingFace's model forward pass can
        in-place modify the DynamicCache, which would corrupt our StaticKVCache.
        
        For newer transformers versions, returns DynamicCache.
        For older versions, returns tuple format.
        
        Returns:
            DynamicCache or Tuple of (key, value) tensors for each layer
        """
        if HAS_DYNAMIC_CACHE:
            # Use DynamicCache for newer transformers versions
            cache = DynamicCache()
            for layer_idx in range(self.num_layers):
                # CRITICAL: Use .clone() to prevent HuggingFace from modifying our data
                key = self.keys[layer_idx, :, :, :self.current_len, :].clone()
                value = self.values[layer_idx, :, :, :self.current_len, :].clone()
                cache.update(key, value, layer_idx)
            return cache
        else:
            # Fall back to tuple format for older versions
            # Also use .clone() here for safety
            return tuple(
                (
                    self.keys[layer_idx, :, :, :self.current_len, :].clone(),
                    self.values[layer_idx, :, :, :self.current_len, :].clone()
                )
                for layer_idx in range(self.num_layers)
            )
    
    def to_tuple_format(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Convert cache to tuple format (legacy).
        
        Returns:
            Tuple of (key, value) tensors for each layer
        """
        return tuple(
            (
                self.keys[layer_idx, :, :, :self.current_len, :],
                self.values[layer_idx, :, :, :self.current_len, :]
            )
            for layer_idx in range(self.num_layers)
        )
    
    def get_seq_len(self) -> int:
        """Get current sequence length in cache."""
        return self.current_len
    
    def get_memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self._memory_allocated / (1024 * 1024)
    
    def __repr__(self) -> str:
        return (
            f"StaticKVCache("
            f"layers={self.num_layers}, "
            f"heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"max_len={self.max_seq_len}, "
            f"current_len={self.current_len}, "
            f"memory={self.get_memory_usage_mb():.1f}MB)"
        )


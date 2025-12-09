"""
KV Cache Compression Methods Registry

This module provides a registry of all available KV cache compression methods
and a factory function to retrieve them by name.
"""

from typing import Callable, Dict, List

from .l2_compress import l2_compress
from .fix_size_l2 import fix_size_l2_compress
from .streaming_llm import streaming_llm_compress
from .recent_only import recent_only_compress
from .head_aware_compress import (
    head_aware_compress,
    HeadAwareCompressor,
    HeadCompressionConfig,
    LayerCompressionConfig,
)

# Registry of all compression methods
COMPRESS_METHODS: Dict[str, Callable] = {
    "l2_compress": l2_compress,
    "fix_size_l2": fix_size_l2_compress,
    "streaming_llm": streaming_llm_compress,
    "recent_only": recent_only_compress,
    "head_aware": head_aware_compress,
}


def get_compress_fn(method: str) -> Callable:
    """
    Get compression function by name.
    
    Args:
        method: Compression method name
                - "l2_compress": Original KnormPress ratio-based compression
                - "fix_size_l2": Fixed-size KV cache with L2-based eviction
                - "streaming_llm": StreamingLLM with attention sinks
                - "recent_only": Simple sliding window (baseline)
    
    Returns:
        Compression function
    
    Raises:
        ValueError: If method name is not found in registry
    """
    if method not in COMPRESS_METHODS:
        available = list(COMPRESS_METHODS.keys())
        raise ValueError(f"Unknown method: {method}. Available: {available}")
    
    return COMPRESS_METHODS[method]


def list_methods() -> List[str]:
    """Return list of available compression method names."""
    return list(COMPRESS_METHODS.keys())


def register_method(name: str, fn: Callable) -> None:
    """
    Register a new compression method.
    
    Args:
        name: Method name
        fn: Compression function with signature:
            fn(past_key_values, **kwargs) -> List[Tuple[torch.Tensor, torch.Tensor]]
    """
    COMPRESS_METHODS[name] = fn


__all__ = [
    # Functions
    'l2_compress',
    'fix_size_l2_compress',
    'streaming_llm_compress',
    'recent_only_compress',
    'head_aware_compress',
    'get_compress_fn',
    'list_methods',
    'register_method',
    # Classes
    'HeadAwareCompressor',
    'HeadCompressionConfig',
    'LayerCompressionConfig',
    # Registry
    'COMPRESS_METHODS',
]


"""
Speculative Decoding Core Module

This module implements speculative decoding for accelerating LLM inference.

Available Generators:
- SpeculativeGenerator: Basic linear speculative decoding
- SpeculativeGeneratorWithStaticCache: Linear with pre-allocated cache
- StreamingSpeculativeGenerator: Linear with StreamingLLM compression
- TreeSpeculativeGenerator: Tree-based speculative decoding (SpecInfer-style)
- TreeStreamingSpeculativeGenerator: Tree-based with StreamingLLM compression
"""

from .static_cache import StaticKVCache
from .speculative_generator import SpeculativeGenerator, SpeculativeGeneratorWithStaticCache
from .streaming_speculative_generator import (
    StreamingSpeculativeGenerator,
    StreamingSpeculativeGeneratorV2
)
from .token_tree import TokenTree, TreeNode, build_tree_from_topk
from .tree_speculative_generator import (
    TreeSpeculativeGenerator,
    TreeSpeculativeGeneratorV2,
    TreeStreamingSpeculativeGenerator
)

__all__ = [
    # Cache utilities
    "StaticKVCache",
    
    # Linear speculative decoding
    "SpeculativeGenerator",
    "SpeculativeGeneratorWithStaticCache",
    "StreamingSpeculativeGenerator",
    "StreamingSpeculativeGeneratorV2",
    
    # Tree-based speculative decoding
    "TokenTree",
    "TreeNode",
    "build_tree_from_topk",
    "TreeSpeculativeGenerator",
    "TreeSpeculativeGeneratorV2",
    "TreeStreamingSpeculativeGenerator",
]




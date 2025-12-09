"""
Head-Aware KV Cache Compression

This module implements head-level KV cache compression that applies different
compression strategies to different attention heads based on their behavior.

Key insight: Different attention heads have different roles:
- Sink-Positional heads: Need sink tokens + small window (StreamingLLM style)
- True-Positional heads: Only need sliding window (no sinks needed)
- Sink-Mixed heads: Need sink tokens + larger window
- Gathering heads: Need full KV cache (cannot be compressed)
- Dead heads: Can be pruned

This allows for more efficient compression than uniform strategies.

Reference:
    Analysis based on attention head specialization patterns discovered in
    Pythia models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import json
import os
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


@dataclass
class HeadCompressionConfig:
    """Configuration for compressing a single attention head."""
    layer_idx: int
    head_idx: int
    strategy: str = "full"  # "sink_window", "window_only", "full", "prune"
    keep_sinks: bool = True
    sink_size: int = 4
    window_size: int = 8
    
    def total_cache_size(self) -> int:
        """Return the total cache size for this head."""
        if self.strategy == "prune":
            return 0
        elif self.strategy == "full":
            return -1  # Unlimited
        elif self.strategy == "sink_window":
            return self.sink_size + self.window_size
        elif self.strategy == "window_only":
            return self.window_size
        return -1


@dataclass
class LayerCompressionConfig:
    """Configuration for compressing all heads in a layer."""
    layer_idx: int
    head_configs: Dict[int, HeadCompressionConfig] = field(default_factory=dict)
    
    # Quick lookup for uniform strategies
    has_mixed_strategies: bool = False
    uniform_strategy: Optional[str] = None
    
    def add_head(self, config: HeadCompressionConfig) -> None:
        """Add a head configuration."""
        self.head_configs[config.head_idx] = config
        
    def get_head(self, head_idx: int) -> Optional[HeadCompressionConfig]:
        """Get configuration for a specific head."""
        return self.head_configs.get(head_idx)


class HeadAwareCompressor:
    """
    Head-aware KV cache compressor.
    
    This compressor applies different compression strategies to different
    attention heads based on pre-computed head classifications.
    
    Usage:
        # Load configurations from classification results
        compressor = HeadAwareCompressor.from_classifications(
            "results/attention_analysis_pythia-2.8b/head_classifications.json"
        )
        
        # Compress KV cache
        compressed = compressor.compress(past_key_values)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        default_strategy: str = "full",
        default_sink_size: int = 4,
        default_window_size: int = 8,
    ):
        """
        Initialize the compressor.
        
        Args:
            num_layers: Number of layers in the model
            num_heads: Number of heads per layer
            default_strategy: Default compression strategy for unconfigured heads
            default_sink_size: Default number of sink tokens
            default_window_size: Default window size
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.default_strategy = default_strategy
        self.default_sink_size = default_sink_size
        self.default_window_size = default_window_size
        
        # Layer configurations
        self.layer_configs: Dict[int, LayerCompressionConfig] = {}
        
        # Pre-computed masks for efficient compression
        self._head_masks: Dict[int, torch.Tensor] = {}  # layer_idx -> mask
        self._initialized = False
        
    @classmethod
    def from_classifications(
        cls,
        classifications_path: str,
        strategy_mapping: Optional[Dict[str, str]] = None,
    ) -> "HeadAwareCompressor":
        """
        Create compressor from classification JSON file.
        
        Args:
            classifications_path: Path to head_classifications.json
            strategy_mapping: Optional mapping from head_type to compression strategy
        
        Returns:
            Configured HeadAwareCompressor instance
        """
        with open(classifications_path, 'r') as f:
            data = json.load(f)
        
        classifications = data.get('classifications', data)
        if isinstance(classifications, dict):
            classifications = classifications.get('classifications', [])
        
        # Infer model dimensions from classifications
        num_layers = max(c['layer_idx'] for c in classifications) + 1
        num_heads = max(c['head_idx'] for c in classifications) + 1
        
        # Default strategy mapping based on head type
        # NOTE: We avoid 'full' strategy as it prevents any memory savings when
        # mixed with other strategies in the same layer (due to padding requirements).
        # Instead, we use sink_window with larger windows for heads that need more context.
        if strategy_mapping is None:
            strategy_mapping = {
                'sink_positional': 'sink_window',   # sink(4) + window(8) = 12
                'true_positional': 'window_only',   # window(8)
                'sink_mixed': 'sink_window',        # sink(4) + window(24) = 28
                'gathering': 'sink_window',         # sink(4) + window(256) = 260 (large window)
                'mixed': 'sink_window',             # sink(4) + window(64) = 68
                'dead': 'prune',
                # Legacy types
                'positional': 'sink_window',        # sink(4) + window(8) = 12
            }
        
        compressor = cls(
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # Configure heads based on classifications
        # Window sizes by head type (tuned based on analysis)
        window_size_by_type = {
            'sink_positional': 8,      # Very focused, small window OK
            'true_positional': 8,      # Very focused, small window OK
            'sink_mixed': 24,          # Needs some context
            'mixed': 64,               # Needs more context
            'gathering': 256,          # Needs large context (content-dependent)
            'positional': 8,           # Legacy type
            'dead': 0,                 # Will be pruned
        }
        
        for c in classifications:
            head_type = c['head_type']
            strategy = strategy_mapping.get(head_type, 'sink_window')
            
            # Get compression parameters
            keep_sinks = c.get('keep_sinks', strategy == 'sink_window')
            sink_size = c.get('sink_size', 4)
            
            # Use classification's recommended_window if available and valid,
            # otherwise use type-based default
            recommended = c.get('recommended_window', -1)
            if recommended > 0:
                window_size = recommended
            else:
                window_size = window_size_by_type.get(head_type, 64)
            
            config = HeadCompressionConfig(
                layer_idx=c['layer_idx'],
                head_idx=c['head_idx'],
                strategy=strategy,
                keep_sinks=keep_sinks,
                sink_size=sink_size,
                window_size=window_size,
            )
            
            compressor.add_head_config(config)
        
        return compressor
    
    @classmethod
    def from_model_config(
        cls,
        model_config,
        default_strategy: str = "streaming_llm",
        sink_size: int = 4,
        window_size: int = 508,
    ) -> "HeadAwareCompressor":
        """
        Create compressor with uniform strategy for all heads.
        
        This is useful as a baseline or when no classification is available.
        
        Args:
            model_config: Model configuration object
            default_strategy: Strategy to use for all heads
            sink_size: Sink token count
            window_size: Recent window size
        
        Returns:
            HeadAwareCompressor with uniform configuration
        """
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_attention_heads
        
        compressor = cls(
            num_layers=num_layers,
            num_heads=num_heads,
            default_strategy=default_strategy,
            default_sink_size=sink_size,
            default_window_size=window_size,
        )
        
        # Configure all heads uniformly
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                config = HeadCompressionConfig(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    strategy="sink_window" if default_strategy == "streaming_llm" else default_strategy,
                    keep_sinks=True,
                    sink_size=sink_size,
                    window_size=window_size,
                )
                compressor.add_head_config(config)
        
        return compressor
    
    def add_head_config(self, config: HeadCompressionConfig) -> None:
        """Add configuration for a specific head."""
        layer_idx = config.layer_idx
        
        if layer_idx not in self.layer_configs:
            self.layer_configs[layer_idx] = LayerCompressionConfig(layer_idx=layer_idx)
        
        self.layer_configs[layer_idx].add_head(config)
        self._initialized = False  # Need to re-initialize masks
    
    def get_head_config(self, layer_idx: int, head_idx: int) -> HeadCompressionConfig:
        """Get configuration for a specific head (creates default if not exists)."""
        if layer_idx in self.layer_configs:
            config = self.layer_configs[layer_idx].get_head(head_idx)
            if config is not None:
                return config
        
        # Return default configuration
        return HeadCompressionConfig(
            layer_idx=layer_idx,
            head_idx=head_idx,
            strategy=self.default_strategy,
            keep_sinks=True,
            sink_size=self.default_sink_size,
            window_size=self.default_window_size,
        )
    
    def _analyze_layer_strategies(self, layer_idx: int) -> Tuple[bool, Optional[str]]:
        """
        Analyze if a layer can use optimized uniform compression.
        
        Returns:
            (has_mixed, uniform_strategy): 
            - If has_mixed=False, uniform_strategy is the single strategy for all heads
            - If has_mixed=True, uniform_strategy is None
        """
        if layer_idx not in self.layer_configs:
            return False, self.default_strategy
        
        layer_config = self.layer_configs[layer_idx]
        strategies = set()
        
        for head_idx in range(self.num_heads):
            config = layer_config.get_head(head_idx)
            if config is None:
                strategies.add(self.default_strategy)
            else:
                strategies.add(config.strategy)
        
        if len(strategies) == 1:
            return False, strategies.pop()
        return True, None
    
    def compress(
        self,
        past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
        current_seq_len: Optional[int] = None,
        skip_layers: Optional[List[int]] = None,
        **kwargs,  # Accept additional arguments for interface compatibility
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress KV cache using head-aware strategies.
        
        Args:
            past_key_values: KV cache, shape per layer: (batch, heads, seq, dim)
            current_seq_len: Current sequence length (for computing which tokens to keep)
            skip_layers: List of layer indices to skip compression (keep full cache)
            **kwargs: Additional arguments (ignored, for interface compatibility)
        
        Returns:
            Compressed KV cache as list of (key, value) tuples
        """
        past_key_values = list(normalize_kv_cache(past_key_values))
        
        if not past_key_values:
            return past_key_values
        
        # Get sequence length from cache if not provided
        if current_seq_len is None:
            current_seq_len = past_key_values[0][0].size(2)
        
        # Default skip_layers to empty list
        if skip_layers is None:
            skip_layers = []
        
        compressed = []
        
        for layer_idx, (keys, values) in enumerate(past_key_values):
            # keys/values shape: (batch, heads, seq, dim)
            batch_size, num_heads, seq_len, head_dim = keys.shape
            
            # Skip compression for specified layers
            if layer_idx in skip_layers:
                compressed.append((keys, values))
                continue
            
            # Check for uniform strategy (optimization)
            has_mixed, uniform_strategy = self._analyze_layer_strategies(layer_idx)
            
            if not has_mixed and uniform_strategy is not None:
                # All heads use the same strategy - use optimized path
                if uniform_strategy == "full":
                    # All heads need full cache, skip compression
                    compressed.append((keys, values))
                    continue
                
                # Check if compression is needed based on sequence length
                config = self.get_head_config(layer_idx, 0)
                total_cache = config.total_cache_size()
                if total_cache < 0 or seq_len <= total_cache:
                    compressed.append((keys, values))
                    continue
                
                # Use optimized uniform compression for this layer
                comp_keys, comp_values = self._compress_layer_uniform(
                    keys, values, layer_idx, uniform_strategy
                )
            else:
                # Mixed strategies in this layer - use per-head compression
                # This handles cases where some heads need full cache and others don't
                comp_keys, comp_values = self._compress_layer_per_head(
                    keys, values, layer_idx
                )
            
            compressed.append((comp_keys, comp_values))
        
        return compressed
    
    def _get_max_cache_size(self, layer_idx: int) -> int:
        """Get maximum cache size for a layer (returns -1 if any head needs full cache)."""
        max_size = 0
        
        for head_idx in range(self.num_heads):
            config = self.get_head_config(layer_idx, head_idx)
            head_size = config.total_cache_size()
            if head_size < 0:
                return -1  # At least one head needs full cache
            max_size = max(max_size, head_size)
        
        return max_size
    
    def _compress_layer_uniform(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        strategy: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a layer using uniform strategy for all heads."""
        seq_len = keys.size(2)
        config = self.get_head_config(layer_idx, 0)  # Get config from first head
        
        if strategy == "full":
            return keys, values
        
        elif strategy == "prune":
            # Zero out the cache (keep shape for compatibility)
            return torch.zeros_like(keys), torch.zeros_like(values)
        
        elif strategy == "sink_window":
            sink_size = config.sink_size
            window_size = config.window_size
            
            if seq_len <= sink_size + window_size:
                return keys, values
            
            # Keep sink tokens + recent window
            sink_keys = keys[:, :, :sink_size, :]
            sink_values = values[:, :, :sink_size, :]
            
            recent_keys = keys[:, :, -window_size:, :]
            recent_values = values[:, :, -window_size:, :]
            
            comp_keys = torch.cat([sink_keys, recent_keys], dim=2)
            comp_values = torch.cat([sink_values, recent_values], dim=2)
            
            return comp_keys, comp_values
        
        elif strategy == "window_only":
            window_size = config.window_size
            
            if seq_len <= window_size:
                return keys, values
            
            return keys[:, :, -window_size:, :], values[:, :, -window_size:, :]
        
        return keys, values
    
    def _compress_layer_per_head(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a layer with different strategies per head.
        
        This is more complex as we need to handle heads with different cache sizes.
        For simplicity, we pad smaller caches to match the largest.
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape
        device = keys.device
        dtype = keys.dtype
        
        # First pass: determine max cache size needed
        max_cache_size = 0
        head_cache_sizes = []
        
        for head_idx in range(num_heads):
            config = self.get_head_config(layer_idx, head_idx)
            head_size = config.total_cache_size()
            
            if head_size < 0:
                head_size = seq_len  # Full cache
            elif head_size > seq_len:
                head_size = seq_len  # Cap at actual sequence length
            
            head_cache_sizes.append(head_size)
            max_cache_size = max(max_cache_size, head_size)
        
        # Allocate output tensors
        comp_keys = torch.zeros(
            batch_size, num_heads, max_cache_size, head_dim,
            device=device, dtype=dtype
        )
        comp_values = torch.zeros(
            batch_size, num_heads, max_cache_size, head_dim,
            device=device, dtype=dtype
        )
        
        # Second pass: compress each head
        for head_idx in range(num_heads):
            config = self.get_head_config(layer_idx, head_idx)
            strategy = config.strategy
            
            head_keys = keys[:, head_idx:head_idx+1, :, :]  # Keep head dim
            head_values = values[:, head_idx:head_idx+1, :, :]
            
            if strategy == "full":
                # Copy full cache
                comp_keys[:, head_idx:head_idx+1, :seq_len, :] = head_keys
                comp_values[:, head_idx:head_idx+1, :seq_len, :] = head_values
            
            elif strategy == "prune":
                # Leave as zeros
                pass
            
            elif strategy == "sink_window":
                sink_size = config.sink_size
                window_size = config.window_size
                
                if seq_len <= sink_size + window_size:
                    comp_keys[:, head_idx:head_idx+1, :seq_len, :] = head_keys
                    comp_values[:, head_idx:head_idx+1, :seq_len, :] = head_values
                else:
                    # Sink tokens
                    comp_keys[:, head_idx:head_idx+1, :sink_size, :] = head_keys[:, :, :sink_size, :]
                    comp_values[:, head_idx:head_idx+1, :sink_size, :] = head_values[:, :, :sink_size, :]
                    # Recent window
                    comp_keys[:, head_idx:head_idx+1, sink_size:sink_size+window_size, :] = head_keys[:, :, -window_size:, :]
                    comp_values[:, head_idx:head_idx+1, sink_size:sink_size+window_size, :] = head_values[:, :, -window_size:, :]
            
            elif strategy == "window_only":
                window_size = config.window_size
                actual_size = min(seq_len, window_size)
                
                comp_keys[:, head_idx:head_idx+1, :actual_size, :] = head_keys[:, :, -actual_size:, :]
                comp_values[:, head_idx:head_idx+1, :actual_size, :] = head_values[:, :, -actual_size:, :]
        
        return comp_keys, comp_values
    
    def get_compression_summary(self) -> dict:
        """Get summary of compression configuration."""
        strategy_counts = {
            "sink_window": 0,
            "window_only": 0,
            "full": 0,
            "prune": 0,
        }
        
        total_heads = self.num_layers * self.num_heads
        compressible_heads = 0
        
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                config = self.get_head_config(layer_idx, head_idx)
                strategy_counts[config.strategy] = strategy_counts.get(config.strategy, 0) + 1
                if config.strategy != "full":
                    compressible_heads += 1
        
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "total_heads": total_heads,
            "strategy_distribution": strategy_counts,
            "compressible_heads": compressible_heads,
            "compressible_percentage": compressible_heads / total_heads * 100,
        }


def head_aware_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    classifications_path: Optional[str] = None,
    compressor: Optional[HeadAwareCompressor] = None,
    num_layers: int = 32,
    num_heads: int = 32,
    default_sink_size: int = 4,
    default_window_size: int = 8,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Head-aware KV cache compression function.
    
    This is the main entry point for head-aware compression, compatible with
    the existing compression method interface.
    
    Args:
        past_key_values: KV cache
        classifications_path: Path to head classifications JSON file
        compressor: Pre-configured HeadAwareCompressor (if provided, ignores other args)
        num_layers: Number of layers (used if compressor not provided)
        num_heads: Number of heads per layer (used if compressor not provided)
        default_sink_size: Default sink size (used if no classifications)
        default_window_size: Default window size (used if no classifications)
        skip_layers: Layer indices to skip compression
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Compressed KV cache
    """
    if compressor is None:
        if classifications_path is not None and os.path.exists(classifications_path):
            compressor = HeadAwareCompressor.from_classifications(classifications_path)
        else:
            # Create default compressor with StreamingLLM-style uniform strategy
            compressor = HeadAwareCompressor(
                num_layers=num_layers,
                num_heads=num_heads,
                default_strategy="sink_window",
                default_sink_size=default_sink_size,
                default_window_size=default_window_size,
            )
    
    # Compress with skip_layers support
    return compressor.compress(past_key_values, skip_layers=skip_layers)


__all__ = [
    'HeadCompressionConfig',
    'LayerCompressionConfig',
    'HeadAwareCompressor',
    'head_aware_compress',
]


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


class HeadAwareMaskGenerator:
    """
    Generate per-head 4D attention masks based on head classifications.
    
    This class creates attention masks that allow different heads to attend to
    different portions of the sequence, effectively simulating heterogeneous
    attention windows without modifying the KV cache.
    
    Key insight: While this doesn't reduce memory usage (full KV cache is kept),
    it allows us to validate whether head-aware windowing improves PPL/accuracy
    compared to uniform windowing strategies like StreamingLLM.
    
    Usage:
        mask_gen = HeadAwareMaskGenerator.from_classifications(
            "results/attention_analysis_pythia-2.8b/head_classifications.json"
        )
        
        # Generate mask for a specific layer and sequence length
        mask = mask_gen.generate_layer_mask(layer_idx=0, seq_len=1024, device='cuda')
        # Shape: (1, num_heads, seq_len, seq_len)
    """
    
    # Default window sizes by head type
    DEFAULT_WINDOW_SIZES = {
        'positional': 8,        # Positional heads: sink(4) + window(8) = 12
        'sink_positional': 8,
        'true_positional': 8,
        'mixed': 64,            # Mixed heads: sink(4) + window(64) = 68
        'sink_mixed': 24,
        'gathering': -1,        # Gathering heads: full context (no masking)
        'dead': 0,              # Dead heads: mask everything (effectively prune)
    }
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        default_sink_size: int = 4,
        default_window_size: int = 64,
    ):
        """
        Initialize the mask generator.
        
        Args:
            num_layers: Number of layers in the model
            num_heads: Number of heads per layer
            default_sink_size: Default number of sink tokens to preserve
            default_window_size: Default recent window size
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.default_sink_size = default_sink_size
        self.default_window_size = default_window_size
        
        # Per-head configuration: {(layer_idx, head_idx): {'sink_size': int, 'window_size': int, 'head_type': str}}
        self.head_configs: Dict[Tuple[int, int], Dict] = {}
        
        # Cache for generated masks
        self._mask_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        
    @classmethod
    def from_classifications(
        cls,
        classifications_path: str,
        window_size_override: Optional[Dict[str, int]] = None,
        sink_size: int = 4,
    ) -> "HeadAwareMaskGenerator":
        """
        Create mask generator from classification JSON file.
        
        Args:
            classifications_path: Path to head_classifications.json
            window_size_override: Optional dict to override default window sizes by head type
            sink_size: Number of sink tokens (default: 4)
        
        Returns:
            Configured HeadAwareMaskGenerator instance
        """
        with open(classifications_path, 'r') as f:
            data = json.load(f)
        
        classifications = data.get('classifications', data)
        if isinstance(classifications, dict):
            classifications = classifications.get('classifications', [])
        
        # Infer model dimensions
        num_layers = max(c['layer_idx'] for c in classifications) + 1
        num_heads = max(c['head_idx'] for c in classifications) + 1
        
        # Merge default window sizes with overrides
        window_sizes = cls.DEFAULT_WINDOW_SIZES.copy()
        if window_size_override:
            window_sizes.update(window_size_override)
        
        generator = cls(
            num_layers=num_layers,
            num_heads=num_heads,
            default_sink_size=sink_size,
        )
        
        # Configure each head based on classification
        for c in classifications:
            layer_idx = c['layer_idx']
            head_idx = c['head_idx']
            head_type = c['head_type']
            
            # Get window size for this head type
            # Use recommended_window from classification if available and positive
            recommended = c.get('recommended_window', -1)
            if recommended > 0:
                window_size = recommended
            else:
                window_size = window_sizes.get(head_type, 64)
            
            generator.head_configs[(layer_idx, head_idx)] = {
                'head_type': head_type,
                'sink_size': sink_size,
                'window_size': window_size,
            }
        
        return generator
    
    @classmethod
    def create_uniform(
        cls,
        num_layers: int,
        num_heads: int,
        sink_size: int = 4,
        window_size: int = 508,
    ) -> "HeadAwareMaskGenerator":
        """
        Create a mask generator with uniform settings for all heads.
        
        This is useful as a baseline (equivalent to StreamingLLM masking).
        
        Args:
            num_layers: Number of layers
            num_heads: Number of heads per layer
            sink_size: Number of sink tokens
            window_size: Recent window size
        
        Returns:
            HeadAwareMaskGenerator with uniform configuration
        """
        generator = cls(
            num_layers=num_layers,
            num_heads=num_heads,
            default_sink_size=sink_size,
            default_window_size=window_size,
        )
        
        # Configure all heads with the same settings
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                generator.head_configs[(layer_idx, head_idx)] = {
                    'head_type': 'uniform',
                    'sink_size': sink_size,
                    'window_size': window_size,
                }
        
        return generator
    
    def get_head_config(self, layer_idx: int, head_idx: int) -> Dict:
        """Get configuration for a specific head."""
        key = (layer_idx, head_idx)
        if key in self.head_configs:
            return self.head_configs[key]
        
        # Return default configuration
        return {
            'head_type': 'default',
            'sink_size': self.default_sink_size,
            'window_size': self.default_window_size,
        }
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create a standard causal mask (lower triangular)."""
        # Create causal mask: 0 for allowed positions, -inf for masked positions
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        # Upper triangular (excluding diagonal) should be -inf
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
        return mask
    
    def _create_sink_window_mask(
        self,
        seq_len: int,
        sink_size: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create a sink + window mask for a single head.
        
        Each query position q can attend to:
        - Positions [0, sink_size): Always visible (attention sinks)
        - Positions [q - window_size + 1, q]: Recent window (causal)
        
        Args:
            seq_len: Sequence length
            sink_size: Number of sink tokens at the beginning
            window_size: Size of recent window
            device: Device to create mask on
            dtype: Data type for mask
        
        Returns:
            Mask tensor of shape (seq_len, seq_len)
        """
        # Start with all masked (-inf)
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        
        for q_pos in range(seq_len):
            # Sink tokens: always visible (positions 0 to sink_size-1)
            if sink_size > 0:
                mask[q_pos, :min(sink_size, q_pos + 1)] = 0.0
            
            # Recent window: positions within window_size of current position
            # Must also be causal (k_pos <= q_pos)
            window_start = max(0, q_pos - window_size + 1)
            window_end = q_pos + 1  # Inclusive of current position (causal)
            mask[q_pos, window_start:window_end] = 0.0
        
        return mask
    
    def generate_layer_mask(
        self,
        layer_idx: int,
        seq_len: int,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate a 4D attention mask for a specific layer.
        
        Args:
            layer_idx: Layer index
            seq_len: Current sequence length
            device: Device to create mask on
            dtype: Data type for mask
            use_cache: Whether to cache generated masks
        
        Returns:
            Attention mask of shape (1, num_heads, seq_len, seq_len)
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check cache
        cache_key = (layer_idx, seq_len, str(device))
        if use_cache and cache_key in self._mask_cache:
            cached = self._mask_cache[cache_key]
            if cached.device == device:
                return cached.to(dtype)
        
        # Create per-head masks
        masks = []
        
        for head_idx in range(self.num_heads):
            config = self.get_head_config(layer_idx, head_idx)
            head_type = config['head_type']
            sink_size = config['sink_size']
            window_size = config['window_size']
            
            if head_type == 'dead':
                # Dead heads: mask everything except current token (to avoid NaN)
                mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
                for i in range(seq_len):
                    mask[i, i] = 0.0  # Allow self-attention to prevent NaN
            elif window_size < 0 or head_type == 'gathering':
                # Full context: standard causal mask
                mask = self._create_causal_mask(seq_len, device, dtype)
            else:
                # Sink + window mask
                mask = self._create_sink_window_mask(
                    seq_len, sink_size, window_size, device, dtype
                )
            
            masks.append(mask)
        
        # Stack to create (num_heads, seq_len, seq_len)
        layer_mask = torch.stack(masks, dim=0)
        
        # Add batch dimension: (1, num_heads, seq_len, seq_len)
        layer_mask = layer_mask.unsqueeze(0)
        
        # Cache the result
        if use_cache:
            self._mask_cache[cache_key] = layer_mask
        
        return layer_mask
    
    def generate_all_layer_masks(
        self,
        seq_len: int,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        """
        Generate attention masks for all layers.
        
        Args:
            seq_len: Sequence length
            device: Device to create masks on
            dtype: Data type for masks
        
        Returns:
            List of masks, one per layer, each of shape (1, num_heads, seq_len, seq_len)
        """
        return [
            self.generate_layer_mask(layer_idx, seq_len, device, dtype)
            for layer_idx in range(self.num_layers)
        ]
    
    def clear_cache(self):
        """Clear the mask cache."""
        self._mask_cache.clear()
    
    def get_summary(self) -> Dict:
        """Get a summary of the mask generator configuration."""
        type_counts: Dict[str, int] = {}
        total_effective_context = 0
        
        for (layer_idx, head_idx), config in self.head_configs.items():
            head_type = config['head_type']
            type_counts[head_type] = type_counts.get(head_type, 0) + 1
            
            window_size = config['window_size']
            sink_size = config['sink_size']
            
            if window_size < 0:
                # Full context - we'll estimate based on typical sequence length
                effective = 1024  # placeholder
            else:
                effective = sink_size + window_size
            total_effective_context += effective
        
        num_configured = len(self.head_configs)
        avg_effective = total_effective_context / num_configured if num_configured > 0 else 0
        
        return {
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'total_heads': self.num_layers * self.num_heads,
            'configured_heads': num_configured,
            'head_type_distribution': type_counts,
            'average_effective_context': avg_effective,
            'default_sink_size': self.default_sink_size,
            'default_window_size': self.default_window_size,
        }


def generate_head_aware_mask(
    layer_idx: int,
    seq_len: int,
    num_heads: int,
    classifications_path: Optional[str] = None,
    mask_generator: Optional[HeadAwareMaskGenerator] = None,
    device: Union[torch.device, str] = 'cpu',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate a head-aware attention mask for a single layer.
    
    This is a convenience function for generating per-head attention masks.
    
    Args:
        layer_idx: Layer index
        seq_len: Sequence length
        num_heads: Number of attention heads
        classifications_path: Path to head classifications JSON
        mask_generator: Pre-configured HeadAwareMaskGenerator (if provided, ignores classifications_path)
        device: Device to create mask on
        dtype: Data type for mask
    
    Returns:
        Attention mask of shape (1, num_heads, seq_len, seq_len)
    """
    if mask_generator is None:
        if classifications_path is not None and os.path.exists(classifications_path):
            mask_generator = HeadAwareMaskGenerator.from_classifications(classifications_path)
        else:
            raise ValueError("Either classifications_path or mask_generator must be provided")
    
    return mask_generator.generate_layer_mask(layer_idx, seq_len, device, dtype)


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
    'HeadAwareMaskGenerator',
    'head_aware_compress',
    'generate_head_aware_mask',
]


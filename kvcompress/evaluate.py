"""
Unified Evaluation Module for KV Cache Compression

This module provides evaluation functions that work with any compression method.
It measures PPL (perplexity), accuracy, and timing metrics with KV cache compression.

Key implementation points:
1. Process one token at a time (autoregressive)
2. Use CrossEntropyLoss(reduction="none") for per-token NLL
3. Apply compression after EVERY forward pass (if enabled)
4. PPL = exp(mean(nlls))
5. Accuracy = num_correct / num_tokens
6. TTFT/TPOT measured across ALL tokens (not just initial generation)

Extended features:
- evaluate_with_head_aware_mask: Uses per-layer 4D attention masks to simulate 
  per-head different attention windows WITHOUT modifying KV cache (for validation)
- Per-layer mask injection via forward hooks for true head-aware masking
"""

import time
from typing import Callable, Dict, List, Optional, Union
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import DynamicCache

from .utils import to_dynamic_cache, normalize_kv_cache
from .methods.head_aware_compress import HeadAwareMaskGenerator


class PerLayerMaskInjector:
    """
    Injects per-layer attention masks into GPT-NeoX model using forward hooks.
    
    This allows different layers to use different attention masks based on
    their specific head type configurations.
    """
    
    def __init__(self, model, mask_generator: HeadAwareMaskGenerator, device, dtype):
        self.model = model
        self.mask_generator = mask_generator
        self.device = device
        self.dtype = dtype
        self.hooks = []
        self.current_seq_len = 0
        self.use_incremental = False  # Whether we're in incremental decoding mode
        self._layer_masks = {}  # Cache for current seq_len
        self.verbose = False  # Set to True to enable logging
        
    def _get_layer_mask(self, layer_idx: int) -> torch.Tensor:
        """Get or create mask for a specific layer."""
        if layer_idx not in self._layer_masks:
            mask = self.mask_generator.generate_layer_mask(
                layer_idx=layer_idx,
                seq_len=self.current_seq_len,
                device=self.device,
                dtype=self.dtype,
                use_cache=False,
            )
            self._layer_masks[layer_idx] = mask
        return self._layer_masks[layer_idx]
    
    def _create_hook(self, layer_idx: int):
        """Create a forward pre-hook for a specific layer."""
        def hook(module, args, kwargs):
            # Get the layer-specific mask
            mask = self._get_layer_mask(layer_idx)
            
            # For incremental decoding, extract the last row
            if self.use_incremental and mask.shape[2] > 1:
                mask = mask[:, :, -1:, :]
            
            # Inject the mask into kwargs
            kwargs['attention_mask'] = mask
            
            return args, kwargs
        
        return hook
    
    def update_seq_len(self, seq_len: int, incremental: bool = False):
        """Update sequence length and clear mask cache."""
        self.current_seq_len = seq_len
        self.use_incremental = incremental
        self._layer_masks.clear()  # Clear cache when seq_len changes
    
    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        self.remove_hooks()  # Remove any existing hooks
        
        for layer_idx, layer in enumerate(self.model.gpt_neox.layers):
            hook = layer.attention.register_forward_pre_hook(
                self._create_hook(layer_idx),
                with_kwargs=True
            )
            self.hooks.append(hook)
        
        if self.verbose:
            print(f"[PerLayerMaskInjector] Registered {len(self.hooks)} hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def log_mask_summary(self):
        """Log a summary of masks for each layer (for debugging)."""
        if not self.verbose:
            return
            
        print(f"\n[PerLayerMaskInjector] Mask summary for seq_len={self.current_seq_len}:")
        for layer_idx in range(self.mask_generator.num_layers):
            config_summary = {}
            for head_idx in range(self.mask_generator.num_heads):
                config = self.mask_generator.get_head_config(layer_idx, head_idx)
                t = config['head_type']
                config_summary[t] = config_summary.get(t, 0) + 1
            print(f"  Layer {layer_idx}: {config_summary}")


def evaluate_with_compression(
    model,
    tokenizer,
    text: str,
    compress_fn: Optional[Callable] = None,
    compress_kwargs: Optional[Dict] = None,
    max_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate PPL and Accuracy with KV cache compression.
    
    This function supports any compression method through the compress_fn parameter.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        compress_fn: Compression function with signature:
                    fn(past_key_values, skip_layers=..., **kwargs) -> compressed_kv
                    If None, no compression is applied (baseline evaluation)
        compress_kwargs: Additional kwargs to pass to compress_fn
                        (e.g., keep_ratio, fix_kv_size, start_size, etc.)
        max_tokens: Maximum number of tokens to evaluate
        skip_layers: Layer indices to skip compression
        device: Device to use (auto-detected if None)
        show_progress: Whether to show progress bar
    
    Returns:
        Dict containing:
        - perplexity: The perplexity score
        - accuracy: Next token prediction accuracy
        - num_tokens: Number of tokens evaluated
        - final_cache_size: Final KV cache size after compression
    
    Example:
        >>> from kvcompress.methods import l2_compress
        >>> results = evaluate_with_compression(
        ...     model, tokenizer, text,
        ...     compress_fn=l2_compress,
        ...     compress_kwargs={"keep_ratio": 0.8, "prune_after": 1000}
        ... )
        
        >>> from kvcompress.methods import streaming_llm_compress
        >>> results = evaluate_with_compression(
        ...     model, tokenizer, text,
        ...     compress_fn=streaming_llm_compress,
        ...     compress_kwargs={"start_size": 4, "recent_size": 508}
        ... )
    """
    if device is None:
        device = next(model.parameters()).device
    
    if compress_kwargs is None:
        compress_kwargs = {}
    
    # Reset memory stats for VRAM measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "num_tokens": 0,
            "final_cache_size": 0,
            "ttft": 0.0,
            "tpot": 0.0,
            "throughput": 0.0,
            "total_time": 0.0,
            "peak_vram_gb": 0.0,
        }
    
    # Loss function for per-token NLL
    loss_fn = CrossEntropyLoss(reduction="none")
    
    # Initialize
    past_key_values = None
    nlls = []
    num_correct = []
    
    # Timing metrics
    ttft = None
    token_times = []
    
    model.eval()
    
    # Create progress bar if requested
    token_range = range(seq_len - 1)
    if show_progress:
        token_range = tqdm(token_range, desc="Evaluating")
    
    total_start = time.perf_counter()
    
    with torch.inference_mode():
        for idx in token_range:
            token_start = time.perf_counter()
            
            # Current token (single token input)
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
            
            # Calculate NLL for this token
            nll = loss_fn(logits, target)
            nlls.append(nll.item())
            
            # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)
            num_correct.append((predicted == target).int().item())
            
            # Get KV cache from outputs
            past_key_values = outputs.past_key_values
            
            # Apply compression if compress_fn is provided
            if compress_fn is not None and past_key_values is not None:
                # Convert to list format for compression
                kv_list = list(normalize_kv_cache(past_key_values))
                
                # Apply compression
                compressed_kv = compress_fn(
                    kv_list,
                    skip_layers=skip_layers,
                    **compress_kwargs
                )
                
                # Convert back to DynamicCache for next iteration
                past_key_values = to_dynamic_cache(compressed_kv)
            
            # Record timing
            token_time = time.perf_counter() - token_start
            token_times.append(token_time)
            
            # Record TTFT (first token time)
            if ttft is None:
                ttft = token_time
            
            # Update progress bar
            if show_progress:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                current_acc = sum(num_correct) / len(num_correct)
                token_range.set_description(
                    f"PPL: {current_ppl:.2f}, Acc: {current_acc:.2%}"
                )
    
    total_time = time.perf_counter() - total_start
    
    # Calculate final metrics
    perplexity = torch.exp(torch.tensor(nlls).mean()).item()
    accuracy = sum(num_correct) / len(num_correct)
    
    # Calculate timing metrics
    num_tokens = len(nlls)
    if num_tokens > 1:
        # TPOT: average time per token (excluding first token)
        tpot = sum(token_times[1:]) / (num_tokens - 1) if num_tokens > 1 else token_times[0]
    else:
        tpot = ttft if ttft else 0.0
    
    throughput = num_tokens / total_time if total_time > 0 else 0.0
    
    # Get final cache size (actual number of tokens stored)
    # Note: skip_layers may not be compressed, so check a compressed layer
    final_cache_size = 0
    if past_key_values is not None:
        kv_list = list(normalize_kv_cache(past_key_values))
        if kv_list:
            # Find a layer that was actually compressed (not in skip_layers)
            # to get the true compressed cache size
            for layer_idx, (k, v) in enumerate(kv_list):
                if layer_idx not in skip_layers:
                    final_cache_size = k.size(2)
                    break
            # If all layers were skipped, fall back to first layer
            if final_cache_size == 0:
                final_cache_size = kv_list[0][0].size(2)
    
    # Measure peak VRAM usage
    peak_memory_gb = 0.0
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": num_tokens,
        "final_cache_size": final_cache_size,
        # Timing metrics (measured across ALL tokens)
        "ttft": ttft if ttft else 0.0,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "peak_vram_gb": peak_memory_gb,
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
        compress_fn=None,  # No compression
        max_tokens=max_tokens,
        device=device,
        show_progress=show_progress,
    )


def compare_methods(
    model,
    tokenizer,
    text: str,
    methods_config: List[Dict],
    max_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Compare PPL and Accuracy across different compression methods.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        methods_config: List of dicts, each containing:
                       - "name": Method name for display
                       - "compress_fn": Compression function
                       - "kwargs": Dict of compression parameters
        max_tokens: Maximum number of tokens to evaluate
        skip_layers: Layer indices to skip compression
        device: Device to use
    
    Returns:
        List of result dicts, one per method
    
    Example:
        >>> from kvcompress.methods import l2_compress, streaming_llm_compress
        >>> methods = [
        ...     {"name": "baseline", "compress_fn": None, "kwargs": {}},
        ...     {"name": "l2_0.8", "compress_fn": l2_compress, "kwargs": {"keep_ratio": 0.8}},
        ...     {"name": "streaming", "compress_fn": streaming_llm_compress, "kwargs": {"start_size": 4, "recent_size": 508}},
        ... ]
        >>> results = compare_methods(model, tokenizer, text, methods)
    """
    results = []
    
    for config in methods_config:
        name = config.get("name", "unknown")
        compress_fn = config.get("compress_fn", None)
        kwargs = config.get("kwargs", {})
        
        print(f"\nEvaluating {name}...")
        
        result = evaluate_with_compression(
            model=model,
            tokenizer=tokenizer,
            text=text,
            compress_fn=compress_fn,
            compress_kwargs=kwargs,
            max_tokens=max_tokens,
            skip_layers=skip_layers,
            device=device,
            show_progress=True,
        )
        
        result['method'] = name
        result['config'] = kwargs
        results.append(result)
        
        print(f"  PPL: {result['perplexity']:.2f}")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Final cache size: {result['final_cache_size']}")
    
    return results


def evaluate_with_head_aware_mask(
    model,
    tokenizer,
    text: str,
    mask_generator: Optional[HeadAwareMaskGenerator] = None,
    classifications_path: Optional[str] = None,
    max_tokens: int = 3000,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate PPL and Accuracy using head-aware 4D attention masks.
    
    This function keeps the FULL KV cache but uses per-head attention masks
    to simulate different attention windows for different heads. This is useful
    for validating whether head-aware strategies improve PPL/accuracy before
    implementing actual KV cache compression.
    
    Key insight: By using 4D attention masks (batch, heads, query, key), we can
    force different heads to attend to different portions of the context:
    - Positional heads: Only see sink tokens + small recent window
    - Mixed heads: See sink tokens + medium window
    - Gathering heads: See full context (standard causal mask)
    
    This does NOT save memory (full KV cache is kept), but allows us to
    validate the head-aware strategy concept quickly.
    
    Args:
        model: The language model (must be GPT-NeoX/Pythia architecture)
        tokenizer: The tokenizer
        text: Input text to evaluate
        mask_generator: Pre-configured HeadAwareMaskGenerator
        classifications_path: Path to head_classifications.json (used if mask_generator is None)
        max_tokens: Maximum number of tokens to evaluate
        device: Device to use (auto-detected if None)
        show_progress: Whether to show progress bar
    
    Returns:
        Dict containing:
        - perplexity: The perplexity score
        - accuracy: Next token prediction accuracy
        - num_tokens: Number of tokens evaluated
        - final_cache_size: Final KV cache size (full, not compressed)
        - effective_context: Average effective context per head
        - timing metrics: ttft, tpot, throughput, total_time
    
    Example:
        >>> from kvcompress.methods import HeadAwareMaskGenerator
        >>> mask_gen = HeadAwareMaskGenerator.from_classifications(
        ...     "results/attention_analysis_pythia-2.8b/head_classifications.json"
        ... )
        >>> results = evaluate_with_head_aware_mask(
        ...     model, tokenizer, text,
        ...     mask_generator=mask_gen,
        ...     max_tokens=2000
        ... )
    """
    import os
    
    if device is None:
        device = next(model.parameters()).device
    
    # Create mask generator if not provided
    if mask_generator is None:
        if classifications_path is not None and os.path.exists(classifications_path):
            mask_generator = HeadAwareMaskGenerator.from_classifications(classifications_path)
        else:
            raise ValueError(
                "Either mask_generator or classifications_path must be provided"
            )
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "num_tokens": 0,
            "final_cache_size": 0,
            "effective_context": 0.0,
            "ttft": 0.0,
            "tpot": 0.0,
            "throughput": 0.0,
            "total_time": 0.0,
            "peak_vram_gb": 0.0,
        }
    
    # Loss function for per-token NLL
    loss_fn = CrossEntropyLoss(reduction="none")
    
    # Initialize
    past_key_values = None
    nlls = []
    num_correct = []
    
    # Timing metrics
    ttft = None
    token_times = []
    
    model.eval()
    
    # Create progress bar if requested
    token_range = range(seq_len - 1)
    if show_progress:
        token_range = tqdm(token_range, desc="Evaluating (head-aware mask)")
    
    total_start = time.perf_counter()
    
    # Get model config
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Determine mask dtype - use same as model for efficiency
    mask_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    if hasattr(model, 'dtype') and model.dtype is not None:
        mask_dtype = model.dtype
    
    # Clear any existing mask cache to avoid memory buildup
    mask_generator.clear_cache()
    
    # Create per-layer mask injector
    # This uses forward hooks to inject different masks into each layer
    mask_injector = PerLayerMaskInjector(model, mask_generator, device, mask_dtype)
    mask_injector.verbose = show_progress  # Enable logging if showing progress
    mask_injector.register_hooks()
    
    # Log mask configuration for first few layers (once at start)
    if show_progress:
        print("\n[Per-Layer Mask Config] Head type distribution per layer:")
        for layer_idx in range(min(5, num_layers)):  # Show first 5 layers
            type_counts = {}
            for head_idx in range(num_heads):
                config = mask_generator.get_head_config(layer_idx, head_idx)
                t = config['head_type']
                w = config['window_size']
                key = f"{t}(w={w})"
                type_counts[key] = type_counts.get(key, 0) + 1
            types_str = ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
            print(f"  Layer {layer_idx}: {types_str}")
        if num_layers > 5:
            print(f"  ... (showing first 5 of {num_layers} layers)")
        print()
    
    try:
        with torch.inference_mode():
            for idx in token_range:
                token_start = time.perf_counter()
                
                # Current token (single token input)
                current_token = input_ids[:, idx:idx+1]
                
                # Target token (next token)
                target = input_ids[:, idx+1:idx+2].view(-1)
                
                # Current sequence length (including past)
                current_seq_len = idx + 1
                
                # Update mask injector with current sequence length
                # The hooks will inject layer-specific masks during forward pass
                mask_injector.update_seq_len(current_seq_len, incremental=(past_key_values is not None))
                
                # Forward pass - masks are injected via hooks, so we don't pass attention_mask
                outputs = model(
                    current_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                # Get logits for next token prediction
                logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
                
                # Calculate NLL for this token
                nll = loss_fn(logits, target)
                nlls.append(nll.item())
                
                # Calculate accuracy
                predicted = torch.argmax(logits, dim=-1)
                num_correct.append((predicted == target).int().item())
                
                # Get KV cache from outputs (keep full, no compression)
                past_key_values = outputs.past_key_values
                
                # Record timing
                token_time = time.perf_counter() - token_start
                token_times.append(token_time)
                
                # Record TTFT (first token time)
                if ttft is None:
                    ttft = token_time
                
                # Update progress bar
                if show_progress:
                    current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                    current_acc = sum(num_correct) / len(num_correct)
                    token_range.set_description(
                        f"PPL: {current_ppl:.2f}, Acc: {current_acc:.2%}"
                    )
    finally:
        # Always remove hooks to avoid memory leaks
        mask_injector.remove_hooks()
    
    total_time = time.perf_counter() - total_start
    
    # Clear mask cache to free memory
    mask_generator.clear_cache()
    
    # Calculate final metrics
    perplexity = torch.exp(torch.tensor(nlls).mean()).item()
    accuracy = sum(num_correct) / len(num_correct)
    
    # Calculate timing metrics
    num_tokens = len(nlls)
    if num_tokens > 1:
        tpot = sum(token_times[1:]) / (num_tokens - 1)
    else:
        tpot = ttft if ttft else 0.0
    
    throughput = num_tokens / total_time if total_time > 0 else 0.0
    
    # Get final cache size
    final_cache_size = 0
    if past_key_values is not None:
        kv_list = list(normalize_kv_cache(past_key_values))
        if kv_list:
            final_cache_size = kv_list[0][0].size(2)
    
    # Get effective context from mask generator
    summary = mask_generator.get_summary()
    effective_context = summary.get('average_effective_context', 0.0)
    
    # Measure peak VRAM usage
    peak_memory_gb = 0.0
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": num_tokens,
        "final_cache_size": final_cache_size,
        "effective_context": effective_context,
        # Timing metrics
        "ttft": ttft if ttft else 0.0,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "peak_vram_gb": peak_memory_gb,
    }


__all__ = [
    'evaluate_with_compression',
    'evaluate_with_head_aware_mask',
    'evaluate_baseline',
    'compare_methods',
]


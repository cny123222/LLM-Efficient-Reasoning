"""
Attention Head Analyzer for Head Specialization Detection

This module analyzes attention patterns to identify:
1. Positional Heads (定位头): Low entropy, focus on fixed relative positions
2. Gathering Heads (汇聚头): High entropy, content-dependent attention
3. Dead Heads (死头): Near-uniform distribution, contribute little

Key metrics computed:
- Attention entropy: Measures how spread out attention is
- Position preference: Which relative positions each head focuses on
- Sink ratio: Attention allocated to initial tokens (attention sinks)
- Uniformity score: KL divergence from uniform distribution
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import torch
import numpy as np
from tqdm import tqdm
from transformers import DynamicCache


class HeadType(Enum):
    """Classification of attention head types."""
    # Original types (kept for backward compatibility)
    POSITIONAL = "positional"        # Low entropy, fixed position focus (legacy)
    GATHERING = "gathering"          # High entropy, content-dependent
    DEAD = "dead"                    # Near-uniform, low contribution
    MIXED = "mixed"                  # Does not fit clearly into categories
    
    # Refined types for better optimization
    SINK_POSITIONAL = "sink_positional"  # Low entropy, high sink ratio - needs sink + window
    TRUE_POSITIONAL = "true_positional"  # Low entropy, low sink, high local - window only
    SINK_MIXED = "sink_mixed"            # Medium entropy, high sink - needs sink + larger window


@dataclass
class HeadStatistics:
    """Statistics for a single attention head."""
    layer_idx: int
    head_idx: int
    
    # Core metrics
    mean_entropy: float = 0.0
    std_entropy: float = 0.0
    max_attention_mean: float = 0.0  # Mean of max attention values
    
    # Position preference metrics
    position_preference: Dict[str, float] = field(default_factory=dict)
    # Keys: "sink" (first 4), "recent" (last 10%), "local" (within 8), "global" (others)
    
    # Sink analysis
    sink_ratio: float = 0.0  # Attention on first 4 tokens
    
    # Uniformity analysis
    uniformity_score: float = 0.0  # Lower = more uniform (dead head indicator)
    
    # Relative position distribution (for visualization)
    relative_position_dist: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HeadClassification:
    """Classification result for a head."""
    layer_idx: int
    head_idx: int
    head_type: HeadType
    confidence: float  # 0-1, how confident we are in this classification
    
    # Recommendations
    can_prune: bool = False           # Can this head be completely removed?
    can_limit_window: bool = False    # Can we limit this head's KV cache?
    recommended_window: int = -1      # Recommended window size (-1 = no limit)
    
    # Enhanced compression recommendations
    keep_sinks: bool = False          # Whether to keep sink tokens (initial tokens)
    sink_size: int = 4                # Number of sink tokens to keep
    use_full_cache: bool = True       # Whether to use full KV cache
    compression_strategy: str = "none"  # "none", "window_only", "sink_window", "full"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "head_type": self.head_type.value,
            "confidence": self.confidence,
            "can_prune": self.can_prune,
            "can_limit_window": self.can_limit_window,
            "recommended_window": self.recommended_window,
            "keep_sinks": self.keep_sinks,
            "sink_size": self.sink_size,
            "use_full_cache": self.use_full_cache,
            "compression_strategy": self.compression_strategy,
        }


class AttentionAnalyzer:
    """
    Analyzer for attention head behavior and specialization.
    
    Usage:
        analyzer = AttentionAnalyzer(model, tokenizer)
        stats, classifications = analyzer.analyze(text, max_tokens=2048)
        analyzer.save_results(stats, classifications, output_dir)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        sink_size: int = 4,
        local_window: int = 8,
        recent_ratio: float = 0.1,
    ):
        """
        Initialize the analyzer.
        
        Args:
            model: The language model (must support output_attentions=True)
            tokenizer: The tokenizer
            device: Device to use (auto-detected if None)
            sink_size: Number of initial tokens considered as "sinks"
            local_window: Window size for "local" attention
            recent_ratio: Ratio of tokens considered "recent"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        self.sink_size = sink_size
        self.local_window = local_window
        self.recent_ratio = recent_ratio
        
        # Get model config
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        
        # Storage for accumulated statistics
        self._accumulated_stats: Dict[Tuple[int, int], Dict] = {}
    
    def _compute_entropy(self, attention_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention distribution.
        
        Args:
            attention_probs: Attention probabilities, shape (batch, heads, seq, seq)
        
        Returns:
            Entropy per head, shape (batch, heads, seq)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        # entropy = -sum(p * log(p))
        entropy = -torch.sum(
            attention_probs * torch.log(attention_probs + eps),
            dim=-1
        )
        return entropy
    
    def _compute_position_metrics(
        self,
        attention_probs: torch.Tensor,
        seq_len: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute position-based metrics for attention.
        
        Args:
            attention_probs: Shape (batch, heads, seq, seq)
            seq_len: Current sequence length
        
        Returns:
            Dict of metrics per head
        """
        batch, num_heads, query_len, key_len = attention_probs.shape
        
        # For each query position, compute attention to different regions
        # We'll average over query positions
        
        # Sink attention (first sink_size tokens)
        sink_mask = torch.zeros(key_len, device=attention_probs.device)
        sink_mask[:min(self.sink_size, key_len)] = 1.0
        sink_attention = (attention_probs * sink_mask.view(1, 1, 1, -1)).sum(dim=-1).mean(dim=-1)
        
        # Recent attention (last recent_ratio of tokens)
        recent_start = max(0, int(key_len * (1 - self.recent_ratio)))
        recent_mask = torch.zeros(key_len, device=attention_probs.device)
        recent_mask[recent_start:] = 1.0
        recent_attention = (attention_probs * recent_mask.view(1, 1, 1, -1)).sum(dim=-1).mean(dim=-1)
        
        # Local attention (within local_window positions)
        # This is position-dependent, so we need to compute per query position
        local_attention_sum = torch.zeros(batch, num_heads, device=attention_probs.device)
        for q in range(query_len):
            local_start = max(0, q - self.local_window)
            local_end = min(key_len, q + 1)  # Causal: can only attend to past
            local_mask = torch.zeros(key_len, device=attention_probs.device)
            local_mask[local_start:local_end] = 1.0
            local_attention_sum += (attention_probs[:, :, q, :] * local_mask).sum(dim=-1)
        local_attention = local_attention_sum / query_len
        
        # Global attention (everything else)
        global_attention = 1.0 - sink_attention - local_attention
        global_attention = torch.clamp(global_attention, min=0.0)
        
        return {
            "sink": sink_attention,      # Shape: (batch, heads)
            "recent": recent_attention,  # Shape: (batch, heads)
            "local": local_attention,    # Shape: (batch, heads)
            "global": global_attention,  # Shape: (batch, heads)
        }
    
    def _compute_relative_position_distribution(
        self,
        attention_probs: torch.Tensor,
        max_distance: int = 64,
    ) -> torch.Tensor:
        """
        Compute how attention is distributed over relative positions.
        
        Args:
            attention_probs: Shape (batch, heads, seq, seq)
            max_distance: Maximum relative distance to track
        
        Returns:
            Distribution over relative positions, shape (batch, heads, max_distance)
        """
        batch, num_heads, query_len, key_len = attention_probs.shape
        device = attention_probs.device
        
        # Initialize distribution
        rel_pos_dist = torch.zeros(batch, num_heads, max_distance, device=device)
        
        # For each query position, accumulate attention by relative distance
        for q in range(query_len):
            for k in range(min(q + 1, key_len)):  # Causal mask
                rel_dist = q - k  # How far back we're looking
                if rel_dist < max_distance:
                    rel_pos_dist[:, :, rel_dist] += attention_probs[:, :, q, k]
        
        # Normalize by number of query positions
        rel_pos_dist = rel_pos_dist / query_len
        
        return rel_pos_dist
    
    def _compute_uniformity_score(self, attention_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute how uniform the attention distribution is.
        Uses KL divergence from uniform distribution.
        Lower score = more uniform (dead head indicator).
        
        Args:
            attention_probs: Shape (batch, heads, seq, seq)
        
        Returns:
            Uniformity score per head, shape (batch, heads)
        """
        batch, num_heads, query_len, key_len = attention_probs.shape
        
        # For causal attention, uniform distribution varies by position
        # Average over query positions
        kl_divs = []
        
        for q in range(query_len):
            # Only first q+1 positions are valid (causal)
            valid_len = min(q + 1, key_len)
            if valid_len < 2:
                continue
            
            # Uniform distribution for this position
            uniform = torch.ones(valid_len, device=attention_probs.device) / valid_len
            
            # Get attention for this query position
            attn = attention_probs[:, :, q, :valid_len]  # (batch, heads, valid_len)
            
            # KL divergence: sum(p * log(p/q))
            eps = 1e-10
            kl = torch.sum(attn * torch.log((attn + eps) / (uniform + eps)), dim=-1)
            kl_divs.append(kl)
        
        if not kl_divs:
            return torch.zeros(batch, num_heads, device=attention_probs.device)
        
        # Average KL divergence across positions
        kl_div_mean = torch.stack(kl_divs, dim=-1).mean(dim=-1)  # (batch, heads)
        
        return kl_div_mean
    
    def analyze_single_forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single forward pass and collect attention statistics.
        
        Args:
            input_ids: Input token IDs, shape (batch, seq)
            position_offset: Offset for position tracking
        
        Returns:
            Dict of statistics tensors
        """
        self.model.eval()
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                output_attentions=True,
                use_cache=False,  # Don't need cache for analysis
            )
        
        # outputs.attentions: tuple of (batch, heads, seq, seq) for each layer
        attentions = outputs.attentions
        
        all_stats = {
            "entropy": [],           # Per layer: (batch, heads, seq)
            "max_attention": [],     # Per layer: (batch, heads, seq)
            "position_metrics": [],  # Per layer: dict of (batch, heads)
            "rel_pos_dist": [],      # Per layer: (batch, heads, max_dist)
            "uniformity": [],        # Per layer: (batch, heads)
        }
        
        for layer_idx, attn in enumerate(attentions):
            # attn shape: (batch, heads, seq, seq)
            seq_len = attn.shape[-1]
            
            # Entropy
            entropy = self._compute_entropy(attn)  # (batch, heads, seq)
            all_stats["entropy"].append(entropy)
            
            # Max attention per query position
            max_attn = attn.max(dim=-1).values  # (batch, heads, seq)
            all_stats["max_attention"].append(max_attn)
            
            # Position metrics
            pos_metrics = self._compute_position_metrics(attn, seq_len)
            all_stats["position_metrics"].append(pos_metrics)
            
            # Relative position distribution
            rel_pos = self._compute_relative_position_distribution(attn)
            all_stats["rel_pos_dist"].append(rel_pos)
            
            # Uniformity score
            uniformity = self._compute_uniformity_score(attn)
            all_stats["uniformity"].append(uniformity)
        
        return all_stats
    
    def analyze(
        self,
        text: str,
        max_tokens: int = 2048,
        chunk_size: int = 512,
        show_progress: bool = True,
    ) -> Tuple[List[HeadStatistics], List[HeadClassification]]:
        """
        Analyze attention patterns for the given text.
        
        Args:
            text: Input text to analyze
            max_tokens: Maximum tokens to process
            chunk_size: Process in chunks to manage memory
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (head_statistics, head_classifications)
        """
        # Tokenize
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids[:, :max_tokens].to(self.device)
        total_len = input_ids.shape[1]
        
        print(f"Analyzing {total_len} tokens across {self.num_layers} layers, {self.num_heads} heads per layer")
        
        # Initialize accumulators
        accumulated = {
            "entropy_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "entropy_sq_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "max_attn_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "sink_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "recent_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "local_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "global_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "uniformity_sum": torch.zeros(self.num_layers, self.num_heads, device=self.device),
            "rel_pos_dist_sum": torch.zeros(self.num_layers, self.num_heads, 64, device=self.device),
            "count": 0,
        }
        
        # Process in chunks
        num_chunks = (total_len + chunk_size - 1) // chunk_size
        
        chunk_iter = range(0, total_len, chunk_size)
        if show_progress:
            chunk_iter = tqdm(chunk_iter, desc="Analyzing chunks", total=num_chunks)
        
        for start in chunk_iter:
            end = min(start + chunk_size, total_len)
            chunk_ids = input_ids[:, start:end]
            
            if chunk_ids.shape[1] < 10:  # Skip very short chunks
                continue
            
            # Analyze this chunk
            stats = self.analyze_single_forward(chunk_ids, position_offset=start)
            
            # Accumulate statistics
            for layer_idx in range(self.num_layers):
                # Entropy: average over sequence
                entropy_mean = stats["entropy"][layer_idx].mean(dim=-1).squeeze(0)  # (heads,)
                accumulated["entropy_sum"][layer_idx] += entropy_mean
                accumulated["entropy_sq_sum"][layer_idx] += entropy_mean ** 2
                
                # Max attention
                max_attn_mean = stats["max_attention"][layer_idx].mean(dim=-1).squeeze(0)
                accumulated["max_attn_sum"][layer_idx] += max_attn_mean
                
                # Position metrics
                accumulated["sink_sum"][layer_idx] += stats["position_metrics"][layer_idx]["sink"].squeeze(0)
                accumulated["recent_sum"][layer_idx] += stats["position_metrics"][layer_idx]["recent"].squeeze(0)
                accumulated["local_sum"][layer_idx] += stats["position_metrics"][layer_idx]["local"].squeeze(0)
                accumulated["global_sum"][layer_idx] += stats["position_metrics"][layer_idx]["global"].squeeze(0)
                
                # Uniformity
                accumulated["uniformity_sum"][layer_idx] += stats["uniformity"][layer_idx].squeeze(0)
                
                # Relative position distribution
                accumulated["rel_pos_dist_sum"][layer_idx] += stats["rel_pos_dist"][layer_idx].squeeze(0)
            
            accumulated["count"] += 1
        
        # Compute final statistics
        count = accumulated["count"]
        if count == 0:
            raise ValueError("No chunks processed")
        
        head_stats = []
        head_classifications = []
        
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                # Compute means
                mean_entropy = (accumulated["entropy_sum"][layer_idx, head_idx] / count).item()
                mean_sq_entropy = (accumulated["entropy_sq_sum"][layer_idx, head_idx] / count).item()
                std_entropy = np.sqrt(max(0, mean_sq_entropy - mean_entropy ** 2))
                
                max_attn_mean = (accumulated["max_attn_sum"][layer_idx, head_idx] / count).item()
                
                sink_ratio = (accumulated["sink_sum"][layer_idx, head_idx] / count).item()
                recent_ratio = (accumulated["recent_sum"][layer_idx, head_idx] / count).item()
                local_ratio = (accumulated["local_sum"][layer_idx, head_idx] / count).item()
                global_ratio = (accumulated["global_sum"][layer_idx, head_idx] / count).item()
                
                uniformity = (accumulated["uniformity_sum"][layer_idx, head_idx] / count).item()
                
                rel_pos_dist = (accumulated["rel_pos_dist_sum"][layer_idx, head_idx] / count).tolist()
                
                # Create HeadStatistics
                stats = HeadStatistics(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    mean_entropy=mean_entropy,
                    std_entropy=std_entropy,
                    max_attention_mean=max_attn_mean,
                    position_preference={
                        "sink": sink_ratio,
                        "recent": recent_ratio,
                        "local": local_ratio,
                        "global": global_ratio,
                    },
                    sink_ratio=sink_ratio,
                    uniformity_score=uniformity,
                    relative_position_dist=rel_pos_dist,
                )
                head_stats.append(stats)
                
                # Classify head
                classification = self._classify_head(stats)
                head_classifications.append(classification)
        
        return head_stats, head_classifications
    
    def _classify_head(self, stats: HeadStatistics) -> HeadClassification:
        """
        Classify a head based on its statistics with refined categorization.
        
        The refined classification distinguishes between:
        - SINK_POSITIONAL: Low entropy + high sink ratio -> needs sink tokens + small window
        - TRUE_POSITIONAL: Low entropy + low sink + high local -> window only
        - SINK_MIXED: Medium entropy + high sink -> needs sink tokens + larger window
        - GATHERING: High entropy -> needs full KV cache
        - DEAD: Near-uniform distribution -> can be pruned
        - MIXED: Everything else
        
        Args:
            stats: HeadStatistics for this head
        
        Returns:
            HeadClassification with detailed compression recommendations
        """
        layer_idx = stats.layer_idx
        head_idx = stats.head_idx
        
        # Classification thresholds (tuned based on Pythia-2.8b analysis)
        LOW_ENTROPY_THRESHOLD = 1.5       # Below this = positional head candidate
        HIGH_ENTROPY_THRESHOLD = 3.0      # Above this = gathering head candidate
        DEAD_UNIFORMITY_THRESHOLD = 0.1   # Below this KL divergence = dead head
        HIGH_LOCAL_THRESHOLD = 0.6        # Above this local ratio = true positional head
        HIGH_SINK_THRESHOLD = 0.3         # Above this sink ratio = sink-focused head
        
        # Default values
        head_type = HeadType.MIXED
        confidence = 0.5
        can_prune = False
        can_limit_window = False
        recommended_window = -1
        keep_sinks = False
        sink_size = 4
        use_full_cache = True
        compression_strategy = "none"
        
        # Check for dead head first (near-uniform distribution)
        if stats.uniformity_score < DEAD_UNIFORMITY_THRESHOLD:
            head_type = HeadType.DEAD
            confidence = 1.0 - stats.uniformity_score / DEAD_UNIFORMITY_THRESHOLD
            can_prune = True
            use_full_cache = False
            compression_strategy = "prune"
        
        # Check for low entropy heads (positional candidates)
        elif stats.mean_entropy < LOW_ENTROPY_THRESHOLD:
            # Distinguish between sink-positional and true-positional
            if stats.sink_ratio > HIGH_SINK_THRESHOLD:
                # SINK_POSITIONAL: High sink ratio - needs sink tokens + small window
                head_type = HeadType.SINK_POSITIONAL
                confidence = stats.sink_ratio
                can_limit_window = True
                keep_sinks = True
                sink_size = 4
                recommended_window = 8  # Small recent window
                use_full_cache = False
                compression_strategy = "sink_window"
                
            elif stats.position_preference["local"] > HIGH_LOCAL_THRESHOLD:
                # TRUE_POSITIONAL: High local attention, low sink - window only
                head_type = HeadType.TRUE_POSITIONAL
                confidence = (1.0 - stats.mean_entropy / LOW_ENTROPY_THRESHOLD) * \
                            (stats.position_preference["local"] / HIGH_LOCAL_THRESHOLD)
                confidence = min(1.0, confidence)
                can_limit_window = True
                keep_sinks = False  # No need for sinks
                use_full_cache = False
                compression_strategy = "window_only"
                
                # Recommend window based on where 80% of attention is concentrated
                cumsum = 0
                for i, val in enumerate(stats.relative_position_dist[:16]):
                    cumsum += val
                    if cumsum > 0.8:
                        recommended_window = max(8, i + 4)  # Add buffer
                        break
                if recommended_window == -1:
                    recommended_window = 16
            else:
                # Low entropy but neither high sink nor high local - keep as mixed
                head_type = HeadType.MIXED
                confidence = 0.5
                use_full_cache = True
                compression_strategy = "none"
        
        # Check for medium entropy with high sink ratio (SINK_MIXED)
        elif stats.mean_entropy < HIGH_ENTROPY_THRESHOLD and stats.sink_ratio > HIGH_SINK_THRESHOLD:
            head_type = HeadType.SINK_MIXED
            confidence = stats.sink_ratio * (1.0 - (stats.mean_entropy - LOW_ENTROPY_THRESHOLD) / 
                                             (HIGH_ENTROPY_THRESHOLD - LOW_ENTROPY_THRESHOLD))
            confidence = min(1.0, max(0.0, confidence))
            can_limit_window = True
            keep_sinks = True
            sink_size = 4
            # Larger window for mixed heads (16-32 based on entropy)
            entropy_ratio = (stats.mean_entropy - LOW_ENTROPY_THRESHOLD) / \
                           (HIGH_ENTROPY_THRESHOLD - LOW_ENTROPY_THRESHOLD)
            recommended_window = int(16 + entropy_ratio * 16)  # 16-32 tokens
            use_full_cache = False
            compression_strategy = "sink_window"
        
        # Check for gathering head (high entropy, distributed attention)
        elif stats.mean_entropy > HIGH_ENTROPY_THRESHOLD:
            head_type = HeadType.GATHERING
            # Higher confidence for very high entropy
            confidence = min(1.0, (stats.mean_entropy - HIGH_ENTROPY_THRESHOLD) / 2.0)
            use_full_cache = True
            compression_strategy = "full"
        
        # For backward compatibility, also set legacy POSITIONAL type
        legacy_type = head_type
        if head_type in (HeadType.SINK_POSITIONAL, HeadType.TRUE_POSITIONAL):
            legacy_type = HeadType.POSITIONAL
        
        return HeadClassification(
            layer_idx=layer_idx,
            head_idx=head_idx,
            head_type=head_type,
            confidence=confidence,
            can_prune=can_prune,
            can_limit_window=can_limit_window,
            recommended_window=recommended_window,
            keep_sinks=keep_sinks,
            sink_size=sink_size,
            use_full_cache=use_full_cache,
            compression_strategy=compression_strategy,
        )
    
    def save_results(
        self,
        stats: List[HeadStatistics],
        classifications: List[HeadClassification],
        output_dir: str,
    ) -> None:
        """
        Save analysis results to JSON files.
        
        Args:
            stats: List of HeadStatistics
            classifications: List of HeadClassification
            output_dir: Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics
        stats_data = {
            "model_info": {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "sink_size": self.sink_size,
                "local_window": self.local_window,
                "recent_ratio": self.recent_ratio,
            },
            "head_statistics": [s.to_dict() for s in stats],
        }
        
        with open(os.path.join(output_dir, "head_statistics.json"), "w") as f:
            json.dump(stats_data, f, indent=2)
        
        # Save classifications
        class_data = {
            "classifications": [c.to_dict() for c in classifications],
            "summary": self._compute_classification_summary(classifications),
        }
        
        with open(os.path.join(output_dir, "head_classifications.json"), "w") as f:
            json.dump(class_data, f, indent=2)
        
        print(f"Results saved to {output_dir}")
    
    def _compute_classification_summary(
        self,
        classifications: List[HeadClassification]
    ) -> dict:
        """Compute summary statistics for classifications."""
        total = len(classifications)
        
        type_counts = {t.value: 0 for t in HeadType}
        prunable_count = 0
        limitable_count = 0
        
        # Compression strategy counts
        strategy_counts = {
            "none": 0,
            "prune": 0,
            "window_only": 0,
            "sink_window": 0,
            "full": 0,
        }
        
        sink_heads_count = 0  # Heads that need sink tokens preserved
        
        for c in classifications:
            type_counts[c.head_type.value] += 1
            if c.can_prune:
                prunable_count += 1
            if c.can_limit_window:
                limitable_count += 1
            if c.compression_strategy in strategy_counts:
                strategy_counts[c.compression_strategy] += 1
            if c.keep_sinks:
                sink_heads_count += 1
        
        return {
            "total_heads": total,
            "type_distribution": type_counts,
            "type_percentages": {k: v / total * 100 for k, v in type_counts.items()},
            "prunable_heads": prunable_count,
            "prunable_percentage": prunable_count / total * 100,
            "limitable_heads": limitable_count,
            "limitable_percentage": limitable_count / total * 100,
            "compression_strategies": strategy_counts,
            "strategy_percentages": {k: v / total * 100 for k, v in strategy_counts.items()},
            "sink_heads": sink_heads_count,
            "sink_heads_percentage": sink_heads_count / total * 100,
        }


def analyze_attention_heads(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 2048,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Tuple[List[HeadStatistics], List[HeadClassification]]:
    """
    Convenience function to analyze attention heads.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to analyze
        max_tokens: Maximum tokens to process
        output_dir: If provided, save results to this directory
        device: Device to use
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (head_statistics, head_classifications)
    """
    analyzer = AttentionAnalyzer(model, tokenizer, device)
    stats, classifications = analyzer.analyze(
        text, max_tokens=max_tokens, show_progress=show_progress
    )
    
    if output_dir:
        analyzer.save_results(stats, classifications, output_dir)
    
    return stats, classifications


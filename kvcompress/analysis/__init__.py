"""
Attention Analysis Module for KV Cache Optimization

This module provides tools for analyzing attention head behavior,
enabling head specialization detection and optimized compression strategies.

Key Components:
    - AttentionAnalyzer: Main class for analyzing attention patterns
    - HeadStatistics: Data class for head-level statistics
    - HeadClassification: Data class for classification results
    - HeadType: Enum defining head type categories

Head Types:
    - SINK_POSITIONAL: Low entropy + high sink ratio -> needs sink + small window
    - TRUE_POSITIONAL: Low entropy + low sink + high local -> window only
    - SINK_MIXED: Medium entropy + high sink -> needs sink + larger window
    - GATHERING: High entropy -> needs full KV cache
    - DEAD: Near-uniform distribution -> can be pruned
    - MIXED: Does not fit clearly into other categories

Usage:
    >>> from kvcompress.analysis import AttentionAnalyzer, HeadType
    >>> analyzer = AttentionAnalyzer(model, tokenizer)
    >>> stats, classifications = analyzer.analyze(text)
    >>> sink_heads = [c for c in classifications if c.head_type == HeadType.SINK_POSITIONAL]

See Also:
    - kvcompress.methods.head_aware_compress: Use classifications for compression
    - scripts/analyze_attention.py: Command-line analysis tool
    - scripts/validate_head_classification.py: Validation and visualization
"""

from .attention_analyzer import (
    AttentionAnalyzer,
    HeadStatistics,
    HeadClassification,
    HeadType,
    analyze_attention_heads,
)
from .visualize import (
    plot_entropy_heatmap,
    plot_position_preference,
    plot_sink_ratio,
    plot_head_clustering,
    create_full_report,
)

__all__ = [
    # Analyzer
    'AttentionAnalyzer',
    'HeadStatistics',
    'HeadClassification',
    'HeadType',
    'analyze_attention_heads',
    # Visualization
    'plot_entropy_heatmap',
    'plot_position_preference',
    'plot_sink_ratio',
    'plot_head_clustering',
    'create_full_report',
]


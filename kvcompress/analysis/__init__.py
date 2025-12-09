"""
Attention Analysis Module for KV Cache Optimization

This module provides tools for analyzing attention head behavior,
enabling head specialization detection and pruning decisions.
"""

from .attention_analyzer import (
    AttentionAnalyzer,
    HeadStatistics,
    HeadClassification,
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
    'analyze_attention_heads',
    # Visualization
    'plot_entropy_heatmap',
    'plot_position_preference',
    'plot_sink_ratio',
    'plot_head_clustering',
    'create_full_report',
]


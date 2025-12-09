"""
Visualization Tools for Attention Head Analysis

This module provides visualization functions for attention head statistics:
1. Entropy heatmap - Shows entropy distribution across layers and heads
2. Position preference - Shows where each head focuses attention
3. Sink ratio analysis - Shows attention allocated to initial tokens
4. Head clustering - Automatic classification visualization
"""

import os
import json
from typing import List, Optional, Dict, Tuple
import numpy as np

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from .attention_analyzer import HeadStatistics, HeadClassification, HeadType


def _setup_style():
    """Setup matplotlib style for better visualizations."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def plot_entropy_heatmap(
    stats: List[HeadStatistics],
    output_path: str,
    title: str = "Attention Entropy Heatmap",
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    Create a heatmap showing entropy distribution across layers and heads.
    
    Lower entropy = more focused attention (positional head candidate)
    Higher entropy = more distributed attention (gathering head candidate)
    
    Args:
        stats: List of HeadStatistics
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size (width, height)
    """
    _setup_style()
    
    # Organize data into matrix
    layers = sorted(set(s.layer_idx for s in stats))
    heads = sorted(set(s.head_idx for s in stats))
    num_layers = len(layers)
    num_heads = len(heads)
    
    entropy_matrix = np.zeros((num_layers, num_heads))
    for s in stats:
        entropy_matrix[s.layer_idx, s.head_idx] = s.mean_entropy
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap: blue (low) -> white (medium) -> red (high)
    colors = ['#2166AC', '#67A9CF', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#EF8A62', '#B2182B']
    cmap = LinearSegmentedColormap.from_list('entropy', colors, N=256)
    
    im = ax.imshow(entropy_matrix, aspect='auto', cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Entropy (nats)', rotation=270, labelpad=15)
    
    # Labels
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    ax.set_title(title)
    
    # Set ticks
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels([str(h) for h in heads])
    ax.set_yticklabels([str(l) for l in layers])
    
    # Add value annotations for small models
    if num_layers * num_heads <= 64:
        for i in range(num_layers):
            for j in range(num_heads):
                val = entropy_matrix[i, j]
                color = 'white' if val > np.median(entropy_matrix) else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=7)
    
    # Add interpretation guide
    ax.text(1.02, 0.95, 'High entropy\n(Gathering)', transform=ax.transAxes, 
            fontsize=8, color='#B2182B', va='top')
    ax.text(1.02, 0.05, 'Low entropy\n(Positional)', transform=ax.transAxes, 
            fontsize=8, color='#2166AC', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved entropy heatmap to {output_path}")


def plot_position_preference(
    stats: List[HeadStatistics],
    output_path: str,
    title: str = "Attention Position Preference by Head",
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Create a visualization showing where each head focuses attention.
    
    Categories:
    - Sink: Attention on first 4 tokens (attention sinks)
    - Local: Attention within 8 positions
    - Recent: Attention on last 10% of tokens
    - Global: Everything else
    
    Args:
        stats: List of HeadStatistics
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    _setup_style()
    
    # Organize data
    layers = sorted(set(s.layer_idx for s in stats))
    heads = sorted(set(s.head_idx for s in stats))
    num_layers = len(layers)
    num_heads = len(heads)
    
    # Create stacked bar data
    categories = ['sink', 'local', 'recent', 'global']
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    
    # Create subplot for each layer
    fig, axes = plt.subplots(num_layers, 1, figsize=figsize, sharex=True)
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, ax in enumerate(axes):
        layer_stats = [s for s in stats if s.layer_idx == layer_idx]
        layer_stats.sort(key=lambda x: x.head_idx)
        
        x = np.arange(num_heads)
        bottom = np.zeros(num_heads)
        
        for cat, color in zip(categories, colors):
            values = [s.position_preference.get(cat, 0) for s in layer_stats]
            ax.bar(x, values, bottom=bottom, label=cat.capitalize(), color=color, width=0.8)
            bottom += values
        
        ax.set_ylabel(f'Layer {layer_idx}')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Only show legend on first subplot
        if layer_idx == 0:
            ax.legend(loc='upper right', ncol=4, fontsize=8)
    
    axes[-1].set_xlabel('Head Index')
    axes[-1].set_xticks(range(num_heads))
    axes[-1].set_xticklabels([str(h) for h in heads])
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved position preference plot to {output_path}")


def plot_sink_ratio(
    stats: List[HeadStatistics],
    output_path: str,
    title: str = "Attention Sink Ratio Analysis",
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    Create a heatmap showing attention allocated to initial tokens (sinks).
    
    High sink ratio indicates the head relies heavily on attention sinks,
    which is important for StreamingLLM-style optimizations.
    
    Args:
        stats: List of HeadStatistics
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    _setup_style()
    
    layers = sorted(set(s.layer_idx for s in stats))
    heads = sorted(set(s.head_idx for s in stats))
    num_layers = len(layers)
    num_heads = len(heads)
    
    # Create matrices for sink ratio and local ratio
    sink_matrix = np.zeros((num_layers, num_heads))
    local_matrix = np.zeros((num_layers, num_heads))
    
    for s in stats:
        sink_matrix[s.layer_idx, s.head_idx] = s.sink_ratio
        local_matrix[s.layer_idx, s.head_idx] = s.position_preference.get('local', 0)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Sink ratio heatmap
    im1 = axes[0].imshow(sink_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=0.5)
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Sink Attention Ratio', rotation=270, labelpad=15)
    axes[0].set_xlabel('Head Index')
    axes[0].set_ylabel('Layer Index')
    axes[0].set_title('Attention on Initial Tokens (Sinks)')
    axes[0].set_xticks(range(num_heads))
    axes[0].set_yticks(range(num_layers))
    
    # Add annotations for high sink ratio heads
    if num_layers * num_heads <= 64:
        for i in range(num_layers):
            for j in range(num_heads):
                val = sink_matrix[i, j]
                if val > 0.2:  # Highlight high values
                    axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', 
                               color='white', fontsize=7, fontweight='bold')
    
    # Local ratio heatmap
    im2 = axes[1].imshow(local_matrix, aspect='auto', cmap='Blues', vmin=0, vmax=1.0)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Local Attention Ratio', rotation=270, labelpad=15)
    axes[1].set_xlabel('Head Index')
    axes[1].set_ylabel('Layer Index')
    axes[1].set_title('Attention within Local Window')
    axes[1].set_xticks(range(num_heads))
    axes[1].set_yticks(range(num_layers))
    
    # Add annotations
    if num_layers * num_heads <= 64:
        for i in range(num_layers):
            for j in range(num_heads):
                val = local_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                axes[1].text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=color, fontsize=7)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved sink ratio analysis to {output_path}")


def plot_head_clustering(
    stats: List[HeadStatistics],
    classifications: List[HeadClassification],
    output_path: str,
    title: str = "Head Classification and Clustering",
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Create comprehensive visualization of head classifications.
    
    Shows:
    1. Classification matrix (color-coded by type)
    2. Entropy vs Uniformity scatter plot
    3. Summary statistics pie chart
    4. Relative position distribution for selected heads
    
    Args:
        stats: List of HeadStatistics
        classifications: List of HeadClassification
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    _setup_style()
    
    layers = sorted(set(s.layer_idx for s in stats))
    heads = sorted(set(s.head_idx for s in stats))
    num_layers = len(layers)
    num_heads = len(heads)
    
    # Color scheme for head types
    type_colors = {
        HeadType.POSITIONAL: '#377EB8',  # Blue
        HeadType.GATHERING: '#4DAF4A',   # Green
        HeadType.DEAD: '#E41A1C',        # Red
        HeadType.MIXED: '#984EA3',       # Purple
    }
    
    fig = plt.figure(figsize=figsize)
    
    # 1. Classification matrix (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    
    class_matrix = np.zeros((num_layers, num_heads))
    type_to_num = {HeadType.POSITIONAL: 0, HeadType.GATHERING: 1, HeadType.DEAD: 2, HeadType.MIXED: 3}
    
    for c in classifications:
        class_matrix[c.layer_idx, c.head_idx] = type_to_num[c.head_type]
    
    # Custom colormap
    cmap_colors = [type_colors[HeadType.POSITIONAL], type_colors[HeadType.GATHERING],
                   type_colors[HeadType.DEAD], type_colors[HeadType.MIXED]]
    cmap = matplotlib.colors.ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax1.imshow(class_matrix, aspect='auto', cmap=cmap, norm=norm)
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Layer Index')
    ax1.set_title('Head Classification Matrix')
    ax1.set_xticks(range(num_heads))
    ax1.set_yticks(range(num_layers))
    
    # Legend
    patches = [mpatches.Patch(color=type_colors[t], label=t.value.capitalize()) 
               for t in HeadType]
    ax1.legend(handles=patches, loc='upper right', fontsize=8)
    
    # 2. Entropy vs Uniformity scatter (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    for head_type in HeadType:
        type_stats = [s for s, c in zip(stats, classifications) if c.head_type == head_type]
        if type_stats:
            entropies = [s.mean_entropy for s in type_stats]
            uniformities = [s.uniformity_score for s in type_stats]
            ax2.scatter(entropies, uniformities, c=type_colors[head_type], 
                       label=head_type.value.capitalize(), alpha=0.7, s=50)
    
    ax2.set_xlabel('Mean Entropy')
    ax2.set_ylabel('Uniformity Score (KL Divergence)')
    ax2.set_title('Entropy vs Uniformity by Head Type')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add threshold lines
    ax2.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, label='Low entropy threshold')
    ax2.axvline(x=3.0, color='gray', linestyle=':', alpha=0.5, label='High entropy threshold')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Dead head threshold')
    
    # 3. Summary pie chart (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    type_counts = {t: 0 for t in HeadType}
    for c in classifications:
        type_counts[c.head_type] += 1
    
    labels = [f'{t.value.capitalize()}\n({type_counts[t]})' for t in HeadType if type_counts[t] > 0]
    sizes = [type_counts[t] for t in HeadType if type_counts[t] > 0]
    colors = [type_colors[t] for t in HeadType if type_counts[t] > 0]
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Head Type Distribution')
    
    # 4. Relative position distribution (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Select representative heads from each type
    representative_heads = []
    for head_type in [HeadType.POSITIONAL, HeadType.GATHERING, HeadType.DEAD]:
        type_classifications = [c for c in classifications if c.head_type == head_type]
        if type_classifications:
            # Get the one with highest confidence
            best = max(type_classifications, key=lambda x: x.confidence)
            matching_stat = next(s for s in stats 
                               if s.layer_idx == best.layer_idx and s.head_idx == best.head_idx)
            representative_heads.append((best, matching_stat))
    
    x = np.arange(min(32, len(stats[0].relative_position_dist) if stats else 32))
    for cls, stat in representative_heads:
        label = f"L{cls.layer_idx}H{cls.head_idx} ({cls.head_type.value})"
        y = stat.relative_position_dist[:len(x)]
        ax4.plot(x, y, label=label, color=type_colors[cls.head_type], linewidth=2)
    
    ax4.set_xlabel('Relative Position (tokens back)')
    ax4.set_ylabel('Attention Weight')
    ax4.set_title('Relative Position Distribution (Representative Heads)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 31)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved head clustering visualization to {output_path}")


def plot_relative_position_heatmap(
    stats: List[HeadStatistics],
    output_path: str,
    title: str = "Relative Position Attention Distribution",
    max_positions: int = 32,
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """
    Create a detailed heatmap of relative position attention for each head.
    
    Args:
        stats: List of HeadStatistics
        output_path: Path to save the figure
        title: Plot title
        max_positions: Maximum relative positions to show
        figsize: Figure size
    """
    _setup_style()
    
    layers = sorted(set(s.layer_idx for s in stats))
    heads = sorted(set(s.head_idx for s in stats))
    num_layers = len(layers)
    num_heads = len(heads)
    
    # Create figure with subplots for each layer
    fig, axes = plt.subplots(num_layers, 1, figsize=figsize, sharex=True)
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, ax in enumerate(axes):
        layer_stats = [s for s in stats if s.layer_idx == layer_idx]
        layer_stats.sort(key=lambda x: x.head_idx)
        
        # Create matrix for this layer
        matrix = np.zeros((num_heads, max_positions))
        for s in layer_stats:
            dist = s.relative_position_dist[:max_positions]
            matrix[s.head_idx, :len(dist)] = dist
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_ylabel(f'Layer {layer_idx}\nHead')
        ax.set_yticks(range(num_heads))
        ax.set_yticklabels([str(h) for h in heads])
    
    axes[-1].set_xlabel('Relative Position (tokens back)')
    axes[-1].set_xticks(range(0, max_positions, 4))
    
    # Add colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved relative position heatmap to {output_path}")


def create_full_report(
    stats: List[HeadStatistics],
    classifications: List[HeadClassification],
    output_dir: str,
    model_name: str = "model",
) -> None:
    """
    Create a complete analysis report with all visualizations.
    
    Args:
        stats: List of HeadStatistics
        classifications: List of HeadClassification
        output_dir: Output directory for all files
        model_name: Name of the model for titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating analysis report for {model_name}...")
    
    # 1. Entropy heatmap
    plot_entropy_heatmap(
        stats,
        os.path.join(output_dir, "entropy_heatmap.png"),
        title=f"{model_name} - Attention Entropy Heatmap"
    )
    
    # 2. Position preference
    plot_position_preference(
        stats,
        os.path.join(output_dir, "position_preference.png"),
        title=f"{model_name} - Position Preference by Head"
    )
    
    # 3. Sink ratio analysis
    plot_sink_ratio(
        stats,
        os.path.join(output_dir, "sink_ratio_analysis.png"),
        title=f"{model_name} - Attention Sink Analysis"
    )
    
    # 4. Head clustering
    plot_head_clustering(
        stats,
        classifications,
        os.path.join(output_dir, "head_clustering.png"),
        title=f"{model_name} - Head Classification"
    )
    
    # 5. Relative position heatmap
    plot_relative_position_heatmap(
        stats,
        os.path.join(output_dir, "relative_position_heatmap.png"),
        title=f"{model_name} - Relative Position Attention"
    )
    
    # 6. Generate text summary
    _generate_text_summary(stats, classifications, output_dir, model_name)
    
    print(f"\nComplete report saved to {output_dir}/")


def _generate_text_summary(
    stats: List[HeadStatistics],
    classifications: List[HeadClassification],
    output_dir: str,
    model_name: str,
) -> None:
    """Generate a text summary of the analysis."""
    
    # Compute statistics
    num_layers = len(set(s.layer_idx for s in stats))
    num_heads = len(set(s.head_idx for s in stats))
    total_heads = len(classifications)
    
    type_counts = {t: 0 for t in HeadType}
    prunable = []
    limitable = []
    
    for c in classifications:
        type_counts[c.head_type] += 1
        if c.can_prune:
            prunable.append(c)
        if c.can_limit_window:
            limitable.append(c)
    
    # Write summary
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"ATTENTION HEAD ANALYSIS SUMMARY\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"MODEL ARCHITECTURE\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Heads per layer: {num_heads}\n")
        f.write(f"Total heads: {total_heads}\n\n")
        
        f.write(f"HEAD TYPE DISTRIBUTION\n")
        f.write(f"-" * 40 + "\n")
        for head_type in HeadType:
            count = type_counts[head_type]
            pct = count / total_heads * 100
            f.write(f"{head_type.value.capitalize():12} : {count:3} heads ({pct:5.1f}%)\n")
        f.write("\n")
        
        f.write(f"OPTIMIZATION OPPORTUNITIES\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Prunable heads (can be removed): {len(prunable)} ({len(prunable)/total_heads*100:.1f}%)\n")
        f.write(f"Limitable heads (can use small KV cache): {len(limitable)} ({len(limitable)/total_heads*100:.1f}%)\n\n")
        
        if prunable:
            f.write(f"PRUNABLE HEADS (Dead Heads)\n")
            f.write(f"-" * 40 + "\n")
            for c in prunable:
                f.write(f"  Layer {c.layer_idx}, Head {c.head_idx} (confidence: {c.confidence:.2f})\n")
            f.write("\n")
        
        if limitable:
            f.write(f"WINDOW-LIMITABLE HEADS (Positional Heads)\n")
            f.write(f"-" * 40 + "\n")
            for c in limitable:
                f.write(f"  Layer {c.layer_idx}, Head {c.head_idx} -> window={c.recommended_window} (confidence: {c.confidence:.2f})\n")
            f.write("\n")
        
        # Entropy statistics
        entropies = [s.mean_entropy for s in stats]
        f.write(f"ENTROPY STATISTICS\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Min entropy: {min(entropies):.3f}\n")
        f.write(f"Max entropy: {max(entropies):.3f}\n")
        f.write(f"Mean entropy: {np.mean(entropies):.3f}\n")
        f.write(f"Std entropy: {np.std(entropies):.3f}\n\n")
        
        # Sink ratio statistics
        sink_ratios = [s.sink_ratio for s in stats]
        f.write(f"SINK RATIO STATISTICS\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Min sink ratio: {min(sink_ratios):.3f}\n")
        f.write(f"Max sink ratio: {max(sink_ratios):.3f}\n")
        f.write(f"Mean sink ratio: {np.mean(sink_ratios):.3f}\n")
        high_sink_heads = sum(1 for r in sink_ratios if r > 0.2)
        f.write(f"Heads with sink ratio > 20%: {high_sink_heads}\n\n")
        
        f.write(f"RECOMMENDATIONS\n")
        f.write(f"-" * 40 + "\n")
        if len(prunable) > 0:
            f.write(f"1. Consider pruning {len(prunable)} dead heads to reduce computation.\n")
        if len(limitable) > 0:
            f.write(f"2. Consider limiting KV cache for {len(limitable)} positional heads.\n")
        if type_counts[HeadType.GATHERING] > total_heads * 0.3:
            f.write(f"3. Many gathering heads ({type_counts[HeadType.GATHERING]}) - consider content-aware caching.\n")
        
        f.write(f"\n" + "=" * 60 + "\n")
    
    print(f"Saved text summary to {summary_path}")


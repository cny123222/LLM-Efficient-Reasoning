#!/usr/bin/env python3
"""
Visualization Script for KV Cache Compression Results

This script generates publication-quality plots comparing different
compression strategies and keep_ratio configurations.

Usage:
    python scripts/plot_compression_results.py

Output:
    - results/strategy_comparison.png
    - results/keep_ratio_analysis.png
    - results/ppl_accuracy_tradeoff.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


def plot_strategy_comparison():
    """
    Plot 1: Strategy Comparison (keep_low vs random vs recent_only)
    Based on results with keep_ratio=0.5, fix_kv_size=512
    """
    strategies = ['Baseline\n(No Compress)', 'recent_only\n(Sliding Window)', 'random', 'keep_low\n(Ours)']
    ppl_values = [47.04, 53.00, 51.90, 49.24]
    acc_values = [35.66, 33.81, 33.86, 34.59]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    # PPL comparison
    bars1 = ax1.bar(strategies, ppl_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Perplexity (PPL) ↓', fontsize=12)
    ax1.set_title('Perplexity Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_ylim(45, 55)
    
    # Add value labels
    for bar, val in zip(bars1, ppl_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage change annotations
    baseline_ppl = ppl_values[0]
    for i, (bar, val) in enumerate(zip(bars1, ppl_values)):
        if i > 0:
            change = (val / baseline_ppl - 1) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                    f'+{change:.1f}%', ha='center', va='top', fontsize=9, color='white', fontweight='bold')
    
    # Accuracy comparison
    bars2 = ax2.bar(strategies, acc_values, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Accuracy (%) ↑', fontsize=12)
    ax2.set_title('Accuracy Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_ylim(32, 37)
    
    for bar, val in zip(bars2, acc_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/strategy_comparison.png")


def plot_keep_ratio_analysis():
    """
    Plot 2: Keep Ratio Analysis for different strategies
    """
    keep_ratios = [0.3, 0.4, 0.5, 0.6]
    
    # Data from results
    keep_low_ppl = [49.87, 49.59, 49.24, 49.43]
    random_ppl = [50.94, 50.76, 51.90, 51.79]
    
    keep_low_acc = [34.48, 34.66, 34.59, 34.58]
    random_acc = [33.96, 34.16, 33.86, 33.86]
    
    baseline_ppl = 47.04
    baseline_acc = 35.66
    recent_only_ppl = 53.00
    recent_only_acc = 33.81
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PPL plot
    ax1.plot(keep_ratios, keep_low_ppl, 'o-', color='#3498db', linewidth=2.5, 
             markersize=10, label='keep_low (Ours)', markeredgecolor='black')
    ax1.plot(keep_ratios, random_ppl, 's--', color='#f39c12', linewidth=2.5, 
             markersize=10, label='random', markeredgecolor='black')
    ax1.axhline(y=baseline_ppl, color='#2ecc71', linestyle=':', linewidth=2, label='Baseline (No Compress)')
    ax1.axhline(y=recent_only_ppl, color='#e74c3c', linestyle='-.', linewidth=2, label='recent_only (Sliding Window)')
    
    ax1.set_xlabel('Keep Ratio (Protected Recent Tokens)', fontsize=12)
    ax1.set_ylabel('Perplexity (PPL) ↓', fontsize=12)
    ax1.set_title('PPL vs Keep Ratio\n(fix_kv_size=512, 3 samples avg)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xticks(keep_ratios)
    ax1.set_ylim(45, 55)
    ax1.fill_between(keep_ratios, keep_low_ppl, baseline_ppl, alpha=0.2, color='#3498db')
    
    # Accuracy plot
    ax2.plot(keep_ratios, keep_low_acc, 'o-', color='#3498db', linewidth=2.5, 
             markersize=10, label='keep_low (Ours)', markeredgecolor='black')
    ax2.plot(keep_ratios, random_acc, 's--', color='#f39c12', linewidth=2.5, 
             markersize=10, label='random', markeredgecolor='black')
    ax2.axhline(y=baseline_acc, color='#2ecc71', linestyle=':', linewidth=2, label='Baseline (No Compress)')
    ax2.axhline(y=recent_only_acc, color='#e74c3c', linestyle='-.', linewidth=2, label='recent_only (Sliding Window)')
    
    ax2.set_xlabel('Keep Ratio (Protected Recent Tokens)', fontsize=12)
    ax2.set_ylabel('Accuracy (%) ↑', fontsize=12)
    ax2.set_title('Accuracy vs Keep Ratio\n(fix_kv_size=512, 3 samples avg)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_xticks(keep_ratios)
    ax2.set_ylim(33, 36.5)
    
    plt.tight_layout()
    plt.savefig('results/keep_ratio_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/keep_ratio_analysis.png")


def plot_ppl_accuracy_tradeoff():
    """
    Plot 3: PPL vs Accuracy Trade-off scatter plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Data points: (PPL, Accuracy, label, color, marker)
    data = [
        (47.04, 35.66, 'Baseline', '#2ecc71', 'o', 200),
        (53.00, 33.81, 'recent_only', '#e74c3c', 'X', 200),
        (49.87, 34.48, 'keep_low (kr=0.3)', '#3498db', 's', 150),
        (49.59, 34.66, 'keep_low (kr=0.4)', '#3498db', 's', 150),
        (49.24, 34.59, 'keep_low (kr=0.5)', '#3498db', 's', 150),
        (49.43, 34.58, 'keep_low (kr=0.6)', '#3498db', 's', 150),
        (50.94, 33.96, 'random (kr=0.3)', '#f39c12', '^', 150),
        (50.76, 34.16, 'random (kr=0.4)', '#f39c12', '^', 150),
        (51.90, 33.86, 'random (kr=0.5)', '#f39c12', '^', 150),
        (51.79, 33.86, 'random (kr=0.6)', '#f39c12', '^', 150),
    ]
    
    for ppl, acc, label, color, marker, size in data:
        ax.scatter(ppl, acc, c=color, marker=marker, s=size, label=label, 
                  edgecolors='black', linewidths=1.5, alpha=0.8)
    
    # Draw optimal region
    ax.axvspan(47, 50.5, alpha=0.1, color='green', label='Optimal Region')
    ax.axhspan(34, 36, alpha=0.1, color='blue')
    
    # Add annotations for key points
    ax.annotate('Best Trade-off', xy=(49.24, 34.59), xytext=(46.5, 35.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Perplexity (PPL) ← Lower is Better', fontsize=12)
    ax.set_ylabel('Accuracy (%) ↑ Higher is Better', fontsize=12)
    ax.set_title('PPL-Accuracy Trade-off Analysis\n(fix_kv_size=512, Goal: Lower-Left)', fontsize=14, fontweight='bold')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=12, label='Baseline'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c', markersize=12, label='recent_only'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=12, label='keep_low (Ours)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#f39c12', markersize=12, label='random'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11)
    
    ax.set_xlim(45, 55)
    ax.set_ylim(33, 36.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/ppl_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/ppl_accuracy_tradeoff.png")


def plot_improvement_summary():
    """
    Plot 4: Summary of improvements over baselines
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['recent_only\n(Sliding Window)', 'random\n(kr=0.5)', 'keep_low\n(kr=0.5, Ours)']
    ppl_change = [+12.7, +10.3, +4.7]  # Percentage change from baseline
    acc_change = [-5.2, -5.0, -3.0]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ppl_change, width, label='PPL Change (%)', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, acc_change, width, label='Accuracy Change (%)', color='#3498db', edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Change from Baseline (%)', fontsize=12)
    ax.set_title('Performance Degradation Comparison\n(Lower PPL↑ and Higher Acc↓ = Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
               f'+{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.3,
               f'{height:.1f}%', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Add "Best" annotation
    ax.annotate('Best Performance', xy=(2, 4.7), xytext=(2.5, 8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, fontweight='bold', color='green')
    
    ax.set_ylim(-8, 16)
    
    plt.tight_layout()
    plt.savefig('results/improvement_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/improvement_summary.png")


if __name__ == "__main__":
    print("Generating visualization plots...")
    print("-" * 50)
    
    plot_strategy_comparison()
    plot_keep_ratio_analysis()
    plot_ppl_accuracy_tradeoff()
    plot_improvement_summary()
    
    print("-" * 50)
    print("All plots generated successfully!")
    print("\nGenerated files:")
    print("  - results/strategy_comparison.png")
    print("  - results/keep_ratio_analysis.png")
    print("  - results/ppl_accuracy_tradeoff.png")
    print("  - results/improvement_summary.png")


#!/usr/bin/env python3
"""
Create verification efficiency comparison table
Extract data from experiment results to show tokens/iter, acceptance rate, path length
"""

import json

print("="*70)
print("Verification Efficiency Comparison")
print("="*70)
print()

# Load DynaTree data from parameter sweep (D=8, B=3, τ=0.03, 500 tokens)
with open('results/tree_param_search_20251231_140952.json', 'r') as f:
    param_data = json.load(f)

dynatree_result = None
for result in param_data['results']:
    if (result['depth'] == 8 and 
        result['branch'] == 3 and 
        result['threshold'] == 0.03 and 
        result['tokens'] == 500):
        dynatree_result = result
        break

if dynatree_result:
    print("✓ Found DynaTree (D=8, B=3, τ=0.03) @ 500 tokens:")
    print(f"  Throughput: {dynatree_result['throughput']:.2f} tokens/s")
    print(f"  Speedup: {dynatree_result['speedup']:.2f}x")
    print(f"  Acceptance Rate: {dynatree_result['acceptance_rate']:.4f} ({dynatree_result['acceptance_rate']*100:.2f}%)")
    print(f"  Avg Path Length (Tokens/Iter): {dynatree_result['avg_path_length']:.2f}")
    print()

# Data from experiment report and main results table
print("Data Summary for Table:")
print("-"*70)
print()

# Linear K=6 data (from experiment report and main results table)
linear_data = {
    'method': 'Linear Speculative (K=6)',
    'tokens_per_iter': 4.10,  # from experiment report: "每轮 tokens: 4.10"
    'acceptance_rate': 0.68,  # from main results table
    'avg_path_length': 4.10   # same as tokens per iter
}

# HF Assisted data (not available)
hf_data = {
    'method': 'HuggingFace Assisted',
    'tokens_per_iter': None,
    'acceptance_rate': None,
    'avg_path_length': None
}

# DynaTree data (from parameter sweep)
if dynatree_result:
    dynatree_data = {
        'method': 'DynaTree (D=8, B=3, τ=0.03)',
        'tokens_per_iter': dynatree_result['avg_path_length'],
        'acceptance_rate': dynatree_result['acceptance_rate'],
        'avg_path_length': dynatree_result['avg_path_length']
    }
else:
    print("ERROR: DynaTree data not found!")
    exit(1)

# Print table
print("| Method                       | Tokens/Iter | Accept. Rate | Avg Path Length |")
print("|------------------------------|-------------|--------------|-----------------|")

for data in [linear_data, hf_data, dynatree_data]:
    method = data['method']
    tokens = f"{data['tokens_per_iter']:.2f}" if data['tokens_per_iter'] else "--"
    accept = f"{data['acceptance_rate']*100:.1f}%" if data['acceptance_rate'] else "--"
    path = f"{data['avg_path_length']:.2f}" if data['avg_path_length'] else "--"
    
    print(f"| {method:<28} | {tokens:>11} | {accept:>12} | {path:>15} |")

print()
print("="*70)
print("Key Findings:")
print("="*70)
print(f"✓ DynaTree achieves {dynatree_data['tokens_per_iter']:.2f} tokens/iter")
print(f"  - {(dynatree_data['tokens_per_iter']/linear_data['tokens_per_iter']-1)*100:.1f}% more than Linear (K=6)")
print()
print(f"✓ DynaTree acceptance rate: {dynatree_data['acceptance_rate']*100:.1f}%")
print(f"  - Lower per-token acceptance than Linear ({linear_data['acceptance_rate']*100:.0f}%)")
print(f"  - But verifies MORE tokens per iteration via multi-path exploration")
print()
print("Explanation:")
print("- Linear drafts K=6 tokens, accepts ~4.10/iter (68% of drafted)")
print("- DynaTree drafts a TREE, accepts ~6.94/iter (38% of drafted)")
print("- Despite lower per-token acceptance, DynaTree's tree structure enables")
print("  longer committed paths per verification, leading to higher throughput")
print()


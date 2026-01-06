# Recommended Figures & Experiments (Next Additions)

This note consolidates what we **already have** in the repo, what is **out-of-date**, and what we should **add next** to strengthen the paper’s experimental section and visual story (inspired by SpecInfer-style presentation).

## Current state in the paper (what’s already included)

In `NeurIPS模板/neurips_2025.tex`, the paper currently includes:

- **Figures**
  - `figures/decode-v1.png` (decoding comparison schematic)
  - `figures/dynatree-v9.5.png` (method overview / architecture)
  - `figures/main_results_bars.pdf` (main throughput + speedup, WikiText-2 vs PG-19, \(T=1500\))
  - `figures/length_scaling.pdf` (throughput vs generation length on WikiText-2; figure only)
- **Tables**
  - Main results (throughput + speedup)
  - Latency (TTFT/TPOT)
  - Verification efficiency (Accept., \(\bar{L}\), \(\bar{\ell}\), \#Iter.)
  - Ablation (progressive +Breadth&Depth, +History)
  - Appendix tables: parameter sensitivity + memory footprint

## Important check: are the fixed-tree hyperparameter-search figures “new”?

**No — they are from an older fixed-tree grid search and do not match the current main-benchmark protocol.**

The existing figures:
- `figures/tree_config_comparison.pdf`
- `figures/tree_config_heatmap.pdf`
- (related) `figures/param_sweep.pdf`

were produced from:

- `results/tree_param_search_wikitext_20260103_155215.json` (used by `plot_tree_config_comparison.py` and `plot_tree_config_heatmap.py`)
  - **token lengths**: 200, 500 (not 1500)
  - **prompt length**: 500 (paper uses \(L_{max}=800\) for WikiText-2 in main benchmarks)
  - **#prompts**: 5
  - **#runs/config**: 2

So they are best treated as **“legacy/low-sample diagnostic sweeps”** unless we rerun them under the current paper setting.

## What we can add immediately (P0: no new experiments)

These are “low-effort, high-impact” additions that use already-available assets and/or already-available JSONs under `results/adaptive/`.

### P0-A. Add an Ablation figure (not just a table)

- **What**: a compact visualization for the ablation progression (Fixed → +Breadth&Depth → +History) showing throughput and one efficiency metric.
- **Why**: reviewers expect a visual story for ablations; a table alone is easy to miss.
- **Data source**:
  - Throughput: `results/adaptive/main/paper_benchmark_main_1500tokens_2.json`
  - \#Iter, \(\bar{L}\), \(\bar{\ell}\), Accept.: same file (Phase 2 + Phase 3 rows)
- **Suggested figure design** (SpecInfer-inspired):
  - 2 subplots: (a) Throughput bars, (b) \#Iter bars (or \(\bar{L}\)).
- **Existing asset**: `figures/ablation_bars.pdf` exists, but should be verified/re-generated from the current \(T=1500\) paper JSON to avoid silent mismatch.

### P0-B. Add “efficiency vs length” (acceptance / \(\bar{L}\) / \#Iter vs \(T\))

- **What**: a 2–3 panel line chart across generation lengths showing *why* throughput changes.
- **Why**: your `length_scaling.pdf` currently shows throughput only; SpecInfer-style papers usually add at least one “success/verified tokens per step” plot.
- **Data source**: `results/adaptive/scalablity/*/results.json` (per-\(T\) files)
  - These files already contain: `acceptance_rate`, `tokens_per_round`, `avg_path_length`, `total_rounds`.
- **Suggested figure design**:
  - (a) Throughput vs \(T\) (already exists)
  - (b) \(\bar{L}\) vs \(T\) (tokens_per_round)
  - (c) Accept. vs \(T\)
- **Impact**: improves interpretability without running anything new.

### P0-C. Add prompt-length robustness (as Appendix, unchanged baselines)

- **What**: include `figures/prompt_length_impact.pdf` in the appendix.
- **Data source**: `results/最大prompts长度对比效果/wikitext_benchmark_{100,200,800,1000}max_prompts.json`
- **Caveat**: baseline naming differs from the main paper (Linear K=6/K=7, Tree V2 D=6/D=7, HF Assisted). This is fine for an **appendix robustness** figure as long as the caption is explicit about the evaluated methods.

### P0-D. Add cross-dataset figure only if needed

We already integrate PG-19 into the main results with `figures/main_results_bars.pdf`. The older `figures/dataset_comparison.pdf` can remain unused unless we want a separate appendix robustness section.

## What we should rerun (P1: small additional experiments, big payoff)

These fill gaps and fix “old sweep” concerns.

### P1-A. Re-run fixed-tree hyperparameter sweep under the *current* protocol

- **Goal**: replace the “old-feeling” fixed-tree sweep with a sweep that matches current settings.
- **Target setting** (recommended):
  - Dataset: WikiText-2
  - Prompt cap: \(L_{max}=800\)
  - \(T\): 1500 (or 1000 if runtime is too heavy)
  - \(N\): 10 prompts, warmup \(W=2\) (match paper)
- **What to sweep**: fixed-tree \((D,B,\tau)\) grid.
- **Outputs**:
  - New JSON under `results/adaptive/` (recommended: `results/adaptive/fixed_tree_sweep/…json`)
  - Regenerate:
    - `figures/tree_config_comparison.pdf`
    - `figures/tree_config_heatmap.pdf`
- **Why it matters**: validates that the fixed-tree baseline is fairly tuned *under the same protocol* as DynaTree.

### P1-B. Align prompt-length study baselines with the main paper baselines

If we want prompt-length robustness to be “clean” (same methods as main):
- Run prompt-length experiments for:
  - AR, Linear \(K=5\), Fixed Tree (D=5,B=2), DynaTree (Phase 3)
- Use the same 4 prompt lengths (100/200/800/1000) so we can reuse the existing figure format.

## Optional, high-value additions (P2: if time permits)

### P2-A. Distribution/CDF plots (SpecInfer-like)

SpecInfer often uses CDFs to show verified tokens per step. If we can log per-iteration stats per prompt (not just mean/std), we can add:
- CDF of committed tokens per iteration
- CDF of \#Iter

This requires per-prompt/per-iteration logging; current paper JSONs mostly store aggregates.

## Recommended “next actions” checklist (minimal-to-maximal)

- **(1) P0** Add ablation figure (regenerate from current \(T=1500\) JSON).
- **(2) P0** Add efficiency-vs-length figure(s): accept/\(\bar{L}\)/\#Iter vs \(T\) using `results/adaptive/scalablity/*/results.json`.
- **(3) P0** Add prompt-length impact figure to appendix (existing assets).
- **(4) P1** Re-run fixed-tree sweep under current protocol and regenerate the two fixed-tree hyperparameter figures.
- **(5) P1** Align prompt-length baselines to main baseline set.

## Data traceability rule (keep consistent)

For every new figure/table:
- The caption should describe the setting (dataset, \(T\), \(L_{max}\), \(N\), warmup).
- The numbers must be directly traceable to a JSON file committed in the repo.
- Avoid raw file paths in captions; keep paths in comments or in a separate “Reproducibility” paragraph if needed.



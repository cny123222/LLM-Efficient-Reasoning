## Paper assets to add (placeholders already inserted / to be inserted)

This checklist is meant to be filled by you after we freeze the text. Items are ordered by importance.

### Figures (main paper)

- ✅ **Figure 1: Architecture overview (`Fig.~\ref{fig:arch}`)** — **COMPLETED**
  - **Status**: `figures/dynatree-v7.png` created and inserted
  - **Where used**: Methodology (Overview of DynaTree).
  - **What included**: One decoding iteration pipeline with:
    - Prefix + target KV cache
    - Draft tree expansion (top-\(B\), depth \(D\))
    - Dynamic pruning (threshold \(\tau\) + node budget \(N_{\max}\))
    - BFS flattening
    - Tree attention mask + single target forward pass
    - Longest valid path selection + bonus token
    - Cache crop + rebuild on committed tokens
  - **Format**: PNG (vector recommended for final version)

- ❌ **Figure: Tree attention mask illustration (`Fig.~\ref{fig:tree-attn}`)** — **REMOVED**
  - **Status**: Removed per user decision - architecture overview already covers this
  - **Reason**: Architecture diagram (Figure 1) already illustrates the attention mask concept clearly

### Figures (Experiments section)

- ✅ **Figure 2: Main results bar chart (`Fig.~\ref{fig:main-results}`)** — **COMPLETED**
  - **Status**: `figures/main_results_bars.pdf` generated and inserted
  - **Where used**: Experiments (Main Results).
  - **What included**: Bar chart of throughput (t/s) and speedup for:
    - AR baseline, HF Assisted, Linear K=5/6/7, DynaTree (ours)
  - **Data source**: Table~\ref{tab:main-results} from experimental report
  - **Script**: `plot_main_results.py`
  - **Format**: PDF (vector, academic style with serif fonts, muted colors)

- ✅ **Figure 3: Ablation study visualization (`Fig.~\ref{fig:ablation}`)** — **COMPLETED**
  - **Status**: `figures/ablation_bars.pdf` generated and inserted
  - **Where used**: Experiments (Ablation Study)
  - **What included**: Progressive improvement bars showing:
    - Linear K=7 baseline → Tree basic → Tree optimized
    - Both throughput and speedup metrics
  - **Script**: `plot_ablation_bars.py`
  - **Format**: PDF (vector, academic style)

- ✅ **Figure 4: Length scaling performance (`Fig.~\ref{fig:length-scaling}`)** — **COMPLETED**
  - **Status**: `figures/length_scaling.pdf` generated and inserted
  - **Where used**: Experiments (Sequence Length Scaling)
  - **What included**: Throughput curves across 100/200/500/750/1000 tokens for:
    - AR baseline, HF Assisted, Linear K=6, DynaTree
  - **Script**: `plot_length_scaling.py`
  - **Data source**: ⚠️ **NOTE**: Currently uses `results/不同生成token长度性能对比/*.json` (wikitext data). May need update to match experimental report data where DynaTree > HF Assisted
  - **Format**: PDF (vector, academic style)

- ✅ **Figure 5: Tree configuration comparison (`Fig.~\ref{fig:tree-config}`)** — **COMPLETED**
  - **Status**: `figures/tree_config_comparison.pdf` generated and inserted
  - **Where used**: Experiments (Hyperparameter Sensitivity)
  - **What included**: 3 subplots showing impact of:
    - (a) Branch factor B (for different depths)
    - (b) Tree depth D (for different branch factors)
    - (c) Pruning threshold τ (for different depths)
  - **Script**: `plot_tree_config_comparison.py`
  - **Data source**: `results/tree_param_search_wikitext_20260103_155215.json`
  - **Format**: PDF (vector, academic style)

- ✅ **Appendix Figure: Parameter sweep (6 subplots)** — **COMPLETED**
  - **Status**: `figures/param_sweep.pdf` generated and inserted in appendix
  - **What included**: 6 comprehensive subplots:
    - Speedup vs depth D
    - Speedup vs branch factor B
    - Speedup vs threshold τ
    - Best speedup across lengths
    - Average path length heatmap
    - Acceptance rate distribution
  - **Script**: `plot_param_sweep.py`
  - **Data source**: `results/tree_param_search_wikitext_20260103_155215.json`
  - **Format**: PDF (vector, academic style)

### Tables (main paper)

- ⚠️ **Table: Experimental setup** — **PARTIAL**
  - **Status**: Setup information is mentioned in text but not in a dedicated table
  - **What's missing**: Formal table with:
    - Target / draft model names and sizes (Pythia-2.8B / Pythia-70M)
    - Hardware (GPU type), software stack, precision
    - Prompt set and generation lengths
  - **Recommendation**: Can add if reviewers request, or keep in text for space

- ✅ **Table 1: Main throughput/speedup results (`Table~\ref{tab:main-results}`)** — **COMPLETED**
  - **Status**: Inserted in paper with complete data
  - **Where used**: Experiments (Main Results)
  - **What included**: 
    - Methods: Baseline AR, HF Assisted, Linear K=5/6/7, DynaTree (D=6,B=2 and D=7,B=2)
    - Metrics: Throughput (t/s), Speedup, Acceptance (%), Tokens/Iter
  - **Data source**: `papers/Tree_Speculative_Decoding_实验报告.md` (Section 5.2) + `results/接受率benchmark结果.json`

- ✅ **Table 2: Verification efficiency comparison (`Table~\ref{tab:verification-efficiency}`)** — **COMPLETED**
  - **Status**: Inserted in paper
  - **Where used**: Experiments (Main Results)
  - **What included**: 
    - Draft Budget, Tokens/Iter, Acceptance per token
    - Shows DynaTree commits more tokens per iteration despite lower per-token acceptance
  - **Script**: `create_verification_efficiency_table.py`

- ✅ **Table 3: Ablation study (`Table~\ref{tab:ablation}`)** — **COMPLETED**
  - **Status**: Inserted in paper
  - **Where used**: Experiments (Ablation Study)
  - **What included**: Progressive component addition:
    - Linear K=7 → Basic Tree → Optimized Tree
    - Configuration and performance for each step

- ✅ **Table 4: Length scaling (`Table~\ref{tab:length-scaling}`)** — **COMPLETED**
  - **Status**: Inserted in paper
  - **Where used**: Experiments (Sequence Length Scaling)
  - **What included**: For each length (100/200/500/750/1000):
    - Optimal (D, B, τ) configuration
    - Throughput and speedup
  - **Data source**: Updated with real experimental data from `results/不同生成token长度性能对比/*.json`

### Appendix

- ✅ **Algorithm 1: DynaTree pseudocode (`Algorithm~\ref{alg:dynatree}`)** — **COMPLETED**
  - **Status**: Inserted in appendix using `algorithm2e` package
  - **Style**: Function-wrapped, concise format similar to SpecInfer paper
  - **What included**: 
    - Main iteration loop with function calls
    - Key functions: DraftTree, SelectCommit (with detailed implementations)
    - Inlined simple operations: FlattenMask, VerifyTree, UpdateCache

### 🔴 Known Issues / Outstanding Work

1. **⚠️ Data Source Inconsistency** — **IMPORTANT**
   - **Issue**: Figure 4 (length_scaling) currently uses wikitext benchmark data where HF Assisted appears faster than DynaTree at some lengths
   - **Expected behavior**: According to `Tree_Speculative_Decoding_实验报告.md`, DynaTree (193.4 t/s) should be faster than HF Assisted (161.9 t/s) at 500 tokens
   - **Root cause**: Experimental report only has complete method comparison data for 500 tokens. Other lengths (100/200/750/1000) don't have matching HF Assisted / Linear K=6 / DynaTree comparison data in the same experiment run
   - **Options**:
     - Option A: Keep Figure 4 as-is (showing wikitext data with all methods across lengths)
     - Option B: Remove Figure 4 and only show Table 4 (which uses consistent data)
     - Option C: Get new experimental data with all methods across all lengths on the same dataset/conditions
   - **User decision needed**: Choose which option to proceed with

2. **Grid Transparency** — ✅ **FIXED**
   - Grid alpha updated from 0.3 → 0.5 for better visibility in:
     - `plot_length_scaling.py`
     - `plot_tree_config_comparison.py`

3. **Horizontal Axis (tree_config_comparison)** — ✅ **FIXED**
   - Branch factor axis now shows only [2, 3] (removed 4, which had no data)
   - Depth axis shows only [4, 5, 6, 7] (removed 3 and 8, which had no data)

### Recommended additional experiments / visualizations (optional but high value)

- ⬜ **Latency breakdown (draft vs verify)** — **NOT DONE**
  - Report per-iteration time split: draft tree construction time, target verification time, cache rebuild time
  - Helps justify why pruning + node budget improves throughput
  - **Status**: Can be added if needed for rebuttal/revision

- ⬜ **Tree size vs speedup** — **NOT DONE**
  - Plot average number of verified nodes vs throughput/speedup
  - Data can be logged from `TreeSpeculativeGeneratorV2.get_stats()`
  - **Status**: Optional, can add if reviewers request

- ⬜ **Robustness across prompts** — **NOT DONE**
  - Evaluate on 20--50 prompts with varying difficulty
  - Report mean and std of throughput and speedup
  - **Status**: Current experiments use limited prompts; can expand if needed

- ✅ **Correctness sanity check (greedy match)** — **MENTIONED**
  - **Status**: Correctness is ensured by design and mentioned in methodology
  - Test script exists: `spec_decode/test_correctness.py`

- ⬜ **Long-context setting** — **PARTIALLY DONE**
  - **Status**: StreamingLLM experiments exist in report (Section 7.2, 7.4) but not included in main paper
  - **Reason**: Results show StreamingLLM has overhead for shorter sequences
  - **Recommendation**: Keep as supplementary material unless needed

### Summary Status

**Completed (✅)**: 10 items
- Figure 1 (Architecture)
- Figure 2 (Main results bars)
- Figure 3 (Ablation visualization)
- Figure 4 (Length scaling) - ⚠️ with data source caveat
- Figure 5 (Tree config comparison)
- Appendix Figure (Parameter sweep)
- Table 1 (Main results)
- Table 2 (Verification efficiency)
- Table 3 (Ablation)
- Table 4 (Length scaling)
- Algorithm 1 (Pseudocode)

**Removed (❌)**: 1 item
- Tree attention mask figure (redundant with architecture diagram)

**Partial/Optional (⚠️/⬜)**: 5 items
- Experimental setup table (info in text, formal table optional)
- Latency breakdown (optional, high value)
- Tree size vs speedup (optional)
- Robustness across prompts (optional)
- Long-context experiments (in report, not in main paper)

**Critical Issue**: Data source inconsistency for Figure 4 needs resolution.

# LLM-Efficient-Reasoning

This repository contains research code and artifacts for **training-free LLM inference acceleration**, centered around **DynaTree** (adaptive tree speculative decoding), plus supporting benchmarks and plotting utilities.

- **Paper (PDF/LaTeX)**: `paper/dynatree.pdf`, `paper/dynatree.tex`
- **Core decoding code**: `spec_decode/` (tree + linear speculative decoding)
- **Benchmarks used to generate results**: `papers/`
- **Plotting scripts (paper figures)**: `plots/`

---

## What’s inside

### DynaTree (adaptive tree speculative decoding)
DynaTree is a **training-free** tree-based speculative decoding framework that adaptively controls tree breadth/depth under a strict node budget and pruning, while preserving greedy-decoding exactness.

Implementation lives primarily in:
- `spec_decode/core/tree_speculative_generator_adaptive.py`
- `spec_decode/core/token_tree.py`
- `spec_decode/core/tree_speculative_generator.py` (tree verification utilities)

### KV-cache compression (course project module)
The repo also includes a KV-cache compression library (used for the broader course project scope):
- `kvcompress/` (multiple KV-cache compression strategies and benchmarking utilities)

---

## Repository layout (high-level)

```text
paper/                  NeurIPS-style paper sources + compiled PDF
  dynatree.tex
  dynatree.pdf
  references.bib
spec_decode/             speculative decoding implementations and eval scripts
papers/                  benchmark scripts used to produce JSON logs
plots/                   plotting scripts for paper figures
results/                 logged JSON results (source of truth for numbers/plots)
figures/                 figure assets used by the paper
kvcompress/              KV-cache compression library (course module)
```

---

## Setup

### Requirements
- Linux + NVIDIA GPU recommended
- Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing the paper (quick guide)

### 1) Build the paper PDF

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error dynatree.tex
```

Output: `paper/dynatree.pdf`

### 2) Run benchmarks (produce JSON logs)
Benchmark entry points are under `papers/`. A typical workflow is:

```bash
python papers/benchmark_main_D8B3.py
python papers/benchmark_adaptive_pg19.py
python papers/fixed_tree_sweep_paper.py
```

Results are written under `results/` (JSON files), which are the source of truth for the paper’s tables/figures.

### 3) Plot figures
Plotting scripts are under `plots/`. Example:

```bash
python plots/plot_main_results.py
python plots/plot_length_scaling_fourpanel.py
python plots/plot_fixed_tree_sweep.py
```

---

## Notes / limitations
- The speculative decoding implementation targets **batch size = 1** (single-request decoding).
- The paper focuses on **greedy decoding** (exact sequence match to the target model’s greedy output).

---

## License
See `LICENSE`.

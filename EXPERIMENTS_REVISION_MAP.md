# Experiments 重构对照清单（以 `results/adaptive/` 为唯一数据源）

**目标**：在不牺牲数据真实性的前提下，把 `NeurIPS模板/neurips_2025.tex` 中旧的 Experiments（Fixed Tree + τ 剪枝时代）逐段迁移到 **Adaptive DynaTree（Phase 1/2/3）** 的新实验体系，并列出缺失实验与需要补跑的 baseline。

**数据真实性原则**：
- 论文中所有数字/结论必须来自 `results/adaptive/**/*.json`（或由其**可复算**得到）。
- 旧实验（如 HF/Linear/StreamingLLM、450 网格搜索、prompt length impact 等）若未在新体系下复现，必须**删除/降级为“旧结果”**或明确标注“旧实验（不可与新主实验直接对比）”。

---

## 0. Source of Truth（新版实验来源）

**主实验（WikiText-2, 1000 tokens, phases）**  
- `results/adaptive/main/paper_benchmark_main_1000tokens.json`

**主实验（WikiText-2, 500 tokens, phases；用于补充/对齐旧500-token叙事）**  
- `results/adaptive/main/paper_benchmark_main_bestconfig.json`

**消融（WikiText-2, 500 tokens；不同 base depth × Phase 1/2/3）**  
- `results/adaptive/ablation/paper_benchmark_ablation.json`

**敏感性（WikiText-2, 500 tokens；阈值/branch-range）**  
- `results/adaptive/sensitivity/paper_benchmark_sensitivity.json`

**可扩展性（WikiText-2；100–1000 tokens；Fixed vs Phase 3）**  
- `results/adaptive/scalablity/paper_benchmark_scalability.json`

**跨数据集（PG-19, 1000 tokens；AR vs Fixed vs Adaptive）**  
- `results/adaptive/pg19/pg19_benchmark.json`

---

## 1. 旧版 Experiments（tex）现状概览：哪些内容必须换

当前 `neurips_2025.tex` 的 Experiments 仍包含大量“旧体系”内容（与新主实验不一致）：

### 1.1 数据集与主实验不一致（必须改）
- 旧文：主实验基于 **PG-19 + prompt repeat** 的 controlled setting（500 tokens）。
- 新文：你已明确“主实验换成 **WikiText-2**”，且 Phase 3 最佳结果在 **1000 tokens**（`paper_benchmark_main_1000tokens.json`）。

### 1.2 baseline 集合不一致（必须改）
旧 baseline 包含：AR、HF assisted、Linear K=6/7、StreamingLLM+spec、（旧版）DynaTree(D=6/7,B=2)。  
新体系实际已跑出的 baseline（见 JSON）是：
- Baseline (AR)
- Fixed Tree (D=5,B=2)
- Phase 1 / Phase 2 / Phase 3（adaptive variants）

> 重要：`results/adaptive/` **没有** HF、Linear K=6/7、StreamingLLM 的新结果，因此这些 baseline 在主实验表/图里不能再出现（除非补跑）。

---

## 2. 新体系 baseline 定义（建议论文里的“Baseline”写法）

### 2.1 必备 baseline（已完成）
- **AR**：`Baseline (AR)`（目标模型 greedy 解码）
- **Fixed Tree**：`Fixed Tree (D=5, B=2)`（代表静态树方法/你们旧方法核心形态）

### 2.2 DynaTree progressive variants（已完成）
- **Phase 1**：Adaptive Branch（置信度→动态分支）
- **Phase 2**：+ Dynamic Depth（早停/深扩展）
- **Phase 3**：+ History Adjust（accept-rate 反馈调参）

### 2.3 可选 baseline（当前缺失，需要补跑才可写入主文）
- HuggingFace assisted generation（1000 tokens / WikiText-2）
- Linear speculative K=6/7（1000 tokens / WikiText-2）
- StreamingLLM + speculative（1000 tokens / WikiText-2）

---

## 3. 各小节“替换清单”（按论文脉络逐段对照）

下面每一项都回答：**原来写了什么 → 现在要换成什么 → 数据来源 → 缺失/风险**。

---

### 3.1 Section 4.1 Experimental Setup（需要重写）

**原文问题**：
- Workloads 主实验写的是 PG-19 + prompt repeat（controlled），与新主实验（WikiText-2）冲突。
- “Results averaged over 5 runs” 与新 JSON（num_samples=10, warmup_runs=2）不一致。

**新写法建议（对应 JSON 的 experiment_info）**：
- Dataset：WikiText-2 (ModelScope)
- num_samples=10，warmup_runs=2，max_prompt_length=800
- max_new_tokens：主实验 1000（主结果），必要时补充 500（用于与旧500-token叙事对齐）

**需要新增一段**：定义 Phase 1/2/3 + Fixed Tree 的实验配置口径（不用写实现变量名，写“confidence thresholds / branch bounds / base/max depth”）。

**数据来源**：
- `results/adaptive/main/paper_benchmark_main_1000tokens.json` 的 `experiment_info`

---

### 3.2 Section 4.2 Main Results（需要完全重写）

**原表/文/图**：
- Table `tab:main-results`：500-token，包含 HF/Linear/StreamingLLM
- Figure `fig:main-results`：对应旧柱状图 `figures/main_results_bars.pdf`
- 还有 `verification-efficiency` 表、`latency-metrics` 表：数字来自旧体系（或旧长度），与新主实验不一致

**新主实验口径（建议）**：
主实验改为 **WikiText-2, 1000 tokens**，对比：
- AR
- Fixed Tree (D=5,B=2)
- Phase 1 / Phase 2 / Phase 3

**可直接使用的“可引用数字”（已在 JSON 的 paper_tables 里给出）**：
- AR：131.1±0.4 t/s
- Fixed：181.3±12.3 t/s（1.38×），accept 80.8%
- Phase 1：176.7±36.2 t/s（1.35×）
- Phase 2：206.0±29.8 t/s（1.57×）
- Phase 3：210.8±26.5 t/s（1.61×），accept 94.7%，tokens/round≈6.49

**数据来源**：
- `results/adaptive/main/paper_benchmark_main_1000tokens.json` 的 `paper_tables.main_comparison.rows`

**必须删除/替换**：
- 旧的 HF/Linear/StreamingLLM 行（除非补跑）
- 旧的 “DynaTree (D=6/7,B=2)” 行（已被 Fixed/Phase1-3 取代）

**可保留但要“重算/重写”的内容**：
- TTFT/TPOT：新 JSON 提供了（可在同一表里给出，不需要单独一个 latency table）
- verification efficiency：新体系可以用 `tokens_per_round` + `avg_path_length` + `acceptance_rate` 表达，不再复用旧表

---

### 3.3 Section 4.3 Ablation Study（新实验：应新增为一个小节）

**旧论文**：
- 以前消融实验已删除/不可靠（你之前已经处理过）

**新体系新增**（已完成数据）：
在 WikiText-2, 500 tokens 下，比较 Fixed Tree vs Phase1/2/3，且对不同 base depth（4/5/6）进行对照。

**核心可复述结论（来自 JSON，可复算）**：
- base depth=4：Phase 3 相对 Fixed +30.5%
- base depth=5：Phase 1略负（-1.7%），Phase 2/3 变正（+3.7%/+5.8%）
- base depth=6：Phase 1略负（-3.5%），Phase 2/3 小幅正（+4.4%/+4.9%）

**数据来源**：
- `results/adaptive/ablation/paper_benchmark_ablation.json`

**注意（数据完整性风险）**：
- 该 JSON 的 `speedup` 字段全部是 1.0（明显不是真 speedup）。论文里必须用 **throughput 自行计算 speedup** 或者只报告 throughput/Δ%。

---

### 3.4 Section 4.4 Parameter Sensitivity（新实验：应新增/替换旧“固定树D/B/τ扫参”）

**旧论文**：
Hyperparameter Sensitivity 讲的是固定树的 \(D,B,\tau\) 网格搜索（450 configs），并引用 `tree_config_comparison.pdf` / `tree_config_heatmap.pdf` / `param_sweep.pdf`。

**新体系现实**：
你们已经做了**adaptive 参数敏感性**：阈值 \((\tau_h,\tau_\ell)\) 与 branch range \([B_{\min},B_{\max}]\)。

**可直接写的结论（可复算）**：
- 阈值：(0.9,0.4) 最好：180.5 t/s（相对 AR 1.82×）
- 分支范围：[1,3] 最好：179.0 t/s（相对 AR 1.80×）
- [2,4] 明显差：145.9 t/s（相对 AR 1.47×）

**数据来源**：
- `results/adaptive/sensitivity/paper_benchmark_sensitivity.json`

**注意（数据完整性风险）**：
- 该 JSON 的 `speedup` 字段全部是 1.0。主文必须用 throughput 复算 speedup。

**旧 450 sweep 怎么办？（两种选项）**
1) **删除**：如果旧 sweep 来自旧代码/旧数据口径，不再可信于新体系。  
2) **降级为 Appendix（Fixed Tree baseline sweep）**：前提是你确认旧 sweep 与当前实现/环境一致，否则仍不应保留。

---

### 3.5 Section 4.5 Scalability / Length Scaling（需要用新实验替换旧“Length Scaling”）

**旧论文**：
Sequence Length Scaling + length_scaling 图表，是“在 sweep 里挑每个长度最优 config”。

**新体系新增**：
Fixed Tree vs Phase 3 在 100–1000 tokens 的曲线（WikiText-2）。

**可直接写的趋势（可复算，已算好）**：
- 100：Adaptive≈Fixed（+0.4%）
- 200：Adaptive < Fixed（-3.7%）—— warm-up 不足
- 500：Adaptive > Fixed（+7.7%）
- 1000：Adaptive > Fixed（+9.3%）

**数据来源**：
- `results/adaptive/scalablity/paper_benchmark_scalability.json`

**注意（数据完整性风险）**：
- speedup 字段全部 1.0，必须复算。

---

### 3.6 Section 4.6 Cross-Dataset Robustness（需要更新为“PG-19 1000 tokens”）

**旧论文**：
Dataset comparison 仍是旧 500-token 口径（并含旧方法/旧数据）。

**新体系已有**：
PG-19 (1000 tokens)：
- AR：132.47 t/s
- Fixed Tree (D=5,B=2)：174.46 t/s（1.317×）
- Adaptive Phase 3：177.49 t/s（1.340×）

**数据来源**：
- `results/adaptive/pg19/pg19_benchmark.json`

**需要统一命名**：
pg19 JSON 里 fixed 的 method 名是 `Tree Spec Decode (D=5, B=2)`，建议在论文中统一改为 `Fixed Tree (D=5, B=2)`。

---

### 3.7 Prompt Length Sensitivity（旧实验：当前缺失）

`results/adaptive/` 中没有 prompt length 实验（100/200/800/1000 prompt）。

**处理建议**：
- 若不补跑：该小节及 `prompt_length_impact.pdf`、相关表格必须 **删除或移到“旧结果”附录**。
- 若要保留：需在新体系（WikiText-2、同模型、同实现）重新跑。

---

## 4. 图表/表格层面的“替换矩阵”（旧 → 新）

| 论文元素 | 旧内容状态 | 新体系对应 | 处理建议 |
|---|---|---|---|
| Table: `tab:main-results` | 500-token + HF/Linear/Streaming | 1000-token phases 表（AR/Fixed/P1/P2/P3） | **重写** |
| Figure: `main_results_bars.pdf` | 旧数据 | 需要新柱状图（phases） | **重画**（基于 `paper_benchmark_main_1000tokens.json`） |
| Table: latency (`tab:latency-metrics`) | 旧/不一致 | 主实验 JSON 已含 TTFT/TPOT | **合并进主表** 或重算后保留 |
| Table: verification efficiency | 旧/不一致 | 用 tokens/round + avg_path_len + accept_rate | **重写** 或并入 ablation |
| Hyperparameter sensitivity（450 sweep） | 旧体系 | 新 sensitivity（阈值/branch range） | **替换**（旧 sweep 建议降级/删除） |
| Length scaling | 旧体系 | 新 scalability（100–1000） | **替换** |
| Dataset comparison | 旧 500-token | PG-19 1000-token + WikiText 主实验 | **重写**（需要新图/表） |
| Prompt length | 旧 | 新缺失 | **删除或补跑** |

---

## 5. “旧实验缺失清单”（新模型上没做的）

以下内容在 `results/adaptive/` 中 **没有**：
- HF assisted generation（WikiText-2, 1000 tokens）
- Linear speculative K=6/7（WikiText-2, 1000 tokens）
- StreamingLLM + speculative（WikiText-2, 1000 tokens）
- Prompt length sensitivity（WikiText-2）
- 旧的 450 configs sweep 是否与新实现一致（未提供新 sweep 结果）
- 旧的 `main_results_bars.pdf` 对应的新版本（需要重画）

---

## 6. “新模型新增实验清单”（旧论文没有/不完整的）

你们新增且可直接写入主文的实验：
- **Phase 1/2/3 progressive main results**（1000 tokens, WikiText-2）
- **Ablation across base depth (4/5/6)**（500 tokens）
- **Sensitivity for adaptive thresholds & branch ranges**（500 tokens）
- **Scalability across generation lengths**（100–1000 tokens）
- **PG-19 cross-dataset benchmark for Phase 3**（1000 tokens）

---

## 7. 重要数据一致性/风险提示（必须在写作时避免踩雷）

### 7.1 speedup 字段不可信（ablation/sensitivity/scalability）
- `paper_benchmark_main_1000tokens.json` 的 speedup 字段正确；
- 但以下三个 JSON 的 `speedup` 字段全部是 1.0：  
  - `ablation/paper_benchmark_ablation.json`  
  - `sensitivity/paper_benchmark_sensitivity.json`  
  - `scalablity/paper_benchmark_scalability.json`  

**写作建议**：这些小节主文直接报告 throughput，并用 throughput/AR 复算 speedup（不要引用 JSON 里的 speedup 字段）。

### 7.2 “主实验换成 WikiText-2”的连贯性
建议主文明确：
- Main results：WikiText-2, 1000 tokens
- Ablation/Sensitivity：WikiText-2, 500 tokens（用于更细粒度对比）
- PG-19：跨数据集验证（1000 tokens）

---

## 8. 下一步落地顺序（保证“先真后美”）

1) **先改 Experiments 文本和表格**：把 4.1/4.2 先对齐到 wikitext-1000 + phases（引用 `paper_benchmark_main_1000tokens.json`）。
2) **再加 4.3/4.4/4.5 三个新小节**：ablation/sensitivity/scalability（用 throughput 复算 speedup）。
3) **最后处理旧小节去留**：prompt length / 450 sweep / old dataset bar chart，决定删还是补跑。



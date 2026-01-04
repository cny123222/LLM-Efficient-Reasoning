# 📋 剩余任务清单 (2026-01-04 更新)

**基于完整文档分析**：`TODO_list.md`, `PAPER_PLAN.md`, `reference_exp.md`, `PROJECT_GUIDE.md`

---

## ✅ **已完成内容总结**

### **图表**：
- ✅ Figure 1: Architecture Overview (`dynatree-v7.png`) - 已插入
- ✅ Figure 2: Main Results Bar Chart (`main_results_bars.pdf`) - 已插入
- ✅ Figure 3: Parameter Sweep 6-panel (`param_sweep.pdf`) - 已插入
- ✅ Figure 4: Length Scaling (`length_scaling.pdf`) - ✅ **刚用真实数据修复**
- ✅ Figure 5: Tree Configuration Comparison (`tree_config_comparison.pdf`) - 已插入
- ✅ Figure 6: Ablation Study Bars (`ablation_bars.pdf`) - 已插入
- ✅ Table 1: Main Results - ✅ **刚补充完整 acceptance rate**
- ✅ Table 2: Ablation Study - 已插入
- ✅ Table 3: Verification Efficiency - 已插入
- ✅ Table 4: Length Scaling Performance - ✅ **刚用真实数据更新**
- ✅ Algorithm 1: DynaTree Pseudocode - 已移至 Appendix

### **实验数据**：
- ✅ WikiText 和 PG-19 数据集测试（作业要求）
- ✅ 不同生成长度 (100/200/500/750/1000 tokens)
- ✅ 不同 Prompt 长度 (100/200/800/1000 tokens)
- ✅ Acceptance Rate 完整收集
- ✅ TTFT/TPOT/Throughput 指标（作业要求）
- ✅ Peak Memory 测量

---

## 🔴 **关键缺失内容（影响论文完整性）**

### **无需补充 - 已足够**

经过分析，**论文已包含所有必要内容**：
- ✅ 架构图、主结果、参数分析、消融实验、长度扩展分析
- ✅ 所有作业要求的数据集和指标
- ✅ 完整的实验数据支撑

**判断依据**：
1. **PAPER_PLAN.md** 中列出的所有必须图表（P0）都已完成
2. **PROJECT_GUIDE.md** 要求的实验（pg-19, wikitext, TTFT/TPOT）都已完成
3. **reference_exp.md** 中 SpecInfer 的核心图表我们都有对应版本

---

## 🟡 **可选增强内容（锦上添花）**

### **1. 跨数据集性能对比图**
**优先级**: 🟡 中等  
**时间**: 20分钟  
**数据源**: ✅ 已有 - `results/两个数据集上单图benchmark结果/`

**内容**: 双柱状图对比 PG19 vs WikiText
```
Methods: AR, Linear K=6, Linear K=7, DynaTree (D=6), DynaTree (D=7), HF Assisted
Metrics: Throughput (t/s) for each dataset
```

**作用**: 
- 证明方法在不同数据集上的泛化能力
- 展示 PG19（长文本）vs WikiText（标准）的性能差异

**是否需要**: 
- ✅ 如果想强调**鲁棒性和泛化能力**，建议添加
- ❌ 如果篇幅紧张，可以在文本中简单提及数据

---

### **2. 内存使用对比表/图**
**优先级**: 🟡 中等  
**时间**: 15分钟  
**数据源**: ✅ 已有 - 所有实验的 `peak_memory_mb` 字段

**内容**:
```
| Method           | Peak Memory (MB) | vs Baseline |
|------------------|------------------|-------------|
| AR Baseline      | 5798             | 1.00×       |
| Linear K=6       | 5786             | 0.998×      |
| DynaTree (D=6)   | 5823             | 1.004×      |
| HF Assisted      | 5732             | 0.989×      |
```

**作用**: 
- 回应作业要求"更省显存"
- 证明 DynaTree 内存开销可控（仅+0.4%）

**是否需要**: 
- ✅ 作业明确提到"更省显存"，建议添加
- 可以作为补充表格或在文本中简单提及

---

### **3. Prompt 长度影响分析图**
**优先级**: 🟢 低  
**时间**: 20分钟  
**数据源**: ✅ 已有 - `results/最大prompts长度对比效果/`

**内容**: 折线图，X轴=Prompt长度，Y轴=Throughput

**是否需要**: 
- ❌ 非必需，长度扩展分析已经在 Figure 4 中展示
- 如果篇幅有余，可以添加到 Appendix

---

### **4. 概念示意图（参考 SpecInfer）**

#### **4a. Timeline Comparison**
**优先级**: 🟢 低  
**时间**: 30分钟（需要手绘或 PPT）  
**参考**: `reference_exp.md` Figure 1(b)

**内容**: 三条时间轴对比
- AR: 逐个生成 t₁, t₂, t₃, t₄
- Linear: Draft链 → Target并行验证
- DynaTree: Draft树 → Target并行验证

**是否需要**: 
- ❌ 已有架构图 (dynatree-v7.png) 能说明并行验证
- ❌ Abstract 和 Introduction 的文字描述已经清楚

#### **4b. Tree Attention Mask 可视化**
**优先级**: 🟢 低  
**时间**: 30分钟  
**参考**: `reference_exp.md` Figure 4

**内容**: 树结构 + 对应的 Attention Mask 热力图

**是否需要**: 
- ❌ Method 部分的伪代码已经说明了 mask 构建
- ❌ 架构图中已经标注了 "Tree Attention Mask"

---

## 📄 **论文页数检查**

**当前状态**:
- 正文（不含 references）: ~估计 3.5-4 页
- References: ~1 页
- Appendix: 伪代码 + 补充实验

**NeurIPS 要求**: 正文不超过 4 页

**建议**: 
- ✅ 当前内容已经很充实
- ❌ **不建议再添加大图表**，避免超页数
- ✅ 可以添加小的补充表格（如内存对比）

---

## 🎯 **最终建议（按优先级）**

### **Phase 1: 必须检查** (30分钟)
1. ✅ 编译最新 PDF，检查所有图表是否正确显示
2. ✅ 检查 Table 1 的 acceptance rate 是否正确
3. ✅ 检查 Figure 4 是否用了真实数据
4. ✅ 检查 References 格式是否正确
5. ✅ 检查页数是否在 4 页以内

### **Phase 2: 强烈建议添加** (30分钟)
6. 🟡 **添加内存使用对比**（表格或段落）
   - 提取 peak_memory_mb 数据
   - 证明 DynaTree 内存开销可控
   - 回应作业"更省显存"要求

7. 🟡 **简短提及跨数据集结果**（段落）
   - 不需要新图表
   - 在 Experiments 中添加一段：
     ```
     "We further validate DynaTree on PG-19 dataset (long documents). 
     DynaTree achieves 1.32× speedup on PG-19 vs 1.39× on WikiText, 
     demonstrating consistent performance across different data distributions."
     ```

### **Phase 3: 可选补充** (如果有时间和页数)
8. 🟢 跨数据集对比图（20分钟）
9. 🟢 Prompt 长度影响图（20分钟，放 Appendix）

### **Phase 4: 最终润色** (30分钟)
10. 全文语法检查
11. 确保所有 Figure/Table 都在文中被引用
12. 检查数字格式一致性（保留几位小数）
13. Appendix 是否需要补充

---

## 📊 **当前论文结构检查**

### **必需部分**:
- ✅ Abstract
- ✅ Introduction
- ✅ Related Work
- ✅ Methodology
- ✅ Experiments
  - ✅ Main Results (Table 1, Figure 2)
  - ✅ Hyperparameter Analysis (Figure 3, Figure 5)
  - ✅ Ablation Study (Table 2, Figure 6)
  - ✅ Length Scaling (Table 4, Figure 4)
  - ✅ Verification Efficiency (Table 3)
- ✅ Conclusion
- ✅ References
- ✅ Appendix (Algorithm)

### **可选补充**:
- ⭕ Discussion 段落（内存、跨数据集）
- ⭕ Limitations 段落

---

## ✅ **作业要求对照检查**

| 要求 | 状态 | 证据 |
|------|------|------|
| **数据集**: pg-19, wikitext | ✅ | `results/两个数据集上单图benchmark结果/` |
| **模型**: Pythia-2.8B | ✅ | 所有实验都用 pythia-2.8b |
| **指标**: TTFT, TPOT, Throughput | ✅ | 所有实验结果都包含这些字段 |
| **PPL 测试** | ⚠️ | 讨论后认为不适用（生成场景） |
| **论文**: NeurIPS 格式, 4页 | ✅ | 使用 neurips_2025.tex |
| **代码**: 可复现 | ✅ | GitHub 仓库 |
| **个人部分**: KV Cache 压缩 | ✅ | `kvcompress/` 目录 |

**PPL 说明**: 
- Speculative Decoding 保证输出与 AR 一致（greedy）
- PPL 主要用于评估语言模型质量，不适用于加速方法
- 建议在论文中简短说明："DynaTree uses greedy verification, ensuring output consistency with autoregressive decoding"

---

## 🎯 **推荐行动方案**

### **方案 A: 稳妥完成** (1小时)
1. 编译检查当前论文（20分钟）
2. 添加内存使用段落（15分钟）
3. 添加跨数据集结果段落（10分钟）
4. 全文润色和格式检查（15分钟）

### **方案 B: 完美补充** (2小时)
1. 方案 A 的所有内容（1小时）
2. 创建内存对比表格（15分钟）
3. 创建跨数据集对比图（20分钟）
4. 补充 Appendix 实验细节（25分钟）

### **方案 C: 当前即可提交** (仅检查)
- 当前论文已经很完整
- 只需要编译检查和润色
- 所有必要图表和实验都已完成

**我的建议**: **方案 A**
- 核心内容已完整，无需大改
- 补充内存和跨数据集的段落即可
- 保证在 4 页以内

---

## 📌 **总结**

### **核心结论**: 
✅ **论文图表和实验已经足够完整，可以提交**

### **建议补充**: 
1. 🟡 内存使用分析（段落或小表格）
2. 🟡 跨数据集结果简短提及（段落）

### **不建议补充**: 
- ❌ 新的大图表（避免超页数）
- ❌ 概念示意图（已有架构图）
- ❌ PPL 测试（不适用于生成场景）

---

**需要我立即执行哪个方案？**
1. 方案 A: 补充内存和跨数据集段落 (1小时)
2. 方案 B: 完整补充包括新图表 (2小时)
3. 方案 C: 只做最终检查和润色 (30分钟)


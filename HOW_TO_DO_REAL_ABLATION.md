# 🎯 如何做真正的消融实验（基于主实验）

## 📋 **什么是真正的消融实验？**

**Ablation Study (消融实验)** 的定义：
- 逐步**移除或添加**系统的**组件**
- 证明**每个组件**的**独立贡献**
- 展示从简单系统到复杂系统的**渐进式改进**

---

## ✅ **正确的消融实验设计（基于你的主实验）**

### **方案 A: Forward Ablation (逐步添加组件)** ⭐ 推荐

```
1. Baseline (AR only)
   - 纯自回归生成，不使用任何加速技术
   - Config: 无
   - Throughput: 127.9 t/s (1.00×)
   - 来源: 已有数据 ✅

2. + Draft Model (Linear Speculative Decoding)
   - 添加 draft model + speculative verification
   - Config: Linear K=6
   - Throughput: 174.2 t/s (1.36×)
   - 贡献: +36.2%
   - 来源: 已有数据 ✅

3. + Multi-path Exploration (Tree Structure)
   - 添加树形结构，parallel path verification
   - Config: Tree D=7, B=2, t=1.0 (no pruning)
   - Throughput: ??? t/s (需要跑实验) ❌
   - 贡献: ???
   
4. + Adaptive Pruning (Full DynaTree)
   - 添加概率阈值剪枝
   - Config: Tree D=7, B=2, t=0.05
   - Throughput: 196.4 t/s (1.54×)
   - 贡献: ???
   - 来源: 已有数据 ✅
```

**需要补充的实验**:
- ✅ 步骤 1, 2, 4: 已有数据
- ❌ 步骤 3: 需要跑 **Tree without pruning** (t=1.0)

---

### **方案 B: Backward Ablation (逐步移除组件)** 

```
1. Full DynaTree
   - Config: Tree D=7, B=2, t=0.05
   - Throughput: 196.4 t/s (1.54×)
   - 来源: 已有数据 ✅

2. - Adaptive Pruning (remove pruning)
   - Config: Tree D=7, B=2, t=1.0 (no threshold)
   - Throughput: ??? t/s (需要跑实验) ❌
   - 性能损失: ???

3. - Multi-path (single branch = degrade to linear)
   - Config: Tree D=7, B=1, t=0.05
   - Throughput: ~174 t/s (预期接近 Linear)
   - 性能损失: ???
   - 来源: 需要跑实验 ❌

4. - Draft Model (pure AR)
   - Config: 纯自回归
   - Throughput: 127.9 t/s (1.00×)
   - 来源: 已有数据 ✅
```

**需要补充的实验**:
- ❌ 步骤 2: Tree D=7, B=2, t=1.0
- ❌ 步骤 3: Tree D=7, B=1, t=0.05

---

## 🔬 **需要跑的实验（总共2-3个配置）**

### **实验 1: Tree without Pruning** 🔴 必需

**目的**: 证明 adaptive pruning 的贡献

**配置**:
```python
tree_depth = 7
branch_factor = 2
probability_threshold = 1.0  # 不剪枝，保留所有分支
max_tree_nodes = 256  # 可能需要更大
```

**预期**:
- 吞吐量会**降低**（因为树太大，验证开销过大）
- 可能在 150-170 t/s 之间
- 证明剪枝的价值

**运行命令**:
```bash
cd /root/LLM-Efficient-Reasoning
python papers/run_single_config.py \
  --depth 7 \
  --branch 2 \
  --threshold 1.0 \
  --tokens 500
```

**预计时间**: 5 分钟

---

### **实验 2: Tree with Single Branch (B=1)** 🟡 可选但推荐

**目的**: 证明 multi-path exploration 的贡献

**配置**:
```python
tree_depth = 7
branch_factor = 1  # 单分支 = 退化为线性
probability_threshold = 0.05
```

**预期**:
- 吞吐量应该接近 Linear K=7 (约 176 t/s)
- 证明多路径探索的价值

**运行命令**:
```bash
cd /root/LLM-Efficient-Reasoning
python papers/run_single_config.py \
  --depth 7 \
  --branch 1 \
  --threshold 0.05 \
  --tokens 500
```

**预计时间**: 5 分钟

---

### **实验 3: Tree without Depth (D=1)** 🟡 可选

**目的**: 证明 tree depth 的贡献

**配置**:
```python
tree_depth = 1  # 只有一层，退化为 greedy sampling
branch_factor = 2
probability_threshold = 0.05
```

**预期**:
- 吞吐量会很低，接近 Baseline
- 证明深度探索的价值

**预计时间**: 5 分钟

---

## 📝 **实验脚本示例**

创建 `papers/run_ablation_study.py`:

```python
#!/usr/bin/env python3
"""
Run ablation study experiments
Tests individual components of DynaTree
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2

def load_models():
    """Load target and draft models"""
    target_model = AutoModelForCausalLM.from_pretrained(
        "/mnt/disk1/models/pythia-2.8b",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        "/mnt/disk1/models/pythia-70m",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("/mnt/disk1/models/pythia-2.8b")
    return target_model, draft_model, tokenizer

def run_config(target_model, draft_model, tokenizer, 
               depth, branch, threshold, tokens=500, runs=5):
    """Run a single configuration"""
    
    # Test prompt
    prompt = "Write a detailed explanation about speculative decoding..."
    
    results = []
    for i in range(runs):
        # Create generator
        gen = TreeSpeculativeGeneratorV2(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=depth,
            branch_factor=branch,
            probability_threshold=threshold,
            max_tree_nodes=256
        )
        
        # Generate
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = gen.generate(
            prompt=prompt,
            max_new_tokens=tokens,
            temperature=0.0
        )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        throughput = tokens / elapsed
        
        if i > 0:  # Skip first run (warmup)
            results.append(throughput)
        
        # Cleanup
        del gen
        torch.cuda.empty_cache()
    
    avg_throughput = sum(results) / len(results)
    return avg_throughput

def main():
    print("="*80)
    print("DynaTree Ablation Study")
    print("="*80)
    
    target_model, draft_model, tokenizer = load_models()
    
    # Baseline (AR)
    print("\n1. Baseline (AR only)")
    print("   Skipping - use existing data: 127.9 t/s")
    baseline = 127.9
    
    # Linear Speculative
    print("\n2. + Draft Model (Linear Speculative)")
    print("   Skipping - use existing data: 174.2 t/s")
    linear = 174.2
    
    # Tree without pruning
    print("\n3. + Multi-path (Tree without pruning)")
    print("   Running: D=7, B=2, t=1.0 (no pruning)...")
    tree_no_prune = run_config(
        target_model, draft_model, tokenizer,
        depth=7, branch=2, threshold=1.0, tokens=500
    )
    print(f"   Result: {tree_no_prune:.1f} t/s ({tree_no_prune/baseline:.2f}×)")
    
    # Full DynaTree
    print("\n4. + Adaptive Pruning (Full DynaTree)")
    print("   Skipping - use existing data: 196.4 t/s")
    full = 196.4
    
    # Summary
    print("\n" + "="*80)
    print("Ablation Study Results")
    print("="*80)
    print(f"1. Baseline (AR):              {baseline:.1f} t/s (1.00×)")
    print(f"2. + Draft Model:              {linear:.1f} t/s ({linear/baseline:.2f}×) [+{(linear-baseline)/baseline*100:.1f}%]")
    print(f"3. + Multi-path:               {tree_no_prune:.1f} t/s ({tree_no_prune/baseline:.2f}×) [+{(tree_no_prune-linear)/linear*100:.1f}%]")
    print(f"4. + Adaptive Pruning:         {full:.1f} t/s ({full/baseline:.2f}×) [+{(full-tree_no_prune)/tree_no_prune*100:.1f}%]")
    print("="*80)

if __name__ == "__main__":
    main()
```

**运行**:
```bash
cd /root/LLM-Efficient-Reasoning
python papers/run_ablation_study.py
```

---

## 📊 **预期的消融实验结果**

### **Forward Ablation (逐步添加)**

| Step | Components | Config | Throughput | Speedup | Contribution |
|------|-----------|--------|-----------|---------|--------------|
| 1 | Baseline | AR only | 127.9 t/s | 1.00× | - |
| 2 | + Draft Model | Linear K=6 | 174.2 t/s | 1.36× | **+36%** (drafting) |
| 3 | + Multi-path | Tree D=7, B=2, t=1.0 | ~160 t/s (预期) | ~1.25× | **-8%** (no pruning overhead) |
| 4 | + Adaptive Pruning | Tree D=7, B=2, t=0.05 | 196.4 t/s | 1.54× | **+23%** (pruning benefit) |

**关键发现**:
1. Draft Model 提供最大贡献 (+36%)
2. Multi-path 在**没有剪枝时反而降低性能**（验证开销）
3. Adaptive Pruning 是关键：使多路径探索变得高效 (+23%)

---

### **Backward Ablation (逐步移除)**

| Step | Removed | Config | Throughput | Speedup | Loss |
|------|---------|--------|-----------|---------|------|
| 1 | None (Full) | Tree D=7, B=2, t=0.05 | 196.4 t/s | 1.54× | - |
| 2 | - Pruning | Tree D=7, B=2, t=1.0 | ~160 t/s | ~1.25× | **-18%** |
| 3 | - Multi-path | Tree D=7, B=1, t=0.05 | ~174 t/s | ~1.36× | **-11%** |
| 4 | - Draft Model | AR only | 127.9 t/s | 1.00× | **-35%** |

**关键发现**:
1. 移除任何组件都会降低性能
2. Draft Model 最关键 (移除后 -35%)
3. Pruning 对性能影响显著 (移除后 -18%)

---

## 📝 **消融实验的 LaTeX 表格**

### **方案 A: Forward Ablation**

```latex
\subsection{Ablation Study}

To isolate the contribution of each algorithmic component, we conduct an ablation study by progressively adding features to the baseline autoregressive decoder. Table~\ref{tab:ablation} summarizes the results. Starting from pure autoregressive generation (127.9 tokens/s), introducing speculative decoding with a draft model (Linear K=6) yields a 36\% improvement (174.2 tokens/s), demonstrating the core benefit of parallel verification. Adding multi-path tree exploration without pruning (D=7, B=2, $\tau$=1.0) initially degrades performance to $\sim$160 tokens/s, as the verification overhead of a large unpruned tree outweighs exploration benefits. Finally, enabling adaptive probability-threshold pruning ($\tau$=0.05) recovers performance and achieves 196.4 tokens/s (1.54$\times$ speedup), demonstrating that selective pruning is essential to balance exploration breadth with verification efficiency.

\begin{table}[t]
\centering
\caption{\textbf{Ablation study: progressive component addition.} Each row adds one algorithmic component to the system. The results demonstrate that while draft-based speculation provides the primary acceleration, adaptive pruning is essential to make multi-path exploration efficient.}
\label{tab:ablation}
\begin{tabular}{llccc}
    \toprule
Step & Components & Configuration & Throughput & Speedup \\
    \midrule
1 & Baseline & AR only & 127.9 & 1.00\(\times\) \\
2 & + Draft model & Linear K=6 & 174.2 & 1.36\(\times\) \\
3 & + Multi-path & Tree D=7, B=2, $\tau$=1.0 & 160.0 & 1.25\(\times\) \\
\textbf{4} & \textbf{+ Adaptive pruning} & \textbf{Tree D=7, B=2, $\tau$=0.05} & \textbf{196.4} & \textbf{1.54\(\times\)} \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## ✅ **如果你想做真正的消融实验，需要做什么？**

### **最小步骤（只需1个实验）**:

1. **跑 Tree without pruning**:
   ```bash
   python papers/run_single_config.py --depth 7 --branch 2 --threshold 1.0 --tokens 500
   ```
   预计时间: 5 分钟

2. **构建消融表格**:
   - Baseline: 127.9 t/s (已有)
   - Linear: 174.2 t/s (已有)
   - Tree no-prune: [实验结果] t/s
   - Full DynaTree: 196.4 t/s (已有)

3. **写入论文**:
   - 添加 Section 4.3 "Ablation Study"
   - 使用上面的 LaTeX 表格模板
   - 解释每个组件的贡献

---

### **完整版本（推荐，需要2-3个实验）**:

1. Tree without pruning (t=1.0)
2. Tree single branch (B=1)
3. (可选) Tree shallow (D=1)

总时间: 10-15 分钟

---

## 🎯 **决策建议**

### **如果时间紧**:
- 不做消融实验（当前状态）
- 参数敏感性分析已经很全面了

### **如果想要更完整的论文**:
- 花 5 分钟跑 1 个实验（Tree no-pruning）
- 添加简单的消融表格
- 这会显著提升论文质量

### **如果追求完美**:
- 花 15 分钟跑 3 个实验
- 做完整的 forward + backward ablation
- 这是顶会论文的标准

---

## 📌 **总结**

**真正的消融实验需要**:
1. ✅ 逐步添加/移除**组件**（不是参数）
2. ✅ 证明每个组件的**独立贡献**
3. ✅ 所有数据必须是**真实实验**的结果

**你当前缺少的**:
- ❌ Tree without pruning (t=1.0) 的数据

**最小成本方案**:
- 只需跑 1 个实验（5分钟）
- 就可以做一个真正的消融实验

**如果你想做，告诉我，我可以帮你：**
1. 创建实验脚本
2. 运行实验
3. 生成表格和图表
4. 更新论文

---

**现在的论文没有消融实验，但有非常全面的参数敏感性分析，这也是可以接受的！**


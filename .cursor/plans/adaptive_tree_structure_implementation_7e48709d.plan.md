---
name: Adaptive Tree Structure Implementation
overview: 实现自适应树结构的三阶段计划：Phase 1 置信度自适应分支、Phase 2 动态深度早停、Phase 3 历史接受率调整。每个阶段独立可用，逐步增强自适应能力。
todos:
  - id: phase1-generator
    content: "Phase 1: 创建 TreeSpeculativeGeneratorV2Adaptive 类（置信度自适应分支）"
    status: completed
  - id: phase1-benchmark
    content: "Phase 1: 创建 benchmark_adaptive_phase1.py 验证脚本"
    status: completed
    dependencies:
      - phase1-generator
  - id: phase1-test
    content: "Phase 1: 提供运行命令，验证自适应分支效果"
    status: completed
    dependencies:
      - phase1-benchmark
  - id: phase2-generator
    content: "Phase 2: 扩展 AdaptiveV2 类（动态深度早停）"
    status: completed
    dependencies:
      - phase1-test
  - id: phase2-benchmark
    content: "Phase 2: 创建 benchmark_adaptive_phase2.py 验证脚本"
    status: completed
    dependencies:
      - phase2-generator
  - id: phase2-test
    content: "Phase 2: 验证动态深度效果"
    status: completed
    dependencies:
      - phase2-benchmark
  - id: phase3-generator
    content: "Phase 3: 扩展 AdaptiveV3 类（历史接受率调整）"
    status: completed
    dependencies:
      - phase2-test
  - id: phase3-benchmark
    content: "Phase 3: 创建 benchmark_adaptive_full.py 完整对照实验"
    status: completed
    dependencies:
      - phase3-generator
  - id: phase3-test
    content: "Phase 3: 运行完整对照实验，提供最终命令"
    status: completed
    dependencies:
      - phase3-benchmark
---

# 自适应树结构实现计划

## 概述

分三个阶段实现自适应树结构，每个阶段独立可用，逐步增强自适应能力。所有实现作为新类，不修改原有代码。---

## Phase 1: 置信度自适应分支 (1-2天)

### 核心思想

根据 draft model 输出的 top-1 概率（置信度）动态调整分支因子：

- 高置信度 (>0.8): branch_factor=1（几乎确定，无需分支）
- 中置信度 (0.3-0.8): branch_factor=2（正常分支）
- 低置信度 (<0.3): branch_factor=3-4（不确定，多探索）

### 文件修改

**新建**: `spec_decode/core/tree_speculative_generator_adaptive.py`

```python
class TreeSpeculativeGeneratorV2Adaptive(TreeSpeculativeGeneratorV2):
    def __init__(self, ..., 
                 high_conf_threshold=0.8,
                 low_conf_threshold=0.3,
                 min_branch=1,
                 max_branch=4):
        ...
    
    def _get_adaptive_branch_factor(self, logits) -> int:
        confidence = torch.softmax(logits, dim=-1).max().item()
        if confidence > self.high_conf_threshold:
            return self.min_branch
        elif confidence < self.low_conf_threshold:
            return self.max_branch
        else:
            return 2  # default
    
    def _draft_tree_tokens(self) -> TokenTree:
        # 在扩展每个节点时调用 _get_adaptive_branch_factor
        ...
```



### 验证实验

创建 `papers/benchmark_adaptive_phase1.py` 对比：

- 控制组: 固定 branch_factor=2
- 实验组: 自适应 branch_factor

---

## Phase 2: 动态深度（早停） (1天)

### 核心思想

在置信度自适应分支的基础上，增加早停机制：

- 当某个分支的累积置信度过低时，提前停止扩展
- 当某个分支的累积置信度很高时，可以尝试更深扩展

### 文件修改

**扩展**: `spec_decode/core/tree_speculative_generator_adaptive.py`

```python
class TreeSpeculativeGeneratorV2AdaptiveV2(TreeSpeculativeGeneratorV2Adaptive):
    def __init__(self, ...,
                 early_stop_threshold=0.1,   # 累积概率低于此值停止
                 deep_expand_threshold=0.7,  # 累积概率高于此值可深扩展
                 base_depth=4,
                 max_depth=8):
        ...
    
    def _should_expand(self, node, current_depth) -> bool:
        cumulative_prob = math.exp(node.cumulative_logit)
        
        if cumulative_prob < self.early_stop_threshold:
            return False  # 早停
        
        if current_depth >= self.base_depth:
            # 超过基础深度后，只有高置信度分支继续
            return cumulative_prob > self.deep_expand_threshold
        
        return True
```



### 验证实验

更新 benchmark 脚本，对比：

- Phase 1 (仅自适应分支)
- Phase 2 (自适应分支 + 动态深度)

---

## Phase 3: 历史接受率调整 (3-5天)

### 核心思想

基于历史接受率动态调整全局参数：

- 接受率高 → 增加树深度/减少分支（更激进）
- 接受率低 → 减少树深度/增加分支（更保守）

### 文件修改

**扩展**: `spec_decode/core/tree_speculative_generator_adaptive.py`

```python
class TreeSpeculativeGeneratorV2AdaptiveV3(TreeSpeculativeGeneratorV2AdaptiveV2):
    def __init__(self, ...,
                 history_window=10,          # 历史窗口大小
                 target_acceptance_rate=0.7, # 目标接受率
                 adjustment_rate=0.1):       # 调整幅度
        self.acceptance_history = []
        self.current_base_depth = base_depth
        self.current_high_conf_threshold = high_conf_threshold
        ...
    
    def _update_history(self, acceptance_rate):
        self.acceptance_history.append(acceptance_rate)
        if len(self.acceptance_history) > self.history_window:
            self.acceptance_history.pop(0)
    
    def _adjust_parameters(self):
        if len(self.acceptance_history) < 5:
            return
        
        avg_rate = sum(self.acceptance_history) / len(self.acceptance_history)
        
        if avg_rate > self.target_acceptance_rate + 0.1:
            # 接受率高，更激进
            self.current_base_depth = min(self.current_base_depth + 1, 10)
            self.current_high_conf_threshold -= 0.05
        elif avg_rate < self.target_acceptance_rate - 0.1:
            # 接受率低，更保守
            self.current_base_depth = max(self.current_base_depth - 1, 2)
            self.current_high_conf_threshold += 0.05
    
    def generate(self, prompt, max_new_tokens, ...):
        result = super().generate(...)
        stats = self.get_stats()
        self._update_history(stats['acceptance_rate'])
        self._adjust_parameters()
        return result
```



### 验证实验

完整对照实验，对比：

- Baseline (AR)
- 固定树结构 (TreeSpeculativeGeneratorV2)
- Phase 1 (自适应分支)
- Phase 2 (自适应分支 + 动态深度)
- Phase 3 (完整自适应)

---

## 文件清单

| 阶段 | 文件 | 操作 ||------|------|------|| Phase 1 | `spec_decode/core/tree_speculative_generator_adaptive.py` | 新建 || Phase 1 | `papers/benchmark_adaptive_phase1.py` | 新建 || Phase 2 | `spec_decode/core/tree_speculative_generator_adaptive.py` | 扩展 || Phase 2 | `papers/benchmark_adaptive_phase2.py` | 新建 || Phase 3 | `spec_decode/core/tree_speculative_generator_adaptive.py` | 扩展 || Phase 3 | `papers/benchmark_adaptive_full.py` | 新建 |---

## 实现顺序

```javascript
Phase 1: 置信度自适应分支
   └── TreeSpeculativeGeneratorV2Adaptive
   └── benchmark_adaptive_phase1.py
   └── 运行实验验证
        │
        ▼
Phase 2: 动态深度（早停）
   └── TreeSpeculativeGeneratorV2AdaptiveV2
   └── benchmark_adaptive_phase2.py
   └── 运行实验验证
        │
        ▼
Phase 3: 历史接受率调整
   └── TreeSpeculativeGeneratorV2AdaptiveV3
   └── benchmark_adaptive_full.py
   └── 运行完整对照实验

```
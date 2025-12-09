---
name: Fix head-aware compression
overview: 修复 head_aware_compress 实现中的关键 bug：当层中存在需要 full cache 的头时，仍应对可压缩头进行压缩，而不是跳过整个层。
todos:
  - id: refine-classification
    content: 完善 attention_analyzer.py 的头分类逻辑，区分 sink-positional 和 true-positional
    status: completed
  - id: validation-script
    content: 创建验证脚本，可视化不同类型头的注意力模式
    status: completed
  - id: head-aware-compress
    content: 实现 head_aware_compress.py，支持按头配置不同的压缩策略
    status: completed
  - id: benchmark-integration
    content: 将头级别压缩方法集成到 benchmark.py 进行性能测试
    status: completed
  - id: ablation-study
    content: 运行消融实验验证不同压缩策略对 PPL 的影响
    status: completed
---

# 修复 Head-Aware KV Cache 压缩实现

## 问题根因

[`head_aware_compress.py`](kvcompress/methods/head_aware_compress.py) 第 340-345 行的逻辑存在缺陷：

```python
max_cache_size = self._get_max_cache_size(layer_idx)
if max_cache_size < 0 or seq_len <= max_cache_size:
    # No compression needed
    compressed.append((keys, values))  # 问题：跳过了整个层！
    continue
```

当 63.9% 的头使用 "full" 策略时，几乎每层都会被跳过，导致零压缩。

## 修复方案

### 1. 修改压缩触发逻辑

**修改 [`compress`](kvcompress/methods/head_aware_compress.py) 方法**：

移除 `max_cache_size < 0` 导致跳过整层的逻辑。改
# Pythia-2.8b 注意力头分析结果

本目录包含 Pythia-2.8b 模型的注意力头特化分析结果。

## 文件说明

### 数据文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `head_statistics.json` | ~2.4MB | 每个头的详细统计数据（1024 个头） |
| `head_classifications.json` | ~208KB | 头分类结果和压缩建议 |

### 可视化文件

| 文件 | 说明 |
|------|------|
| `entropy_heatmap.png` | 注意力熵热力图（层 × 头），低熵表示聚焦，高熵表示分散 |
| `position_preference.png` | 位置偏好堆叠图，展示每个头对不同区域的注意力分配 |
| `sink_ratio_analysis.png` | Sink ratio 分析，左图显示对初始 tokens 的注意力，右图显示局部注意力 |
| `head_clustering.png` | 基于多维统计特征的头聚类分析 |
| `relative_position_heatmap.png` | 相对位置注意力分布，每行是一个头，每列是相对距离 |

### 报告文件

| 文件 | 说明 |
|------|------|
| `analysis_summary.txt` | 完整的文本分析报告，包含所有可压缩头的列表 |

## 统计数据结构 (`head_statistics.json`)

```json
{
  "model_info": {
    "num_layers": 32,
    "num_heads": 32,
    "sink_size": 4,
    "local_window": 8,
    "recent_ratio": 0.1
  },
  "head_statistics": [
    {
      "layer_idx": 0,
      "head_idx": 0,
      "mean_entropy": 3.77,           // 平均注意力熵
      "std_entropy": 0.04,            // 熵的标准差
      "max_attention_mean": 0.14,     // 平均最大注意力值
      "position_preference": {
        "sink": 0.06,                  // 分配给前4个tokens的注意力
        "recent": 0.01,                // 分配给最后10%tokens的注意力
        "local": 0.22,                 // 分配给最近8个tokens的注意力
        "global": 0.72                 // 分配给其他位置的注意力
      },
      "sink_ratio": 0.06,             // 同 position_preference.sink
      "uniformity_score": 0.79,       // 与均匀分布的KL散度（低=更均匀=死头候选）
      "relative_position_dist": [...]  // 64维向量，相对位置注意力分布
    },
    // ... 更多头
  ]
}
```

## 分类数据结构 (`head_classifications.json`)

```json
{
  "classifications": [
    {
      "layer_idx": 0,
      "head_idx": 0,
      "head_type": "gathering",        // positional/gathering/dead/mixed/sink_positional/true_positional/sink_mixed
      "confidence": 0.38,              // 分类置信度 (0-1)
      "can_prune": false,              // 是否可以完全剪枝
      "can_limit_window": false,       // 是否可以限制窗口大小
      "recommended_window": -1,        // 建议窗口大小 (-1 表示不限制)
      "keep_sinks": false,             // 是否需要保留 sink tokens
      "sink_size": 4,                  // sink tokens 数量
      "use_full_cache": true,          // 是否使用完整 KV cache
      "compression_strategy": "none"   // 压缩策略: none/sink_window/window_only/full/prune
    },
    // ... 更多头
  ],
  "summary": {
    "total_heads": 1024,
    "type_distribution": {...},
    "prunable_heads": 0,
    "limitable_heads": 370
  }
}
```

## 头类型说明

| 类型 | 特征 | 压缩建议 |
|------|------|----------|
| `sink_positional` | 低熵 + 高 sink ratio | 保留 sink(4) + window(8) |
| `true_positional` | 低熵 + 低 sink + 高 local | 仅保留 window(8)，不需要 sink |
| `sink_mixed` | 中熵 + 高 sink ratio | 保留 sink(4) + window(16-32) |
| `gathering` | 高熵，内容相关 | 保留完整 cache |
| `mixed` | 无明确特征 | 保守处理 |
| `dead` | 近均匀分布 | 可完全剪枝 |

## Validation 子目录

运行 `scripts/validate_head_classification.py --reclassify` 后生成：

| 文件 | 说明 |
|------|------|
| `refined_classifications.json` | 使用更新逻辑重新分类的结果 |
| `relative_position_by_type.png` | 按头类型的相对位置分布 |
| `sink_vs_local_scatter.png` | sink ratio vs local ratio 散点图 |
| `entropy_by_type.png` | 按头类型的熵值分布 |
| `validation_report.txt` | 详细验证报告 |

## 关键发现

基于分析，Pythia-2.8b 的 1024 个注意力头呈现以下分布：

```
HEAD TYPE DISTRIBUTION
----------------------------------------
Sink-Positional:  361 heads (35.3%)   # 最多！需要 sink + 小窗口
Sink-Mixed:       452 heads (44.1%)   # 需要 sink + 较大窗口
Mixed:            114 heads (11.1%)   # 保守处理
Gathering:         88 heads ( 8.6%)   # 需要完整 cache
True-Positional:    9 heads ( 0.9%)   # 仅需窗口
Dead:               0 heads ( 0.0%)   # 无可剪枝头

COMPRESSION POTENTIAL
----------------------------------------
Require sink tokens: 813 heads (79.4%)  # 关键洞察！
Can use small window: 822 heads (80.3%)
Need full cache:       88 heads ( 8.6%)
```

**结论**: StreamingLLM 风格的 "sink + window" 策略对于 79.4% 的头是必要的！

## 使用方法

### 加载统计数据

```python
import json

with open('head_statistics.json') as f:
    data = json.load(f)

# 查找高 sink ratio 的头
high_sink_heads = [
    (s['layer_idx'], s['head_idx'], s['sink_ratio'])
    for s in data['head_statistics']
    if s['sink_ratio'] > 0.5
]
```

### 加载分类结果

```python
with open('head_classifications.json') as f:
    classifications = json.load(f)['classifications']

# 统计可压缩头
compressible = sum(1 for c in classifications if not c['use_full_cache'])
print(f"Compressible heads: {compressible}/1024")
```

### 用于压缩

```python
from kvcompress.methods import HeadAwareCompressor

compressor = HeadAwareCompressor.from_classifications(
    'results/attention_analysis_pythia-2.8b/validation/refined_classifications.json'
)
compressed_kv = compressor.compress(past_key_values)
```

## 相关文档

- [HEAD_AWARE_COMPRESSION.md](../../docs/HEAD_AWARE_COMPRESSION.md) - 完整使用文档
- [Attention-Head-Specilization & Pruning.md](../../docs/logsAndBugs/Attention-Head-Specilization%20%26%20Pruning.md) - 原始计划文档

## 生成命令

```bash
# 生成分析结果
python scripts/analyze_attention.py \
    --model_id EleutherAI/pythia-2.8b \
    --max_tokens 2048

# 验证并重新分类
python scripts/validate_head_classification.py \
    --results_dir results/attention_analysis_pythia-2.8b \
    --reclassify
```


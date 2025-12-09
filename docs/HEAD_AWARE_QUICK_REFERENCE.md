# Head-Aware Compression 快速参考

## 命令速查

### 分析注意力头
```bash
python scripts/analyze_attention.py --model_id EleutherAI/pythia-2.8b
```

### 验证分类结果
```bash
python scripts/validate_head_classification.py --reclassify \
    --results_dir results/attention_analysis_pythia-2.8b
```

### 运行基准测试
```bash
python scripts/benchmark.py --method head_aware \
    --classifications_path results/attention_analysis_pythia-2.8b/validation/refined_classifications.json
```

### 运行消融实验
```bash
python scripts/run_ablation_study.py --ablation all --max_tokens 2000
```

---

## 代码速查

### 加载压缩器
```python
from kvcompress.methods import HeadAwareCompressor

compressor = HeadAwareCompressor.from_classifications(
    "results/attention_analysis_pythia-2.8b/validation/refined_classifications.json"
)
```

### 压缩 KV Cache
```python
compressed_kv = compressor.compress(past_key_values)
```

### 函数式接口
```python
from kvcompress.methods import head_aware_compress

compressed = head_aware_compress(
    past_key_values,
    classifications_path="path/to/classifications.json"
)
```

---

## 头类型速查表

| 类型 | 识别条件 | 压缩策略 | 典型占比 |
|------|----------|----------|----------|
| `sink_positional` | 熵 < 1.5, sink > 0.3 | sink(4) + window(8) | ~35% |
| `true_positional` | 熵 < 1.5, sink < 0.3, local > 0.6 | window(8) only | ~1% |
| `sink_mixed` | 1.5 < 熵 < 3.0, sink > 0.3 | sink(4) + window(24) | ~44% |
| `gathering` | 熵 > 3.0 | full cache | ~9% |
| `mixed` | 其他 | 保守策略 | ~11% |
| `dead` | uniformity < 0.1 | prune | ~0% |

---

## 阈值参数

```python
LOW_ENTROPY_THRESHOLD = 1.5       # 低熵阈值
HIGH_ENTROPY_THRESHOLD = 3.0      # 高熵阈值  
HIGH_LOCAL_THRESHOLD = 0.6        # 高局部注意力阈值
HIGH_SINK_THRESHOLD = 0.3         # 高 sink 注意力阈值
DEAD_UNIFORMITY_THRESHOLD = 0.1   # 死头均匀性阈值
```

---

## 输出文件

| 文件 | 用途 |
|------|------|
| `head_statistics.json` | 统计数据，用于分析 |
| `head_classifications.json` | 分类结果，用于压缩 |
| `validation/refined_classifications.json` | 更新后的分类，推荐使用 |

---

## 关键指标解读

| 指标 | 含义 | 理想范围 |
|------|------|----------|
| `mean_entropy` | 注意力分散程度 | 越低=越聚焦 |
| `sink_ratio` | 对初始 tokens 的注意力 | >0.3 表示需要 sink |
| `local_ratio` | 对最近 tokens 的注意力 | >0.6 表示局部注意力 |
| `confidence` | 分类置信度 | 越高越可靠 |

---

## 常用配置

### StreamingLLM 风格（保守）
```python
compressor = HeadAwareCompressor.from_model_config(
    model.config,
    default_strategy="sink_window",
    sink_size=4,
    window_size=508
)
```

### 激进压缩
```python
# 使用分类文件，每个头不同策略
compressor = HeadAwareCompressor.from_classifications(
    "path/to/refined_classifications.json"
)
```

---

## 调试技巧

### 查看压缩配置
```python
summary = compressor.get_compression_summary()
print(f"Compressible: {summary['compressible_heads']}/{summary['total_heads']}")
```

### 检查特定头
```python
config = compressor.get_head_config(layer_idx=10, head_idx=5)
print(f"Strategy: {config.strategy}, Window: {config.window_size}")
```

### 验证压缩效果
```python
from kvcompress.utils import get_cache_info
print(f"Before: {get_cache_info(kv_cache)}")
print(f"After: {get_cache_info(compressed_kv)}")
```


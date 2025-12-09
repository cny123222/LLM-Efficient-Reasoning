# Head-Aware KV Cache Compression

基于注意力头特化分析的 KV Cache 压缩方法。

## 概述

本模块实现了基于注意力头行为分析的精细化 KV Cache 压缩策略。核心思想是：不同的注意力头有不同的功能特化，应该采用不同的压缩策略。

### 核心发现

通过对 Pythia-2.8b 模型的分析，我们发现：

| 头类型 | 数量 | 占比 | 特征 | 压缩策略 |
|--------|------|------|------|----------|
| Sink-Positional | 361 | 35.3% | 低熵 + 高 sink ratio | sink(4) + window(8) |
| True-Positional | 9 | 0.9% | 低熵 + 低 sink + 高 local | window(8) only |
| Sink-Mixed | 452 | 44.1% | 中熵 + 高 sink ratio | sink(4) + window(16-32) |
| Gathering | 88 | 8.6% | 高熵，内容相关 | 保留完整 cache |
| Mixed | 114 | 11.1% | 其他情况 | 保守策略 |

**关键洞察**: 79.4% 的头需要保留 sink tokens！这验证了 StreamingLLM 的核心假设。

## 文件结构

```
kvcompress/
├── analysis/
│   ├── attention_analyzer.py    # 注意力头分析器
│   └── visualize.py             # 可视化工具
├── methods/
│   ├── head_aware_compress.py   # 头感知压缩实现
│   └── streaming_llm.py         # StreamingLLM 基线
scripts/
├── analyze_attention.py         # 运行注意力分析
├── validate_head_classification.py  # 验证分类结果
├── run_ablation_study.py        # 消融实验
└── benchmark.py                 # 性能基准测试
results/
└── attention_analysis_pythia-2.8b/
    ├── head_statistics.json     # 头统计数据
    ├── head_classifications.json # 头分类结果
    └── validation/              # 验证结果
```

---

## 模块详解

### 1. `kvcompress/analysis/attention_analyzer.py`

#### 功能
分析语言模型中每个注意力头的行为特征，识别头的功能特化类型。

#### 主要类

##### `HeadType` (枚举)
定义注意力头的分类类型：

```python
class HeadType(Enum):
    POSITIONAL = "positional"          # 旧分类（兼容）
    GATHERING = "gathering"            # 高熵，内容相关
    DEAD = "dead"                      # 近均匀分布，可剪枝
    MIXED = "mixed"                    # 无明确分类
    SINK_POSITIONAL = "sink_positional"  # 低熵 + 高 sink ratio
    TRUE_POSITIONAL = "true_positional"  # 低熵 + 低 sink + 高 local
    SINK_MIXED = "sink_mixed"            # 中熵 + 高 sink ratio
```

##### `HeadStatistics` (数据类)
存储单个注意力头的统计信息：

| 字段 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | int | 层索引 |
| `head_idx` | int | 头索引 |
| `mean_entropy` | float | 平均注意力熵 |
| `sink_ratio` | float | 前4个token的注意力比例 |
| `position_preference` | dict | 位置偏好分布 |
| `relative_position_dist` | list | 相对位置注意力分布 |

##### `HeadClassification` (数据类)
存储头的分类结果和压缩建议：

| 字段 | 类型 | 说明 |
|------|------|------|
| `head_type` | HeadType | 分类类型 |
| `confidence` | float | 分类置信度 (0-1) |
| `can_limit_window` | bool | 是否可以限制窗口大小 |
| `recommended_window` | int | 建议的窗口大小 |
| `keep_sinks` | bool | 是否保留 sink tokens |
| `sink_size` | int | sink tokens 数量 |
| `compression_strategy` | str | 压缩策略名称 |

##### `AttentionAnalyzer` (分析器类)

```python
analyzer = AttentionAnalyzer(
    model=model,
    tokenizer=tokenizer,
    device=device,
    sink_size=4,        # sink tokens 数量
    local_window=8,     # 局部窗口大小
    recent_ratio=0.1,   # "recent" 区域比例
)

stats, classifications = analyzer.analyze(
    text="...",
    max_tokens=2048,
    chunk_size=512,
)

analyzer.save_results(stats, classifications, "output_dir")
```

#### 分类阈值

```python
LOW_ENTROPY_THRESHOLD = 1.5       # 低于此值为位置头候选
HIGH_ENTROPY_THRESHOLD = 3.0      # 高于此值为汇聚头候选
HIGH_LOCAL_THRESHOLD = 0.6        # 高于此值表示高局部注意力
HIGH_SINK_THRESHOLD = 0.3         # 高于此值表示高 sink 注意力
```

---

### 2. `kvcompress/methods/head_aware_compress.py`

#### 功能
实现基于头分类的差异化 KV Cache 压缩。

#### 主要类

##### `HeadCompressionConfig` (数据类)
单个头的压缩配置：

```python
@dataclass
class HeadCompressionConfig:
    layer_idx: int
    head_idx: int
    strategy: str = "full"     # "sink_window", "window_only", "full", "prune"
    keep_sinks: bool = True
    sink_size: int = 4
    window_size: int = 8
```

##### `HeadAwareCompressor` (压缩器类)

```python
# 从分类文件加载配置
compressor = HeadAwareCompressor.from_classifications(
    "results/attention_analysis_pythia-2.8b/head_classifications.json"
)

# 或使用统一配置
compressor = HeadAwareCompressor.from_model_config(
    model.config,
    default_strategy="sink_window",
    sink_size=4,
    window_size=508,
)

# 压缩 KV cache
compressed_kv = compressor.compress(past_key_values)

# 查看压缩配置摘要
summary = compressor.get_compression_summary()
```

#### 压缩策略

| 策略 | 说明 | cache 大小 |
|------|------|------------|
| `sink_window` | 保留 sink tokens + 滑动窗口 | sink_size + window_size |
| `window_only` | 仅滑动窗口（不保留 sink） | window_size |
| `full` | 保留完整 cache | 无限制 |
| `prune` | 完全剪枝（零输出） | 0 |

#### 函数式接口

```python
from kvcompress.methods import head_aware_compress

compressed = head_aware_compress(
    past_key_values,
    classifications_path="path/to/classifications.json",
    # 或不提供 classifications_path，使用默认配置
    num_layers=32,
    num_heads=32,
    default_sink_size=4,
    default_window_size=8,
)
```

---

## 脚本使用说明

### 1. `scripts/analyze_attention.py`

运行注意力头分析，生成统计数据和分类结果。

```bash
# 分析 Pythia-2.8b
python scripts/analyze_attention.py \
    --model_id EleutherAI/pythia-2.8b \
    --max_tokens 2048 \
    --chunk_size 512 \
    --output_dir results/attention_analysis_pythia-2.8b

# 参数说明
--model_id      # HuggingFace 模型 ID
--max_tokens    # 最大分析 token 数
--chunk_size    # 分块大小（控制内存）
--sink_size     # sink tokens 数量（默认 4）
--local_window  # 局部窗口大小（默认 8）
--no_bf16       # 禁用 bfloat16（使用 float32）
```

**输出文件**:
- `head_statistics.json`: 每个头的详细统计数据
- `head_classifications.json`: 分类结果和压缩建议
- `entropy_heatmap.png`: 熵值热力图
- `position_preference.png`: 位置偏好图
- `sink_ratio_analysis.png`: sink ratio 分析图
- `analysis_summary.txt`: 文本摘要报告

---

### 2. `scripts/validate_head_classification.py`

验证并可视化头分类结果。

```bash
# 使用现有分析结果进行验证
python scripts/validate_head_classification.py \
    --results_dir results/attention_analysis_pythia-2.8b

# 使用更新的分类逻辑重新分类
python scripts/validate_head_classification.py \
    --results_dir results/attention_analysis_pythia-2.8b \
    --reclassify

# 参数说明
--results_dir   # 分析结果目录
--output_dir    # 验证输出目录（默认: {results_dir}/validation）
--reclassify    # 使用更新的逻辑重新分类
```

**输出文件**:
- `refined_classifications.json`: 更新后的分类结果（仅 --reclassify）
- `relative_position_by_type.png`: 按类型的相对位置分布图
- `sink_vs_local_scatter.png`: sink ratio vs local ratio 散点图
- `entropy_by_type.png`: 按类型的熵值分布图
- `validation_report.txt`: 详细验证报告

---

### 3. `scripts/run_ablation_study.py`

运行消融实验，评估不同压缩策略的效果。

```bash
# 完整消融实验
python scripts/run_ablation_study.py \
    --model_id EleutherAI/pythia-2.8b \
    --max_tokens 2000

# 仅测试 sink 重要性
python scripts/run_ablation_study.py \
    --ablation sink_importance \
    --max_tokens 1000

# 仅测试头感知压缩
python scripts/run_ablation_study.py \
    --ablation head_aware \
    --classifications_path results/attention_analysis_pythia-2.8b/validation/refined_classifications.json

# 参数说明
--model_id              # 模型 ID
--max_tokens            # 评估 token 数
--ablation              # 消融类型: all, sink_importance, window_size, head_aware
--classifications_path  # 分类文件路径
--output_dir            # 输出目录
```

**消融实验类型**:

| 类型 | 说明 |
|------|------|
| `sink_importance` | 比较 sink+window vs window-only，验证 sink tokens 的重要性 |
| `window_size` | 测试不同窗口大小对 PPL 的影响 |
| `head_aware` | 比较头感知压缩 vs 统一压缩 |
| `all` | 运行所有消融实验 |

---

### 4. `scripts/benchmark.py`

性能基准测试，支持头感知压缩方法。

```bash
# 测试头感知压缩
python scripts/benchmark.py \
    --method head_aware \
    --classifications_path results/attention_analysis_pythia-2.8b/validation/refined_classifications.json

# 比较所有方法（包括 head_aware）
python scripts/benchmark.py --compare_all

# 参数说明
--method                # 压缩方法: l2_compress, fix_size_l2, streaming_llm, head_aware
--classifications_path  # 头分类文件路径
--num_samples           # 测试样本数
--max_tokens            # 最大 token 数
--max_new_tokens        # 生成 token 数
```

---

## 完整使用流程

### Step 1: 运行注意力分析

```bash
python scripts/analyze_attention.py \
    --model_id EleutherAI/pythia-2.8b \
    --max_tokens 2048
```

### Step 2: 验证并更新分类

```bash
python scripts/validate_head_classification.py \
    --results_dir results/attention_analysis_pythia-2.8b \
    --reclassify
```

### Step 3: 运行消融实验（可选）

```bash
python scripts/run_ablation_study.py \
    --ablation sink_importance \
    --max_tokens 1000
```

### Step 4: 基准测试

```bash
python scripts/benchmark.py \
    --method head_aware \
    --classifications_path results/attention_analysis_pythia-2.8b/validation/refined_classifications.json
```

### Step 5: 在推理中使用

```python
from kvcompress.methods import head_aware_compress, HeadAwareCompressor

# 加载压缩器
compressor = HeadAwareCompressor.from_classifications(
    "results/attention_analysis_pythia-2.8b/validation/refined_classifications.json"
)

# 在生成循环中使用
for step in range(max_steps):
    outputs = model(input_ids, past_key_values=past_kv, use_cache=True)
    past_kv = outputs.past_key_values
    
    # 压缩 KV cache
    past_kv = compressor.compress(past_kv)
    
    # 继续生成...
```

---

## 结果解读

### 验证报告示例

```
CLASSIFICATION SUMMARY
----------------------------------------
  sink_positional     :  361 heads ( 35.3%)  # 需要 sink + 小窗口
  true_positional     :    9 heads (  0.9%)  # 仅需窗口
  sink_mixed          :  452 heads ( 44.1%)  # 需要 sink + 大窗口
  gathering           :   88 heads (  8.6%)  # 需要完整 cache
  mixed               :  114 heads ( 11.1%)  # 保守策略

COMPRESSION POTENTIAL
----------------------------------------
  Compressible (not full cache): 822 (80.3%)
  Require sink tokens: 813 (79.4%)
```

### 关键指标

| 指标 | 说明 | 好的范围 |
|------|------|----------|
| PPL (Perplexity) | 困惑度，越低越好 | 压缩后增加 < 5% |
| Accuracy | 下一个 token 预测准确率 | 压缩后下降 < 5% |
| Cache Size | KV cache 大小 | 压缩率 40-60% |

---

## 常见问题

### Q: 为什么 sink tokens 这么重要？

A: 分析显示 79.4% 的头会将大量注意力分配给初始 tokens（attention sinks）。这些 tokens 作为"注意力汇聚点"，即使内容不重要，也承载了模型的位置编码信息。删除它们会导致显著的质量下降。

### Q: 为什么不能对所有头使用相同的压缩策略？

A: 不同头有不同的功能：
- **Gathering heads** (8.6%) 需要访问完整上下文来捕获语义信息
- **True positional heads** (0.9%) 仅关注相对位置，不需要 sink tokens
- **Sink-focused heads** (79.4%) 需要 sink tokens 但不需要完整历史

统一策略会在某些头上过度压缩（损失质量）或压缩不足（浪费内存）。

### Q: 如何选择窗口大小？

建议：
- `sink_positional`: window = 8（这些头仅关注最近几个 token）
- `sink_mixed`: window = 16-32（需要更多上下文）
- `gathering`: 不压缩或使用 SnapKV 风格的选择性压缩

### Q: 这个方法和 StreamingLLM 有什么区别？

StreamingLLM 对所有头使用统一的 sink + window 策略。本方法：
1. 识别哪些头真正需要 sink tokens
2. 为不同类型的头使用不同的窗口大小
3. 保留 gathering heads 的完整 cache
4. 对 true positional heads 可以跳过 sink tokens

---

## API 参考

### `kvcompress.methods.head_aware_compress`

```python
def head_aware_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    classifications_path: Optional[str] = None,
    compressor: Optional[HeadAwareCompressor] = None,
    num_layers: int = 32,
    num_heads: int = 32,
    default_sink_size: int = 4,
    default_window_size: int = 8,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    头感知 KV Cache 压缩。
    
    Args:
        past_key_values: KV cache
        classifications_path: 分类 JSON 文件路径
        compressor: 预配置的压缩器（如提供则忽略其他参数）
        num_layers: 层数（无分类时使用）
        num_heads: 每层头数（无分类时使用）
        default_sink_size: 默认 sink 大小
        default_window_size: 默认窗口大小
        skip_layers: 跳过压缩的层索引
    
    Returns:
        压缩后的 KV cache
    """
```

### `kvcompress.methods.HeadAwareCompressor`

```python
class HeadAwareCompressor:
    @classmethod
    def from_classifications(cls, path: str) -> "HeadAwareCompressor":
        """从分类文件加载压缩器配置"""
    
    @classmethod
    def from_model_config(cls, config, **kwargs) -> "HeadAwareCompressor":
        """使用统一配置创建压缩器"""
    
    def compress(self, past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """执行压缩"""
    
    def get_compression_summary(self) -> dict:
        """获取压缩配置摘要"""
```

---

## 更新日志

### v1.0.0 (2025-12)
- 初始实现
- 支持 5 种头类型分类
- 支持 4 种压缩策略
- 完整的分析、验证、基准测试流程


# CS3602 LLM Inference Acceleration

CS3602大作业：针对大型语言模型的KV Cache优化与推理加速。

本项目实现了多种 KV Cache 压缩方法，统一在 `kvcompress` 库中管理：

1. **L2 Compress (KnormPress)** - 基于 L2 范数的比例压缩
2. **Fix-Size L2** - 固定大小 KV Cache 压缩
3. **StreamingLLM** - 基于 Attention Sink 的流式压缩

## 项目概述

KVCompress 是一个统一的 KV Cache 压缩库，支持多种压缩策略：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `l2_compress` | 按 `keep_ratio` 比例压缩，保留低 L2 范数 token | 通用压缩 |
| `fix_size_l2_compress` | 维持固定 KV Cache 大小，支持多种驱逐策略 | 内存受限场景 |
| `streaming_llm_compress` | 保留 attention sinks + 最近 tokens | 无限长度流式输入 |

### StreamingLLM 方法

StreamingLLM 是来自 MIT Han Lab 的方法（ICLR 2024），核心发现是：

- LLM 会将大量 attention 分配给初始 tokens（"attention sinks"），即使它们语义上不重要
- 通过保留这些 attention sinks + 滑动窗口的最近 tokens，可以处理无限长度的输入

```
Cache 结构: [initial tokens (0:start_size)] + [recent tokens (seq_len-recent_size:seq_len)]
默认配置: 4 initial tokens + 508 recent tokens = 512 total
```

### Speculative Decoding (投机解码)

本项目还实现了 **Speculative Decoding** 推理加速方法，位于 `spec_decode/` 目录。

**原理**: 使用小型 Draft 模型快速生成候选 tokens，然后由大型 Target 模型并行验证，从而减少 Target 模型的 forward 次数。

| 方法 | 说明 | 加速比 | 适用场景 |
|------|------|--------|----------|
| Custom Spec Decode | 自定义实现的投机解码 | 1.8-2.1x | 学习研究、定制优化 |
| HuggingFace Spec Decode | HF 内置 `assistant_model` | 2.9-3.0x | 生产环境推荐 |

```python
from spec_decode.core import SpeculativeGenerator

generator = SpeculativeGenerator(
    target_model=target_model,  # Pythia-2.8B
    draft_model=draft_model,    # Pythia-70M
    tokenizer=tokenizer,
    K=5  # 每轮草稿 5 个 tokens
)
output = generator.generate("The future of AI is", max_new_tokens=100)
```

详细文档请参考 [spec_decode/README.md](spec_decode/README.md)。

## 项目结构

```
.
├── README.md                    # 项目说明文档
├── LICENSE                      # 许可证
│
├── docs/                        # 文档
│   ├── lab-instruction.md       # 作业要求
│   ├── KnormPress.pdf           # KnormPress 论文
│   ├── L2_COMPRESS_ANALYSIS.md  # 压缩效果分析
│   └── SPEC_DECODE_OPTIMIZATION.md  # Speculative Decoding 优化分析 ⭐
│
├── data/                        # 数据集
│   └── pg19.parquet             # PG-19 长文本数据集
│
├── kvcompress/                  # KV Cache 压缩库 ⭐
│   ├── __init__.py              # 统一导出
│   ├── methods/                 # 压缩方法
│   │   ├── __init__.py          # 方法注册表
│   │   ├── base.py              # 基类和接口
│   │   ├── l2_compress.py       # KnormPress L2 压缩
│   │   ├── fix_size_l2.py       # 固定大小 L2 压缩
│   │   └── streaming_llm.py     # StreamingLLM 方法
│   ├── evaluate.py              # 统一评估模块
│   ├── benchmark.py             # 统一基准测试模块
│   └── utils.py                 # 工具函数
│
├── spec_decode/                 # Speculative Decoding 加速库 ⭐
│   ├── core/                    # 核心实现
│   │   ├── speculative_generator.py  # 主生成器类
│   │   └── static_cache.py      # 静态 KV Cache
│   ├── benchmark_detailed.py    # 详细性能测试
│   ├── test_correctness.py      # 正确性验证
│   └── README.md                # 详细文档
│
├── precision/                   # 精度评估
│   └── precision_eval.py        # FP32/FP16/BF16 性能对比
│
├── scripts/                     # 工具脚本
│   ├── benchmark.py             # 统一基准测试入口
│   └── plot_compression_results.py  # 可视化绘图
│
├── baseline_test.py             # 基线性能测试
│
├── results/                     # 结果图表
```

## 环境配置

### 依赖安装

```bash
# 创建并激活 conda 环境
conda create -n nlp python=3.11
conda activate nlp

# 安装依赖
pip install torch transformers datasets numpy tqdm
```

### 模型和数据集

- **模型**: `EleutherAI/pythia-2.8b`
- **数据集**: `PG-19` (长文本), `wikitext-2-raw-v1` (短文本)

## 使用方法

### 1. 统一基准测试（推荐）

```bash
# 测试 L2 压缩（KnormPress）
python scripts/benchmark.py --method l2_compress --keep_ratios 1.0,0.8,0.5

# 测试固定大小 L2 压缩
python scripts/benchmark.py --method fix_size_l2 --fix_kv_sizes 256,512 --strategies keep_low

# 测试 StreamingLLM
python scripts/benchmark.py --method streaming_llm --start_size 4 --recent_sizes 252,508,1020

# 对比所有方法
python scripts/benchmark.py --compare_all
```

### 2. 在代码中使用

```python
from kvcompress import (
    l2_compress, 
    fix_size_l2_compress, 
    streaming_llm_compress,
    evaluate_with_compression
)

# 方法1: L2 比例压缩 (KnormPress)
compressed_kv = l2_compress(
    past_key_values,
    keep_ratio=0.8,      # 保留 80%
    prune_after=1000,    # 超过 1000 token 才压缩
    skip_layers=[0, 1]   # 跳过前两层
)

# 方法2: 固定大小压缩
compressed_kv = fix_size_l2_compress(
    past_key_values,
    fix_kv_size=512,       # 最多保留 512 token
    keep_ratio=0.2,        # 最近 20% 不驱逐
    strategy="keep_low",   # 保留低范数 token
    skip_layers=[0, 1]
)

# 方法3: StreamingLLM
compressed_kv = streaming_llm_compress(
    past_key_values,
    start_size=4,          # 保留 4 个 attention sink tokens
    recent_size=508,       # 保留最近 508 个 tokens
)

# 使用统一评估接口
results = evaluate_with_compression(
    model, tokenizer, text,
    compress_fn=streaming_llm_compress,
    compress_kwargs={"start_size": 4, "recent_size": 508}
)
print(f"PPL: {results['perplexity']:.2f}, Acc: {results['accuracy']:.2%}")
```

### 3. 使用方法注册表

```python
from kvcompress import get_compress_fn, list_methods

# 查看所有可用方法
print(list_methods())  # ['l2_compress', 'fix_size_l2', 'streaming_llm']

# 通过名称获取压缩函数
compress_fn = get_compress_fn("streaming_llm")
compressed = compress_fn(past_key_values, start_size=4, recent_size=508)
```

## 核心算法

### l2_compress (比例压缩)

```
输入: KV Cache (seq_len tokens), keep_ratio
输出: 压缩后的 KV Cache (seq_len * keep_ratio tokens)

1. 计算每个 token 的 L2 范数
2. 按范数升序排序
3. 保留前 keep_ratio 比例的低范数 token
4. 恢复时间顺序
```

### fix_size_l2_compress (固定大小)

```
输入: KV Cache, fix_kv_size, keep_ratio
输出: 最多 fix_kv_size tokens 的 KV Cache

1. 如果 seq_len <= fix_kv_size，不压缩
2. 计算保护区大小: protected = fix_kv_size * keep_ratio
3. 驱逐区 = 前 (seq_len - protected) 个 token
4. 从驱逐区选择 (fix_kv_size - protected) 个 token 保留
5. 合并: 保留的驱逐区 token + 保护区 token
```

### streaming_llm_compress (StreamingLLM)

```
输入: KV Cache, start_size, recent_size
输出: 最多 (start_size + recent_size) tokens 的 KV Cache

1. 如果 seq_len <= (start_size + recent_size)，不压缩
2. 保留 attention sinks: tokens[0:start_size]
3. 保留最近 tokens: tokens[-recent_size:]
4. 拼接: attention sinks + recent tokens
```

## 实验结果

### 方法对比

| 方法 | 特点 | PPL 影响 | 速度 |
|------|------|----------|------|
| Baseline | 无压缩 | 基准 | 基准 |
| L2 (keep_ratio=0.8) | 保留 80% 重要 token | +2-5% | 快 |
| Fix-Size (512, keep_low) | 固定大小，智能驱逐 | +5-10% | 中等 |
| StreamingLLM (4+508) | Attention sinks + 滑动窗口 | +3-8% | 最快 |

### 可视化结果

运行绘图脚本生成可视化图表：
```bash
python scripts/plot_compression_results.py
```

## 参考文献

- **KnormPress**: [A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression](https://arxiv.org/abs/2406.11430) (EMNLP 2024)
- **StreamingLLM**: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (ICLR 2024)
- **Pythia 模型**: [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)

## 总结

本项目实现了完整的 LLM 推理加速方案：

### KV Cache 压缩 (`kvcompress`)

✅ **多种压缩方法**: l2_compress, fix_size_l2, streaming_llm  
✅ **统一接口**: 所有方法使用相同的函数签名  
✅ **方法注册表**: 方便扩展新方法  
✅ **统一评估**: 支持 PPL, Accuracy, TTFT, TPOT  

### Speculative Decoding (`spec_decode`)

✅ **自定义实现**: 完整的投机解码算法，易于理解和修改  
✅ **性能对比**: 与 HuggingFace 实现的详细 benchmark  
✅ **正确性验证**: 确保输出与原模型一致  
✅ **优化分析**: 包含进一步优化的研究方向

## 作者

CS3602 课程作业

## 致谢

感谢 KnormPress 和 StreamingLLM 论文作者提供的开源实现和详细文档。

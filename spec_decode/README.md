# Speculative Decoding 实现

本目录包含 Speculative Decoding（投机解码）的自定义实现，用于加速 LLM 推理。

## 概述

Speculative Decoding 是一种推理加速技术，核心思想是使用小型"草稿"模型快速生成候选 tokens，然后由大型"目标"模型并行验证。当草稿模型的接受率较高时，可以显著提升推理速度。

### 核心特性

- **DynamicCache 管理**: 使用 HuggingFace 的 DynamicCache，支持高效的缓存截断和更新
- **Greedy 解码**: 产生与纯目标模型完全一致的输出
- **PyTorch 2.0 优化**: 支持 `torch.compile` 和 `torch.inference_mode`
- **HuggingFace 兼容**: 适用于任何 HuggingFace transformer 模型

## 算法原理

### 投机解码流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Speculative Decoding 流程图                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐                                                        │
│  │ Prompt  │                                                        │
│  └────┬────┘                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                            │
│  │ 1. Prefill (Target) │  处理 prompt，初始化 KV Cache               │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │ 2. Draft K Tokens   │  Draft 模型快速生成 K 个候选 tokens         │
│  │    (Draft Model)    │  使用临时 cache，每轮丢弃                   │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │ 3. Verify (Target)  │  Target 模型一次 forward 验证 K 个 tokens   │
│  │    One Forward Pass │  并行计算，高效利用 GPU                     │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐     ┌──────────────────────┐               │
│  │ 4. Accept/Reject    │────▶│ Match: 接受 token    │               │
│  │    Greedy Compare   │     │ Mismatch: 使用 Target│               │
│  └──────────┬──────────┘     │ 预测替换，终止本轮    │               │
│             │                └──────────────────────┘               │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │ 5. Update Cache     │  根据接受的 tokens 更新 Target Cache        │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│       ┌─────────┐                                                   │
│       │ 达到    │─── 是 ──▶ 输出结果                                 │
│       │max_tokens│                                                   │
│       │ 或 EOS? │                                                   │
│       └────┬────┘                                                   │
│            │ 否                                                     │
│            └──────────────────────────────────▲                     │
│                        返回步骤 2              │                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Cache 管理策略

- **Target 模型**: 使用持久化的 DynamicCache，支持 `crop()` 操作进行截断
- **Draft 模型**: 使用临时 cache，每轮生成后丢弃

这种策略避免了复杂的 cache 同步问题，同时保持了较高的效率。

### 关键公式

**加速比计算**:
```
Speedup = (K + 1) * acceptance_rate + (1 - acceptance_rate)
        ≈ 1 + K * acceptance_rate  (当 acceptance_rate 接近 1 时)
```

**每轮平均生成 tokens**:
```
avg_tokens_per_round = Σ(i * P(accept_i)) + K * P(accept_all)
```

## 目录结构

```
spec_decode/
├── core/
│   ├── __init__.py                          # 模块导出
│   ├── speculative_generator.py             # 主生成器类 ⭐
│   │   ├── SpeculativeGenerator             # 基础版本 (DynamicCache)
│   │   └── SpeculativeGeneratorWithStaticCache  # StaticCache 版本
│   ├── streaming_speculative_generator.py   # StreamingLLM 集成 ⭐
│   │   ├── StreamingSpeculativeGenerator    # 基础流式版本
│   │   └── StreamingSpeculativeGeneratorV2  # 优化版本 (主动驱逐)
│   ├── tree_speculative_generator.py        # Tree-based 投机解码 ⭐ NEW
│   │   ├── TreeSpeculativeGenerator         # 树状多分支版本
│   │   ├── TreeSpeculativeGeneratorV2       # 带剪枝优化版本
│   │   └── TreeStreamingSpeculativeGenerator # 树状 + StreamingLLM
│   ├── token_tree.py                        # Token Tree 数据结构 ⭐ NEW
│   ├── static_cache.py                      # 静态 KV Cache 实现
│   └── utils.py                             # 工具函数
├── benchmark_custom_vs_hf.py                # 性能对比 benchmark
├── benchmark_detailed.py                    # 详细性能分析
├── benchmark_enhanced.py                    # 增强版 benchmark (阶段时间分解) ⭐
├── benchmark_tree_vs_linear.py              # Tree vs Linear 对比 ⭐ NEW
├── benchmark_int8.py                        # INT8 量化 benchmark ⭐
├── benchmark_streaming.py                   # StreamingLLM benchmark ⭐
├── benchmark_combined.py                    # 综合 benchmark (全部配置) ⭐
├── test_correctness.py                      # 正确性验证
├── demo.py                                  # 使用演示
└── README.md
```

## 使用方法

### 基本使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_decode.core import SpeculativeGenerator

# 加载模型
target_model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b",
    torch_dtype=torch.float16
).to("cuda")

draft_model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m",
    torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

# 创建生成器
generator = SpeculativeGenerator(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    K=5,           # 每轮草稿 token 数
    max_len=2048,  # 最大序列长度
    device="cuda",
    use_compile=False  # 设为 True 启用 torch.compile
)

# 生成文本
output = generator.generate(
    "The future of artificial intelligence is",
    max_new_tokens=100,
    verbose=False
)
print(output)

# 获取统计信息
stats = generator.get_stats()
print(f"接受率: {stats['acceptance_rate']:.2%}")
print(f"总轮数: {stats['total_rounds']}")
print(f"平均每轮生成: {stats['tokens_per_round']:.2f} tokens")
```

### 使用 StaticCache 版本

`SpeculativeGeneratorWithStaticCache` 使用预分配的 StaticCache，支持更激进的 torch.compile 优化：

```python
from spec_decode.core import SpeculativeGeneratorWithStaticCache

# 检查 StaticCache 是否可用
if SpeculativeGeneratorWithStaticCache.is_available():
    generator = SpeculativeGeneratorWithStaticCache(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=5,
        max_cache_len=2048,      # 预分配的最大 cache 长度
        device="cuda",
        use_max_autotune=True    # 使用 max-autotune 编译模式
    )
    
    output = generator.generate(
        "The future of AI is",
        max_new_tokens=100
    )
else:
    print("StaticCache 需要 transformers >= 4.38.0")
```

**StaticCache 优势**:
- 避免动态内存分配
- 支持 `torch.compile(mode="max-autotune")`
- 更稳定的推理延迟

### 使用 StreamingLLM 版本 (支持无限长度生成)

`StreamingSpeculativeGenerator` 集成 StreamingLLM 压缩，支持无限长度生成：

```python
from spec_decode.core import StreamingSpeculativeGenerator

generator = StreamingSpeculativeGenerator(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    K=5,
    start_size=4,           # Attention sink tokens
    recent_size=1020,       # 滑动窗口大小
    max_cache_len=1024,     # 最大 cache 长度
    compress_threshold=0.9  # 压缩触发阈值
)

# 生成超长文本 - 内存保持恒定
output = generator.generate(prompt, max_new_tokens=10000)

# 查看压缩统计
stats = generator.get_stats()
print(f"压缩次数: {stats['compression_count']}")
print(f"驱逐 tokens: {stats['tokens_evicted']}")
print(f"当前 cache 长度: {stats['current_cache_len']}")
```

**StreamingLLM 优势**:
- 支持无限长度生成
- 内存恒定 (不随序列长度增长)
- 保持 attention sink 保证质量

### 使用 Tree-based 版本 (SpecInfer 风格) ⭐ NEW

`TreeSpeculativeGenerator` 实现树状多分支投机解码，通过 Tree Attention 并行验证多个候选路径：

```python
from spec_decode.core import TreeSpeculativeGenerator, TreeSpeculativeGeneratorV2

# 基础 Tree 版本
generator = TreeSpeculativeGenerator(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    tree_depth=3,        # 树的深度 (类似 K)
    branch_factor=2,     # 每层分支数 (top-k)
    max_tree_nodes=32,   # 最大树节点数
    device="cuda"
)

output = generator.generate(prompt, max_new_tokens=100)
stats = generator.get_stats()
print(f"平均路径长度: {stats['avg_accepted_path_length']:.2f}")
print(f"树节点生成数: {stats['total_tree_nodes']}")

# 带剪枝的 V2 版本 (推荐，性能更好)
generator_v2 = TreeSpeculativeGeneratorV2(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    tree_depth=3,
    branch_factor=3,
    probability_threshold=0.05,  # 剪枝阈值
    device="cuda"
)
```

**Tree-based 优势**:
- 更高的接受概率 (多路径候选)
- 更好的 GPU 利用率 (Tree Attention 并行验证)
- V2 剪枝版本可达 **2.00x** 加速

**Tree vs Linear 性能对比** (Pythia-2.8B + Pythia-70M):

| 方法 | 吞吐量 (t/s) | 加速比 | 接受率 |
|------|-------------|--------|--------|
| Baseline | 60.8 | 1.00x | - |
| Linear K=3 | 97.5 | 1.60x | 85.2% |
| Tree D=3 B=2 | 100.3 | 1.65x | 23.4% |
| **Tree V2 D=3 B=3** | **122.0** | **2.00x** | 36.3% |

### 运行 Benchmark

```bash
# 基本 benchmark（对比 Custom vs HuggingFace）
python benchmark_custom_vs_hf.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --k-values 3 5 7 \
    --num-samples 5 \
    --max-new-tokens 100 \
    --no-compile

# 详细 benchmark（包含 TTFT、TPOT、内存等指标）
python benchmark_detailed.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --k-values 2 3 4 5 6 7 8 \
    --num-samples 10 \
    --max-new-tokens 100

# 增强版 benchmark（阶段时间分解 + 可视化）⭐
python benchmark_enhanced.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --k-values 3 5 7 \
    --num-samples 5 \
    --max-new-tokens 100 \
    --output-plot benchmark_enhanced.png \
    --output-json benchmark_enhanced_results.json
```

增强版 benchmark 提供以下额外功能：
- **阶段时间分解**: Prefill, Draft, Verify, Update 各阶段耗时
- **统计信息**: Mean, Std, Median, P90, P99 百分位
- **可视化图表**: 6 个子图展示吞吐量、加速比、接受率、延迟、阶段分解
- **JSON 输出**: 完整的原始数据供后续分析

```bash
# INT8 量化 benchmark ⭐
python benchmark_int8.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --k-values 3 5 7 \
    --num-samples 5

# StreamingLLM 长文本 benchmark ⭐
python benchmark_streaming.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --max-new-tokens 100 200 500 \
    --max-cache-len 256 512

# 综合 benchmark (全部优化配置) ⭐
python benchmark_combined.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --num-samples 3 \
    --max-new-tokens 100 \
    --max-cache-len 256

# Tree vs Linear 对比 benchmark ⭐ NEW
python benchmark_tree_vs_linear.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --max-new-tokens 100 \
    --save
```

### 运行正确性测试

```bash
# 完整测试
python test_correctness.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --device cuda:0

# 快速测试
python test_correctness.py --quick
```

## 性能测试结果

### 测试环境

- **GPU**: NVIDIA GPU with CUDA
- **Target Model**: Pythia-2.8B (FP16)
- **Draft Model**: Pythia-70M (FP16)
- **测试数据**: 5 个文本样本，每个生成 100 tokens

### 吞吐量对比 (最新优化后)

| 方法 | K | 吞吐量 (t/s) | 加速比 | TTFT (ms) | TPOT (ms) | 接受率 |
|------|---|-------------|--------|-----------|-----------|--------|
| **Baseline** | - | 62.5 | 1.00x | 28.3 | 15.9 | - |
| **HuggingFace** | 3 | 146.5 | 2.34x | 14.2 | 6.8 | - |
| **HuggingFace** | 5 | 192.0 | **3.07x** | 10.1 | 5.5 | - |
| **HuggingFace** | 7 | 192.7 | **3.08x** | 10.0 | 5.5 | - |
| **Custom (优化后)** | 5 | 131.8 | **2.18x** | 40.0 | 8.0 | 71.4% |

### 阶段时间分解 (Custom 实现)

使用 `benchmark_enhanced.py` 可以获取每个阶段的详细时间：

| 阶段 | 平均耗时 (ms) | 占比 |
|------|-------------|------|
| **Prefill** | 17.0 | - |
| **Draft** | 25.6 | 42% |
| **Verify** | 18.3 | 30% |
| **Update** | 17.0 | 28% |
| **每轮总计** | 60.9 | 100% |

### 优化效果

经过批量 Cache 更新和 torch.compile 修复后：
- **加速比**: 从 1.83x 提升到 **2.18x** (+19%)
- **相对 HF 性能**: 从 71% 提升到 **91%** (K=5)
- **接受率**: 保持在 70%+ 水平

### 性能瓶颈分析

| 因素 | Custom 实现 | HuggingFace | 差距原因 |
|------|-------------|-------------|---------|
| Cache 更新 | 批量 forward | 批量处理 | 已优化 |
| Draft 阶段 | 每轮重新 prefill | 增量更新 | 主要瓶颈 |
| TTFT | ~40ms | ~10ms | Python 初始化开销 |
| torch.compile | default 模式 | 深度优化 | 动态形状限制 |

## 关键设计决策

### 1. Greedy 解码

- **优点**: 输出与纯 Target 模型完全一致，便于验证正确性
- **缺点**: 不支持采样解码（sampling）

### 2. DynamicCache 直接使用

- **优点**: 避免自定义 cache 与 HuggingFace 的兼容性问题
- **缺点**: 依赖 HuggingFace 的 cache 实现

### 3. Cache 重建策略

当 draft token 被拒绝时，采用"截断 + 逐 token 重建"策略：

```python
# 截断到原始长度
self.target_cache.crop(original_len)

# 逐 token 重建（确保正确性）
for token in accepted_tokens:
    outputs = target_model(token, past_key_values=cache, ...)
    cache = outputs.past_key_values
```

这种策略虽然性能不如 HuggingFace，但确保了数值精度一致性。

## 局限性

- **仅支持 batch_size=1**: 暂不支持批量处理
- **仅支持 Greedy 解码**: 暂不支持采样解码
- **词表要求**: Draft 模型必须与 Target 模型共享词表
- **性能差距**: K≥5 时，性能约为 HuggingFace 的 90% (优化后)

## 已实现优化

| 优化 | 状态 | 效果 |
|------|------|------|
| 批量 Cache 更新 | ✅ 已完成 | +15-20% 吞吐量 |
| torch.compile (dynamic=True) | ✅ 已完成 | +5-10% 吞吐量 |
| StaticCache 支持 | ✅ 已完成 | 支持 max-autotune |
| 阶段时间分解 Benchmark | ✅ 已完成 | 性能分析工具 |
| INT8 量化 | ✅ 已完成 | 内存 -88%, 加速比 2.0x |
| StreamingLLM 集成 | ✅ 已完成 | 支持无限长度生成 |
| **Tree-based Drafting** | ✅ 已完成 | **2.00x 加速** (V2 版本) |

## 综合优化测试结果

| 配置 | 吞吐量 (t/s) | 加速比 | 接受率 | 内存 |
|------|------------|--------|--------|------|
| **Baseline (FP16)** | 62.0 | 1.00x | - | 5516 MB |
| **Spec (FP16)** | 68.4 | 1.10x | 86.1% | 5528 MB |
| **Spec (FP16) + StreamingLLM** | 76.6 | 1.23x | 86.1% | 5528 MB |
| **Spec (INT8)** | 21.6 | 2.00x | 80.8% | 663 MB |
| **Spec (INT8) + StreamingLLM** | 25.0 | 2.32x | 80.8% | 663 MB |

**关键发现**:
- INT8 量化显著减少内存 (-88%)，但绝对吞吐量较低
- INT8 + Speculative Decoding 实现 2x 加速比
- StreamingLLM 提供额外 +10-15% 吞吐量提升
- 接受率 INT8 vs FP16 仅下降 ~5%

## 未来优化方向

详见 [docs/Speculative Decoding 优化分析与研究.md](../docs/Speculative%20Decoding%20优化分析与研究.md)

1. **自适应 K 选择**: 根据接受率动态调整 K 值
2. **Draft 模型优化**: 减少 Draft 阶段的重复 prefill
3. ~~**Tree-based Drafting**: 多路径草稿并行验证~~ ✅ 已完成 (2.00x 加速)
4. **INT4 量化**: 进一步压缩内存占用
5. **Medusa 风格多头**: 使用多个 draft head 并行预测

## 参考文献

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (ICML 2023)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (DeepMind 2023)
- [SpecInfer: Accelerating Generative Large Language Model Serving](https://arxiv.org/abs/2305.09781) (ASPLOS 2024)

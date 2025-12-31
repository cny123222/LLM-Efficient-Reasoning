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
│   ├── __init__.py                 # 模块导出
│   ├── speculative_generator.py    # 主生成器类 ⭐
│   ├── static_cache.py             # 静态 KV Cache 实现
│   └── utils.py                    # 工具函数
├── benchmark_custom_vs_hf.py       # 性能对比 benchmark
├── benchmark_detailed.py           # 详细性能分析 ⭐
├── test_correctness.py             # 正确性验证
├── demo.py                         # 使用演示
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

### 吞吐量对比

| 方法 | K | 吞吐量 (tokens/s) | 加速比 | 接受率 |
|------|---|------------------|--------|--------|
| **Baseline** | - | 60.7 | 1.00x | - |
| **HuggingFace** | 3 | 98.9 | 1.63x | - |
| **HuggingFace** | 5 | 179.3 | **2.96x** | - |
| **HuggingFace** | 7 | 178.7 | **2.95x** | - |
| **Custom** | 3 | 108.4 | 1.79x | 83.33% |
| **Custom** | 5 | 127.6 | 2.10x | 64.52% |
| **Custom** | 7 | 125.9 | 2.08x | 47.62% |

### 性能分析

1. **K=3 时**: Custom 实现（1.79x）略优于 HuggingFace（1.63x）
2. **K=5/7 时**: HuggingFace（~3.0x）明显优于 Custom（~2.1x）
3. **接受率趋势**: K 值增大时，接受率下降（83% → 48%）

### 性能差距原因

| 因素 | Custom 实现 | HuggingFace |
|------|-------------|-------------|
| Cache 更新 | 逐 token forward（保证正确性） | 批量处理 |
| Python 开销 | 多次 Python 调用 | C++ 后端优化 |
| 验证逻辑 | 纯 Python 实现 | 内部优化 |

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
- **性能差距**: K≥5 时，性能约为 HuggingFace 的 60-70%

## 优化方向

详见 [docs/SPEC_DECODE_OPTIMIZATION.md](../docs/SPEC_DECODE_OPTIMIZATION.md)

1. **批量 Cache 更新**: 一次 forward 处理所有 accepted tokens
2. **StaticCache 集成**: 使用 HuggingFace 的 StaticCache
3. **torch.compile 优化**: 解决与 CUDA Graph 的兼容性问题
4. **自适应 K 选择**: 根据接受率动态调整 K 值
5. **结合 KV 压缩**: 与 StreamingLLM/L2 压缩结合
6. **结合量化**: 与 FP16/INT8 量化结合

## 参考文献

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (ICML 2023)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (DeepMind 2023)
- [SpecInfer: Accelerating Generative Large Language Model Serving](https://arxiv.org/abs/2305.09781) (ASPLOS 2024)

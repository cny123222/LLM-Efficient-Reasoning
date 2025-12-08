# Pythia-6.9B 压缩效果验证实验方案

## 目标

验证 StreamingLLM 在 Pythia-6.9B 上表现更好的原因是：
1. "压缩提升预测能力"（不太可能）
2. "避免了 baseline 的数值/内存问题"（更可能）

## 实验设计

### 实验 1: 序列长度梯度测试

**目的**: 找到 baseline 开始退化的临界点

```bash
# 短序列 (应该 baseline 更好)
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 500 \
    --recent_sizes 252 --num_samples 5 --num_warmup 10

# 中等序列
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 1000 \
    --recent_sizes 252 --num_samples 5 --num_warmup 10

# 长序列
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 2000 \
    --recent_sizes 252 --num_samples 5 --num_warmup 10

# 超长序列
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 5000 \
    --recent_sizes 252 --num_samples 3 --num_warmup 10
```

**预期结果**（如果是数值问题）：
| 序列长度 | Baseline 优势 | 说明 |
|----------|--------------|------|
| 500 | Baseline > Streaming | 正常范围 |
| 1000 | Baseline ≈ Streaming | 开始出现问题 |
| 2000 | Baseline < Streaming | 明显退化 |
| 5000 | Baseline << Streaming | 严重退化 |

### 实验 2: 不同 Cache 大小对比

**目的**: 验证是否存在"最优压缩比"

```bash
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 2000 \
    --recent_sizes 124,252,508,1020 \
    --num_samples 5 --num_warmup 10
```

**预期结果**（如果是"压缩提升能力"）：
- 应该存在一个最优 cache 大小（如 256），太大或太小都不好

**预期结果**（如果是"数值稳定性"）：
- Cache 越大，越接近 baseline 的性能
- 所有 streaming 方法都应该比 baseline 稳定

### 实验 3: 小模型对照实验

**目的**: 验证现象是否只在大模型上出现

```bash
# Pythia-70M (小模型，应该 baseline 更好)
python scripts/benchmark.py --model_id EleutherAI/pythia-70m-deduped \
    --method streaming_llm --max_tokens 2000 \
    --recent_sizes 252 --num_samples 10 --num_warmup 5

# Pythia-410M (中等模型)
python scripts/benchmark.py --model_id EleutherAI/pythia-410m-deduped \
    --method streaming_llm --max_tokens 2000 \
    --recent_sizes 252 --num_samples 5 --num_warmup 10

# Pythia-6.9B (大模型)
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 2000 \
    --recent_sizes 252 --num_samples 5 --num_warmup 10
```

**预期结果**（如果是硬件限制）：
| 模型大小 | 现象 |
|----------|------|
| 70M | Baseline 更好（正常） |
| 410M | 开始出现反转 |
| 6.9B | Streaming 明显更好 |

### 实验 4: 不同硬件对比（如果可能）

如果有 CUDA GPU 访问权限：

```bash
# MPS (Apple Silicon) - 当前环境
python scripts/benchmark.py --model_id EleutherAI/pythia-6.9b \
    --method streaming_llm --max_tokens 2000 \
    --recent_sizes 252 --num_samples 5

# CUDA (NVIDIA GPU) - 如果可用
# 在 GPU 机器上运行相同命令
```

**预期结果**：
- 如果在 CUDA 上 baseline 表现正常，说明是 MPS 的问题
- 如果在 CUDA 上也出现类似现象，说明是模型/序列长度的问题

## 判断标准

### 支持"压缩提升能力"的证据

需要满足：
- ✅ 在所有模型大小上都观察到
- ✅ 在所有硬件上都观察到
- ✅ 存在一个"最优压缩比"
- ✅ 短序列和长序列上都有效

### 支持"数值稳定性问题"的证据

需要满足：
- ✅ 只在大模型上出现
- ✅ 序列越长，现象越明显
- ✅ Cache 越大，越接近 baseline
- ✅ 在 CUDA 上可能不出现

## 补充分析

### 如果是数值问题，为什么 PPL 和 Accuracy 都更好？

这很合理：
- PPL 和 Accuracy 都基于 logits 的质量
- 如果 baseline 的 attention 计算不稳定 → logits 质量下降
- StreamingLLM 的 attention 计算稳定 → logits 质量更好
- 因此 PPL 和 Accuracy 同时改善

### 类比

想象一个学生考试：
- **Baseline**: 拿到一本1000页的参考书，但记忆力有限，越看越糊涂（信息过载）
- **StreamingLLM**: 只给256页精选内容，虽然信息少但能记清楚

在正常情况（小模型/CUDA）下，学生能处理1000页 → Baseline 更好
在受限情况（大模型/MPS）下，学生处理不了1000页 → 256页反而更好

## 建议

1. **先运行实验1**，确认序列长度临界点
2. **运行实验3**，确认是否只在大模型上出现
3. 基于结果，更新 `PYTHIA_6.9B_ANOMALY_ANALYSIS.md`

## 结论模板

基于实验结果，填写：

```markdown
## 结论

### 现象描述
在 Pythia-6.9B + MPS + 长序列（>1000 tokens）的组合下，StreamingLLM 的质量指标优于 baseline。

### 原因分析
□ 压缩提升预测能力（证据：___）
☑ 数值稳定性问题（证据：___）
□ 其他（说明：___）

### 适用范围
- 模型大小: >= ___ 参数
- 序列长度: >= ___ tokens
- 硬件限制: MPS / CUDA / CPU

### 实际意义
这个现象的实际意义是：
1. 在资源受限环境下，固定 cache 大小是必要的
2. StreamingLLM 不仅节省内存，还避免了性能退化
3. 但这不代表"压缩提升能力"，而是"避免了退化"
```


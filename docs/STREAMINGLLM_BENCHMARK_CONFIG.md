# StreamingLLM Benchmark 实验配置说明

本文档详细说明在使用 `scripts/benchmark.py` 进行 StreamingLLM 基准测试时，所有实验组的配置以及 baseline 的测量方式。

---

## 📊 实验组配置

### 默认参数

当运行以下命令时：
```bash
python scripts/benchmark.py --method streaming_llm
```

默认参数为：
- `--start_size`: `4` (attention sinks 数量)
- `--recent_sizes`: `"252,508,1020"` (最近 tokens 数量列表)
- `--no_baseline`: `False` (默认包含 baseline)
- `--no_recent_only`: `False` (默认包含 recent_only 控制组)

### 生成的实验组

基于默认参数，会生成以下实验组：

| 实验组名称 | 压缩方法 | 配置参数 | 说明 |
|-----------|---------|---------|------|
| **baseline** | 无压缩 | `compress_fn: None` | 完全保留所有 KV cache |
| **recent_only_256** | 滑动窗口 | `window_size: 256` | 只保留最近 256 个 tokens |
| **streaming_256** | StreamingLLM | `start_size: 4, recent_size: 252` | 4 个 sinks + 252 个最近 tokens |
| **recent_only_512** | 滑动窗口 | `window_size: 512` | 只保留最近 512 个 tokens |
| **streaming_512** | StreamingLLM | `start_size: 4, recent_size: 508` | 4 个 sinks + 508 个最近 tokens |
| **recent_only_1024** | 滑动窗口 | `window_size: 1024` | 只保留最近 1024 个 tokens |
| **streaming_1024** | StreamingLLM | `start_size: 4, recent_size: 1020` | 4 个 sinks + 1020 个最近 tokens |

### 实验组说明

#### 1. Baseline（基线）
- **配置**：`compress_fn: None`, `kwargs: {}`
- **含义**：不使用任何 KV cache 压缩
- **KV Cache 大小**：等于输入序列长度（无限制增长）
- **用途**：作为性能和质量的上界参考

#### 2. Recent-Only（滑动窗口控制组）
- **配置**：`compress_fn: recent_only_compress`, `window_size: total_size`
- **含义**：只保留最近 N 个 tokens，丢弃所有历史 tokens
- **KV Cache 大小**：固定为 `window_size`
- **用途**：作为简单滑动窗口的对比基线，验证 attention sinks 的重要性

#### 3. StreamingLLM（实验组）
- **配置**：`compress_fn: streaming_llm_compress`, `start_size: 4, recent_size: X`
- **含义**：保留 4 个 attention sinks + 最近 X 个 tokens
- **KV Cache 大小**：固定为 `start_size + recent_size`
- **用途**：验证 StreamingLLM 方法相比纯滑动窗口的优势

### 实验组生成逻辑

代码位置：`scripts/benchmark.py` 的 `build_methods_config()` 函数

```python
elif args.method == "streaming_llm":
    recent_sizes = [int(x) for x in args.recent_sizes.split(",")]  # [252, 508, 1020]
    
    # 1. 添加 recent_only 控制组（如果未禁用）
    if not args.no_recent_only:
        for recent_size in recent_sizes:
            total_size = args.start_size + recent_size  # 4 + 252 = 256, etc.
            methods.append({
                "name": f"recent_only_{total_size}",
                "compress_fn": recent_only_compress,
                "kwargs": {"window_size": total_size}
            })
    
    # 2. 添加 StreamingLLM 实验组
    for recent_size in recent_sizes:
        total_size = args.start_size + recent_size
        methods.append({
            "name": f"streaming_{total_size}",
            "compress_fn": streaming_llm_compress,
            "kwargs": {
                "start_size": args.start_size,  # 4
                "recent_size": recent_size      # 252, 508, 1020
            }
        })
```

---

## 🔬 Baseline 测量方式

### Baseline 的定义

**Baseline = 无压缩的原始模型推理**

- **KV Cache 压缩**：无（`compress_fn: None`）
- **KV Cache 大小**：等于完整输入序列长度（无限制）
- **压缩参数**：空（`kwargs: {}`）

### Baseline 的测量流程

代码位置：`kvcompress/benchmark.py` 的 `measure_generation_metrics()` 函数

#### 1. 输入处理
```python
# Tokenize 输入文本
input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids = input_ids[:, :max_input_tokens].to(device)
```

#### 2. Prefill 阶段（测量 TTFT）
```python
# 第一次前向传播（prefill）
outputs = model(input_ids, use_cache=True, return_dict=True)

# 获取第一个生成的 token
next_token_logits = outputs.logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

# 记录 TTFT（Time To First Token）
ttft = time.perf_counter() - first_start
```

#### 3. KV Cache 处理（Baseline 无压缩）
```python
# 获取 KV cache
past_key_values = outputs.past_key_values

# Baseline: compress_fn 为 None，不进行任何压缩
if compress_fn is not None and past_key_values is not None:
    # 压缩 KV cache（baseline 不会执行这里）
    compressed_kv = compress_fn(kv_list, skip_layers=skip_layers, **compress_kwargs)
    past_key_values = to_dynamic_cache(compressed_kv)
else:
    # Baseline 直接使用原始 KV cache，不做任何处理
    pass
```

#### 4. 生成阶段（测量 TPOT 和 Throughput）
```python
# 逐个生成后续 tokens
for _ in range(max_new_tokens - 1):
    outputs = model(
        next_token,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True
    )
    
    # 获取下一个 token
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens.append(next_token)
    
    # 更新 KV cache（baseline 仍然不压缩）
    past_key_values = outputs.past_key_values
    # Baseline: compress_fn 为 None，past_key_values 保持不变
```

#### 5. 指标计算
```python
total_time = time.perf_counter() - total_start
num_tokens = len(generated_tokens)

# 计算指标
tpot = (total_time - ttft) / (num_tokens - 1)  # 每个输出 token 的平均时间
throughput = num_tokens / total_time              # tokens/秒
```

### Baseline 测量的指标

| 指标 | 说明 | 测量方式 |
|------|------|---------|
| **TTFT** | Time To First Token | Prefill 阶段的时间 |
| **TPOT** | Time Per Output Token | (总时间 - TTFT) / (生成 tokens 数 - 1) |
| **Throughput** | 吞吐量 | 生成 tokens 数 / 总时间 |
| **PPL** | Perplexity（困惑度） | 在评估文本上计算负对数似然 |
| **Accuracy** | 准确率 | Next token 预测准确率 |
| **Cache Size** | KV Cache 大小 | 等于输入序列长度（无压缩） |

### Baseline 的特殊性

1. **无内存限制**：KV cache 会随着序列长度线性增长
2. **最佳质量**：理论上提供最高的生成质量（PPL 最低，Accuracy 最高）
3. **速度参考**：作为速度的上界（无压缩开销）
4. **对比基准**：所有压缩方法都与之对比，计算性能损失

---

## 📈 实验对比逻辑

### 对比指标

在 `scripts/benchmark.py` 的 `main()` 函数中，会计算每个实验组相对于 baseline 的变化：

```python
# 计算相对 baseline 的变化百分比
throughput_imp = (avg_throughput / baseline_throughput - 1) * 100  # 吞吐量提升
tpot_imp = (1 - avg_tpot / baseline_tpot) * 100                     # TPOT 降低
ppl_change = (avg_ppl / baseline_ppl - 1) * 100                     # PPL 变化
acc_change = (avg_acc / baseline_acc - 1) * 100                     # 准确率变化
```

### 输出示例

```
Comparison with baseline (Throughput ↑ better, TPOT ↓ better, PPL ↓ better):
  recent_only_256: Throughput -15.2%, TPOT +18.5%, PPL +12.3%, Acc -3.1%
  streaming_256: Throughput -8.5%, TPOT +9.2%, PPL +5.1%, Acc -1.2%
  streaming_512: Throughput -3.2%, TPOT +3.5%, PPL +2.1%, Acc -0.5%
  streaming_1024: Throughput -1.1%, TPOT +1.2%, PPL +0.8%, Acc -0.2%
```

---

## 🎯 实验设计意图

### 为什么需要 Baseline？

1. **性能上界**：提供无压缩情况下的最佳性能参考
2. **质量基准**：作为生成质量（PPL, Accuracy）的上界
3. **对比标准**：所有压缩方法都与之对比，量化性能损失

### 为什么需要 Recent-Only 控制组？

1. **验证 Attention Sinks 的重要性**：对比 StreamingLLM 和纯滑动窗口
2. **公平对比**：确保在相同的 KV cache 大小下对比
3. **方法验证**：证明保留 attention sinks 的价值

### 为什么测试多个 Cache 大小？

1. **权衡分析**：探索速度 vs 质量的权衡曲线
2. **最优配置**：找到在可接受质量损失下的最佳 cache 大小
3. **实用性**：不同场景可能需要不同的 cache 大小

---

## 📝 使用示例

### 完整实验（包含所有组）

```bash
python scripts/benchmark.py \
    --method streaming_llm \
    --start_size 4 \
    --recent_sizes 252,508,1020 \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3 \
    --max_tokens 2000
```

### 只测试 StreamingLLM（不包含 recent_only）

```bash
python scripts/benchmark.py \
    --method streaming_llm \
    --start_size 4 \
    --recent_sizes 252,508,1020 \
    --no_recent_only \
    --model_id /mnt/disk1/models/pythia-2.8b
```

### 不包含 Baseline（只对比压缩方法）

```bash
python scripts/benchmark.py \
    --method streaming_llm \
    --start_size 4 \
    --recent_sizes 252,508,1020 \
    --no_baseline \
    --model_id /mnt/disk1/models/pythia-2.8b
```

---

## 🔍 关键代码位置

1. **实验组配置**：`scripts/benchmark.py` → `build_methods_config()` (line 284-310)
2. **Baseline 定义**：`scripts/benchmark.py` → `build_methods_config()` (line 232-236)
3. **指标测量**：`kvcompress/benchmark.py` → `measure_generation_metrics()` (line 23-150)
4. **对比计算**：`scripts/benchmark.py` → `main()` (line 549-567)

---

*文档生成时间: 2024*


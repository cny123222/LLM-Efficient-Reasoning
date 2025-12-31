# Benchmark 脚本使用示例

快速开始使用更新后的 benchmark.py 脚本，包含 VRAM 测量、结果保存和可视化功能。

---

## 🚀 快速开始

### 1. 激活虚拟环境

```bash
cd /mnt/disk1/ljm/LLM-Efficient-Reasoning
source ../llm-inference/bin/activate
```

### 2. 运行 Benchmark

```bash
# 测试 StreamingLLM 方法
python scripts/benchmark.py \
    --method streaming_llm \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3 \
    --max_tokens 2000
```

### 3. 查看结果

结果会自动保存到时间戳命名的目录：

```bash
# 结果目录示例
results/streaming_llm_pythia-2.8b_20241230_153045/
├── results.json              # 完整的实验数据
└── benchmark_comparison.png  # 对比图表（包含 VRAM）
```

---

## 📊 新增功能

### 1. VRAM Usage 测量

**自动测量峰值显存占用**，无需额外配置：

```bash
# 控制台输出示例（新增 VRAM 列）
Method                    TTFT(s)    TPOT(s)    Thruput        PPL        Acc   VRAM(GB)    Cache
----------------------------------------------------------------------------------------------------
baseline                   0.0123     0.0045     156.78      42.35     35.67%       5.23     2000
streaming_256              0.0098     0.0052     142.31      45.12     34.89%       3.21      256
streaming_512              0.0105     0.0048     148.56      43.67     35.23%       4.12      512
streaming_1024             0.0115     0.0046     152.34      42.89     35.45%       4.98     1024
```

### 2. 自动保存结果

**每次运行都会创建独立的结果目录**，包含：

#### results.json
完整的实验数据，包括：
- 配置信息（模型、参数等）
- 每个样本的原始结果
- 聚合统计数据
- Baseline 对比数据

#### benchmark_comparison.png
6 个关键指标的对比图：

| 指标 1 | 指标 2 | 指标 3 |
|--------|--------|--------|
| Throughput | TPOT | TTFT |
| Perplexity | **VRAM Usage** ⭐ | Cache Size |

---

## 🎯 常用命令

### 指定 GPU 运行

使用 `CUDA_VISIBLE_DEVICES` 环境变量指定 GPU：

```bash
# 在 GPU 0 上运行（默认）
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py --method streaming_llm ...

# 在 GPU 7 上运行
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py --method streaming_llm ...

# 使用多个 GPU（例如 GPU 6 和 7）
CUDA_VISIBLE_DEVICES=6,7 python scripts/benchmark.py --method streaming_llm ...
```

### 测试 StreamingLLM

```bash
# 基本用法
python scripts/benchmark.py \
    --method streaming_llm \
    --start_size 4 \
    --recent_sizes 252,508,1020 \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3

# 在指定 GPU (GPU 7) 上运行
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py \
    --method streaming_llm \
    --start_size 4 \
    --recent_sizes 252,508,1020 \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3
```

### 测试 Fix-Size L2

```bash
# 基本用法
python scripts/benchmark.py \
    --method fix_size_l2 \
    --fix_kv_sizes 256,512 \
    --strategies keep_low \
    --keep_ratios 0.5,0.7 \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3

# 在 GPU 7 上运行
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py \
    --method fix_size_l2 \
    --fix_kv_sizes 256,512 \
    --strategies keep_low \
    --keep_ratios 0.5,0.7 \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3
```

### 对比所有方法

```bash
# 基本用法
python scripts/benchmark.py \
    --compare_all \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3

# 在 GPU 7 上运行
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py \
    --compare_all \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3
```

### 不包含控制组（加速测试）

```bash
python scripts/benchmark.py \
    --method streaming_llm \
    --no_recent_only \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 2
```

---

## 📈 结果分析

### 查看 JSON 结果

```bash
# 使用 jq 美化输出
cat results/streaming_llm_pythia-2.8b_*/results.json | jq '.aggregated_stats'

# 提取特定方法的 VRAM 数据
cat results/streaming_llm_pythia-2.8b_*/results.json | jq '.aggregated_stats.streaming_512.peak_vram_gb'
```

### 查看对比图

```bash
# 使用图片查看器
eog results/streaming_llm_pythia-2.8b_*/benchmark_comparison.png

# 或者复制到本地查看
scp user@server:/path/to/results/*/benchmark_comparison.png ./
```

---

## 🔍 VRAM 分析

### 预期 VRAM 占用

| 方法 | Cache 大小 | 预期 VRAM | 相比 Baseline |
|------|-----------|-----------|--------------|
| Baseline | 无限制 | 最高 | 100% (基准) |
| streaming_256 | 256 | 较低 | 约 60-70% |
| streaming_512 | 512 | 中等 | 约 75-85% |
| streaming_1024 | 1024 | 较高 | 约 90-95% |

### VRAM 节省效果

```python
# 计算 VRAM 节省百分比
vram_saving = (baseline_vram - method_vram) / baseline_vram * 100

# 示例：
# baseline_vram = 5.23 GB
# streaming_512_vram = 4.12 GB
# 节省 = (5.23 - 4.12) / 5.23 * 100 = 21.2%
```

---

## 🎨 图表说明

### benchmark_comparison.png

**布局**：2 行 × 3 列

**指标说明**：

1. **Throughput** (左上) - 吞吐量，越高越好
   - 单位：tokens/second
   - 反映生成速度

2. **TPOT** (中上) - 每个输出 token 的时间，越低越好
   - 单位：milliseconds
   - 反映解码效率

3. **TTFT** (右上) - 首个 token 时间，越低越好
   - 单位：milliseconds
   - 反映 prefill 速度

4. **Perplexity** (左下) - 困惑度，越低越好
   - 反映生成质量
   - 与 baseline 越接近越好

5. **VRAM Usage** (中下) ⭐ - 峰值显存占用，越低越好
   - 单位：GB
   - 反映内存效率

6. **Cache Size** (右下) - KV cache 大小
   - 单位：tokens
   - 反映内存占用

**颜色编码**：
- 🟦 灰色：baseline（无压缩）
- 🟩 绿色：streaming_* 方法
- 🟥 红色：recent_only_* 控制组
- 🟦 蓝色：其他压缩方法

---

## 📁 结果目录结构

```
results/
├── streaming_llm_pythia-2.8b_20241230_153045/
│   ├── results.json
│   └── benchmark_comparison.png
├── fix_size_l2_pythia-2.8b_20241230_160230/
│   ├── results.json
│   └── benchmark_comparison.png
└── compare_all_pythia-2.8b_20241230_170530/
    ├── results.json
    └── benchmark_comparison.png
```

**命名规则**：`{method}_{model_name}_{timestamp}`

---

## 🖥️ GPU 选择说明

### CUDA_VISIBLE_DEVICES 环境变量

控制脚本使用哪个 GPU：

```bash
# 查看可用的 GPU
nvidia-smi

# 使用单个 GPU
CUDA_VISIBLE_DEVICES=0  # 使用 GPU 0
CUDA_VISIBLE_DEVICES=7  # 使用 GPU 7

# 使用多个 GPU（脚本会使用第一个）
CUDA_VISIBLE_DEVICES=6,7  # 可见 GPU 6 和 7，脚本使用 GPU 6

# 不使用 GPU（仅 CPU）
CUDA_VISIBLE_DEVICES=""
```

### 常见场景

```bash
# 场景 1: 服务器有多个 GPU，想使用空闲的 GPU 7
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py --method streaming_llm ...

# 场景 2: 确保使用特定 GPU 并后台运行
nohup env CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py \
    --method streaming_llm \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 5 > benchmark.log 2>&1 &

# 场景 3: 验证使用的 GPU
CUDA_VISIBLE_DEVICES=7 python -c "import torch; print(f'Using GPU: {torch.cuda.current_device()}')"
```

### 注意事项

1. **GPU 编号从 0 开始**：GPU 7 是第 8 块 GPU
2. **检查 GPU 可用性**：运行前先用 `nvidia-smi` 查看 GPU 状态
3. **显存占用**：确保目标 GPU 有足够的显存（pythia-2.8b 约需 6-8 GB）
4. **多 GPU 场景**：脚本默认只使用单 GPU，设置多个 GPU 时只会用第一个

---

## 💡 提示

### 1. 快速测试

```bash
# 使用较少样本快速测试
python scripts/benchmark.py \
    --method streaming_llm \
    --num_samples 1 \
    --max_tokens 1000 \
    --model_id /mnt/disk1/models/pythia-2.8b

# 在 GPU 7 上快速测试
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py \
    --method streaming_llm \
    --num_samples 1 \
    --max_tokens 1000 \
    --model_id /mnt/disk1/models/pythia-2.8b
```

### 2. 完整实验

```bash
# 使用更多样本获得稳定结果
python scripts/benchmark.py \
    --method streaming_llm \
    --num_samples 5 \
    --max_tokens 3000 \
    --model_id /mnt/disk1/models/pythia-2.8b

# 在 GPU 7 上运行完整实验
CUDA_VISIBLE_DEVICES=7 python scripts/benchmark.py \
    --method streaming_llm \
    --num_samples 5 \
    --max_tokens 3000 \
    --model_id /mnt/disk1/models/pythia-2.8b
```

### 3. 查看历史结果

```bash
# 列出所有结果目录
ls -lt results/

# 查看最新的结果
ls -t results/ | head -1

# 查看最新结果的 JSON
cat results/$(ls -t results/ | head -1)/results.json | jq
```

---

## 🔗 相关文档

- [BENCHMARK_UPDATES.md](./docs/BENCHMARK_UPDATES.md) - 详细的更新说明
- [STREAMINGLLM_BENCHMARK_CONFIG.md](./docs/STREAMINGLLM_BENCHMARK_CONFIG.md) - StreamingLLM 配置说明
- [README.md](./README.md) - 项目总体说明

---

*快速参考指南*
*版本: v1.1*
*更新日期: 2024-12-30*


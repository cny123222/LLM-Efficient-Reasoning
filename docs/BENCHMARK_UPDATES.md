# Benchmark 脚本更新说明

本文档说明 benchmark.py 脚本的最新更新内容，包括 VRAM 测量、结果保存和可视化功能。

---

## 🎯 更新内容

### 1. VRAM Usage 测量

#### 修改文件：`kvcompress/benchmark.py`

**新增功能**：
- 在生成过程中自动测量峰值显存占用
- 支持 CUDA 设备的显存监控
- 返回结果中包含 `peak_vram_gb` 字段（单位：GB）

**实现细节**：
```python
# 在生成开始前重置显存统计
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

# 在生成结束后获取峰值显存
peak_memory_bytes = torch.cuda.max_memory_allocated(device)
peak_memory_gb = peak_memory_bytes / (1024 ** 3)
```

**返回值**：
- 新增字段：`peak_vram_gb` - 峰值显存占用（GB）

---

### 2. 结果保存功能

#### 修改文件：`scripts/benchmark.py`

**新增功能**：
- 自动创建时间戳命名的结果目录
- 保存完整的 JSON 格式结果
- 目录命名格式：`{method}_{model}_{timestamp}`

**目录结构**：
```
results/
├── streaming_llm_pythia-2.8b_20241230_153045/
│   ├── results.json              # 完整的实验结果
│   └── benchmark_comparison.png  # 对比图表
├── fix_size_l2_pythia-2.8b_20241230_160230/
│   ├── results.json
│   └── benchmark_comparison.png
└── ...
```

**JSON 格式**：
```json
{
  "config": {
    "model_id": "/mnt/disk1/models/pythia-2.8b",
    "method": "streaming_llm",
    "num_samples": 3,
    "max_tokens": 2000,
    "max_new_tokens": 500,
    "skip_layers": [0, 1],
    "timestamp": "20241230_153045"
  },
  "raw_results": [
    {
      "method": "baseline",
      "ttft": 0.0123,
      "tpot": 0.0045,
      "throughput": 156.78,
      "perplexity": 42.35,
      "accuracy": 0.3567,
      "peak_vram_gb": 5.23,
      "final_cache_size": 2000
    },
    ...
  ],
  "aggregated_stats": {
    "baseline": {
      "ttft": 0.0123,
      "tpot": 0.0045,
      "throughput": 156.78,
      "perplexity": 42.35,
      "accuracy": 0.3567,
      "peak_vram_gb": 5.23,
      "cache_size": 2000
    },
    "streaming_512": { ... }
  },
  "baseline_stats": {
    "perplexity": 42.35,
    "accuracy": 0.3567,
    "throughput": 156.78,
    "tpot": 0.0045
  }
}
```

---

### 3. 可视化对比图表

#### 新增功能：`plot_benchmark_results()`

**生成图表**：`benchmark_comparison.png`

**布局**：2 行 × 3 列，共 6 个子图

| 行 | 列 1 | 列 2 | 列 3 |
|----|------|------|------|
| 第 1 行 | Throughput<br>(吞吐量) | TPOT<br>(每 token 时间) | TTFT<br>(首 token 时间) |
| 第 2 行 | Perplexity<br>(困惑度) | **VRAM Usage**<br>(显存占用) | Cache Size<br>(缓存大小) |

**图表特点**：
- 颜色编码：
  - `baseline`: 灰色
  - `streaming_*`: 绿色
  - `recent_only_*`: 红色
  - 其他方法：蓝色
- 每个柱状图上标注具体数值
- 自动调整 Y 轴范围以突出差异
- 高分辨率输出（DPI=300）

**参考设计**：
参照 `precision/precision_benchmark_2rows.png` 的样式，采用 2 行布局展示 6 个关键指标。

---

## 📊 新增测量指标

### VRAM Usage (Peak Memory)

| 指标 | 说明 | 单位 | 测量方式 |
|------|------|------|---------|
| **Peak VRAM** | 峰值显存占用 | GB | `torch.cuda.max_memory_allocated()` |

**重要性**：
- 评估不同压缩方法的显存节省效果
- 对比 baseline 和压缩方法的内存开销
- 指导实际部署时的资源规划

**预期结果**：
- Baseline: 最高显存占用（KV cache 完整保留）
- StreamingLLM: 显存占用随 cache 大小固定
- Fix-Size: 显存占用可控（固定大小）

---

## 🚀 使用方式

### 基本用法

```bash
# 测试 StreamingLLM 方法
python scripts/benchmark.py \
    --method streaming_llm \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3 \
    --max_tokens 2000

# 结果会自动保存到：
# results/streaming_llm_pythia-2.8b_YYYYMMDD_HHMMSS/
```

### 输出说明

#### 1. 控制台输出（增强版）

```
Method                    TTFT(s)    TPOT(s)    Thruput        PPL        Acc   VRAM(GB)    Cache
----------------------------------------------------------------------------------------------------
baseline                   0.0123     0.0045     156.78      42.35     35.67%       5.23     2000
streaming_256              0.0098     0.0052     142.31      45.12     34.89%       3.21      256
streaming_512              0.0105     0.0048     148.56      43.67     35.23%       4.12      512
streaming_1024             0.0115     0.0046     152.34      42.89     35.45%       4.98     1024
====================================================================================================
```

**新增列**：`VRAM(GB)` - 显示每个方法的峰值显存占用

#### 2. JSON 结果文件

位置：`results/{method}_{model}_{timestamp}/results.json`

包含：
- 完整配置信息
- 每个样本的原始结果
- 聚合统计数据
- Baseline 对比数据

#### 3. 对比图表

位置：`results/{method}_{model}_{timestamp}/benchmark_comparison.png`

包含：
- 6 个关键指标的柱状图对比
- 颜色编码区分不同方法类型
- 数值标注便于读取

---

## 📈 示例输出

### 实验场景

```bash
python scripts/benchmark.py \
    --method streaming_llm \
    --start_size 4 \
    --recent_sizes 252,508,1020 \
    --model_id /mnt/disk1/models/pythia-2.8b \
    --num_samples 3
```

### 生成的文件

```
results/streaming_llm_pythia-2.8b_20241230_153045/
├── results.json              # 完整结果 JSON
└── benchmark_comparison.png  # 可视化对比图
```

### 对比图示例

图表展示 7 个实验组的对比：
1. **baseline** (灰色) - 无压缩
2. **recent_only_256** (红色) - 滑动窗口 256
3. **streaming_256** (绿色) - StreamingLLM 256
4. **recent_only_512** (红色) - 滑动窗口 512
5. **streaming_512** (绿色) - StreamingLLM 512
6. **recent_only_1024** (红色) - 滑动窗口 1024
7. **streaming_1024** (绿色) - StreamingLLM 1024

---

## 🔍 VRAM 测量详解

### 测量时机

```python
# 1. 重置统计（生成开始前）
torch.cuda.reset_peak_memory_stats(device)

# 2. 执行生成过程
# - Prefill 阶段
# - 逐 token 生成
# - KV cache 压缩（如果启用）

# 3. 获取峰值（生成结束后）
peak_memory = torch.cuda.max_memory_allocated(device)
```

### 测量范围

**包含**：
- 模型参数占用
- 输入数据占用
- KV cache 占用（压缩前后）
- 中间计算的临时内存
- 梯度内存（推理时为 0）

**不包含**：
- 模型加载前的其他显存占用
- 其他进程的显存占用

### 预期差异

| 方法 | 预期 VRAM | 相比 Baseline |
|------|-----------|--------------|
| Baseline | 最高 | 基准 (100%) |
| StreamingLLM 256 | 较低 | 约 60-70% |
| StreamingLLM 512 | 中等 | 约 75-85% |
| StreamingLLM 1024 | 较高 | 约 90-95% |

**公式估算**：
```
VRAM ≈ 模型大小 + KV cache 大小 + 临时计算内存

KV cache 大小 = (batch × layers × heads × seq_len × head_dim × 2) × 2 bytes
                ↑batch=1  ↑key+value                           ↑FP16
```

---

## 🎨 图表样式参考

参考 `precision/precision_benchmark_2rows.png` 的设计：
- 2 行 3 列布局
- 清晰的标题和坐标轴标签
- 柱状图上方标注数值
- 颜色区分不同方法类型
- 网格线辅助读数
- 高分辨率输出

---

## 📝 注意事项

### 1. VRAM 测量精度

- 只在 CUDA 设备上准确
- MPS 和 CPU 返回 0.0
- 包含所有 PyTorch 管理的显存

### 2. 结果目录管理

- 每次运行创建新目录
- 目录名包含时间戳，避免覆盖
- 建议定期清理旧结果

### 3. 图表生成

- 使用非交互式后端（`Agg`）
- 适合服务器环境运行
- 自动保存为高分辨率 PNG

### 4. 性能开销

- VRAM 测量开销极小（< 1%）
- JSON 保存和绘图在实验结束后进行
- 不影响实际测量结果

---

## 🔧 代码修改位置

### 1. `kvcompress/benchmark.py`

**修改函数**：`measure_generation_metrics()`

**修改内容**：
- Line ~75: 添加 `torch.cuda.reset_peak_memory_stats()`
- Line ~135: 添加 VRAM 测量逻辑
- Line ~145: 返回值增加 `peak_vram_gb` 字段

### 2. `scripts/benchmark.py`

**新增导入**：
```python
import json
from datetime import datetime
import matplotlib.pyplot as plt
```

**新增函数**：
- `save_results_to_json()` - 保存 JSON 结果
- `plot_benchmark_results()` - 生成对比图表

**修改内容**：
- Line ~480: 创建结果目录
- Line ~550: 打印输出增加 VRAM 列
- Line ~590: 保存 JSON 和生成图表

---

## 📚 相关文档

- [STREAMINGLLM_BENCHMARK_CONFIG.md](./STREAMINGLLM_BENCHMARK_CONFIG.md) - StreamingLLM 实验配置说明
- [模块功能说明.md](../模块功能说明.md) - precision 和 spec_decode 模块说明

---

*文档生成时间: 2024-12-30*
*更新版本: v1.1*


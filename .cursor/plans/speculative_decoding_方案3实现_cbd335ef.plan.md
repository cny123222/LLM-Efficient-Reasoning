---
name: Speculative Decoding 方案3实现
overview: 基于方案3（Draft 轮内用 cache，轮间丢弃）实现 Speculative Decoding，确保在 Pythia 模型上达到与 HuggingFace 相当的性能和正确性。
todos:
  - id: static-cache
    content: 实现 StaticKVCache 类（预分配 + 指针管理 + HF 格式转换）
    status: completed
  - id: generator-prefill
    content: 实现 _prefill() 方法初始化 target cache
    status: completed
    dependencies:
      - static-cache
  - id: generator-draft
    content: 实现 _draft_k_tokens() 使用临时 cache 生成 K tokens
    status: completed
    dependencies:
      - generator-prefill
  - id: generator-verify
    content: 实现 _verify_tokens() 和 _accept_reject_greedy()
    status: completed
    dependencies:
      - generator-draft
  - id: generator-update
    content: 实现 _update_target_cache() 正确截断
    status: completed
    dependencies:
      - generator-verify
  - id: generator-main
    content: 实现 generate() 主循环并应用 torch.compile
    status: completed
    dependencies:
      - generator-update
  - id: benchmark
    content: 实现对比脚本验证性能与 HuggingFace 匹配
    status: completed
    dependencies:
      - generator-main
  - id: test-correctness
    content: 验证输出正确性和显存无泄漏
    status: completed
    dependencies:
      - benchmark
---

# Speculative Decoding 详细实现方案

## 核心设计决策

| 决策点 | 选择 | 原因 ||--------|------|------|| 接受策略 | Greedy | 与现有代码一致，输出确定性 || Draft Cache | 轮内使用，轮间丢弃 | 简单正确，避免同步问题 || Target Cache | Static Cache + 持久化 | 关键优化点，零拷贝截断 || PyTorch 优化 | torch.compile + inference_mode | 减少 Python 开销 |

## 文件结构

```javascript
spec_decode/
├── core/
│   ├── __init__.py
│   ├── static_cache.py          # Static KV Cache 实现
│   ├── speculative_generator.py # 主生成器
│   └── utils.py                 # 工具函数
├── benchmark_custom_vs_hf.py    # 性能对比
├── test_correctness.py          # 正确性验证
└── demo.py                      # 演示脚本
```



## 详细实现步骤

### Step 1: Static KV Cache 实现

文件: `spec_decode/core/static_cache.py`

```python
class StaticKVCache:
    """
    预分配固定大小的 KV Cache
    
    特性:
                - 预分配 max_seq_len 显存，避免动态分配
                - truncate() 只移动指针，O(1) 操作
                - 支持 HuggingFace 模型的 past_key_values 格式
    """
```

关键方法:

- `__init__(config, max_seq_len)`: 根据模型配置预分配
- `update(new_keys, new_values)`: 追加新的 KV
- `truncate(new_len)`: 截断到指定长度（只移动指针）
- `to_hf_format()`: 转换为 HuggingFace 的 tuple 格式
- `from_hf_format(past_key_values)`: 从 HuggingFace 格式导入

### Step 2: Speculative Generator 核心实现

文件: `spec_decode/core/speculative_generator.py`

```python
class SpeculativeGenerator:
    """
    Speculative Decoding 生成器
    
    算法流程:
                1. Prefill: 处理 prompt，初始化 target cache
                2. Loop:
       a. Draft: 小模型生成 K 个 tokens（轮内用临时 cache）
       b. Verify: 大模型并行验证（更新 target cache）
       c. Accept: Greedy 接受/拒绝
       d. Update: 截断 target cache 到正确位置
    """
```

核心方法详细设计:

#### `__init__(target_model, draft_model, tokenizer, K, max_len)`

```javascript
- 加载模型到 GPU
- 应用 torch.compile 优化
- 初始化 Static Cache（仅 target）
- 设置 K 值和最大长度
```



#### `_prefill(input_ids) -> None`

```javascript
输入: prompt 的 token ids
操作:
        1. Target model forward（整个 prompt）
        2. 将返回的 past_key_values 写入 static cache
        3. 记录当前序列长度
```



#### `_draft_k_tokens() -> Tuple[Tensor, Tensor]`

```javascript
输出: (draft_tokens, draft_logits)
操作:
        1. Draft model prefill 当前序列（不用持久 cache）
        2. 创建临时 cache
        3. 循环 K 次:
                    - 从 logits 取 argmax
                    - 用临时 cache 生成下一个 token
        4. 返回 K 个 tokens 和对应的 logits
```



#### `_verify_tokens(draft_tokens) -> Tensor`

```javascript
输入: K 个 draft tokens
输出: K+1 个位置的 target logits
操作:
        1. 将 draft_tokens 喂给 target model
        2. 使用 static cache（从 past_key_values 转换）
        3. 获取所有位置的 logits
        4. 暂存新的 past_key_values（可能需要截断）
```



#### `_accept_reject_greedy(draft_tokens, target_logits) -> Tuple[Tensor, int]`

```javascript
输入: draft_tokens, target_logits
输出: (accepted_tokens, num_accepted)
操作:
  for i in range(K):
    target_pred = argmax(target_logits[i])
    if draft_tokens[i] == target_pred:
      接受
    else:
      用 target_pred 替换，停止
  
  if 全部接受:
    bonus_token = argmax(target_logits[K])
    num_accepted = K + 1
```



#### `_update_target_cache(num_accepted) -> None`

```javascript
操作:
        1. 计算新的 cache 长度 = original_len + num_accepted
        2. 调用 static_cache.truncate(new_len)
        3. 处理最后一个 token 的 KV（如果被替换）
```



#### `generate(prompt, max_new_tokens) -> str`

```javascript
主生成循环:
        1. tokenize prompt
        2. _prefill()
        3. while generated < max_new_tokens:
       draft_tokens, _ = _draft_k_tokens()
       target_logits = _verify_tokens(draft_tokens)
       accepted, num = _accept_reject_greedy()
       _update_target_cache(num)
       generated += num
       if eos: break
        4. decode and return
```



### Step 3: 性能对比脚本

文件: `spec_decode/benchmark_custom_vs_hf.py`测试内容:

1. **吞吐量对比**: tokens/second
2. **输出一致性**: 与 HuggingFace 输出完全相同
3. **不同 K 值**: K=3,5,7,10
4. **不同序列长度**: 短/中/长

### Step 4: 正确性验证

文件: `spec_decode/test_correctness.py`验证点:

1. Greedy 输出与纯 Target 模型一致
2. 接受率统计合理（应该 > 0.5）
3. Cache 长度正确
4. 无显存泄漏

## 关键代码片段预览

### Static Cache 核心逻辑

```python
class StaticKVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_len, dtype, device):
        self.max_len = max_len
        self.current_len = 0
        
        # 预分配: [num_layers, batch=1, num_heads, max_len, head_dim]
        self.keys = torch.zeros(
            (num_layers, 1, num_heads, max_len, head_dim),
            dtype=dtype, device=device
        )
        self.values = torch.zeros_like(self.keys)
    
    def truncate(self, new_len):
        """O(1) 截断 - 只移动指针"""
        self.current_len = min(new_len, self.current_len)
    
    def to_hf_format(self):
        """转换为 HuggingFace past_key_values 格式"""
        return tuple(
            (self.keys[i, :, :, :self.current_len, :],
             self.values[i, :, :, :self.current_len, :])
            for i in range(self.keys.shape[0])
        )
```



### 主生成循环

```python
@torch.inference_mode()
def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    self._prefill(input_ids)
    
    generated = 0
    while generated < max_new_tokens:
        # Draft: 轮内用临时 cache
        draft_tokens = self._draft_k_tokens()
        
        # Verify: 用 static cache
        target_logits = self._verify_tokens(draft_tokens)
        
        # Accept/Reject
        accepted, num_accepted = self._accept_reject_greedy(draft_tokens, target_logits)
        
        # Update
        self._update_target_cache(num_accepted)
        self.current_ids = torch.cat([self.current_ids, accepted], dim=-1)
        generated += num_accepted
        
        if accepted[0, -1] == self.tokenizer.eos_token_id:
            break
    
    return self.tokenizer.decode(self.current_ids[0], skip_special_tokens=True)
```



## 预期性能

| 指标 | HuggingFace | 我的实现 ||------|-------------|---------|| 吞吐量 (K=5) | ~40 t/s | ~36-44 t/s (90-110%) || 接受率 | ~0.6-0.8 | ~0.6-0.8 (相同) || 显存占用 | 动态增长 | 固定预分配 || 输出正确性 | - | 100% 一致 |

## 实施顺序

```mermaid
flowchart LR
    A[Step1: StaticKVCache] --> B[Step2: Generator核心]
    B --> C[Step3: 性能对比]
    C --> D[Step4: 正确性验证]
    D --> E[Step5: 优化调优]
```
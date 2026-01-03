# Figure 1（Architecture Overview / Fig.~\ref{fig:arch}）中文画图说明书

> 目标：让读者 **在 10 秒内**看懂 DynaTree 一次解码迭代（iteration）的完整流水线：  
> **Draft（草稿树生成）→ Prune（动态剪枝）→ Flatten+Mask（展平与树注意力掩码）→ Verify（目标模型并行验证）→ Select/Commit（选择并提交）→ KV Cache 更新 → 下一轮**。  
> 这张图是“系统总览图”，不要塞公式细节；树注意力 mask 的细节放到 Fig.2。

---

## 1. 画图推荐工具与导出规范

- **推荐工具**：Figma（首选）或 draw.io（diagrams.net）。  
- **导出格式**：优先 **PDF（矢量）**；也可 SVG 再转 PDF。  
- **画布建议**：横向（Landscape），适配论文单栏宽度（NeurIPS 常用 `\linewidth`）。  
- **风格建议**：
  - 颜色只用 2–3 个主色 + 灰色（避免花）。
  - 箭头方向统一从左到右（主流程），循环回到左侧用虚线或弯箭头。
  - 字体用无衬线（Inter/Helvetica），字号保证缩到单栏仍清晰。

---

## 2. 这张图必须包含的“6 个模块 + 2 个辅助元素”

建议把主流程画成一条从左到右的 pipeline（每个模块用圆角矩形框，框间用粗箭头连接）。

### 模块 (1) 输入：Prefix + KV Cache

**你要画什么：**
- 一个小框：`Prefix / Prompt  x_{1:t}`
- 旁边一个小“堆叠/数据库”图标或小框：`KV cache (prefix)`

**你要表达什么：**
- 我们不是从头跑模型，而是在已有 prefix 上增量生成。

**建议标注（图中文字）：**
- `x_{1:t}`  
- `KV cache`

---

### 模块 (2) 草稿树生成：Draft model 生成 token tree

**你要画什么：**
- 一个大框：`Draft model  M_D`
- 框内画一棵 **小树**（只要 2–3 层即可，不要画很深）：
  - 根节点代表“下一 token 的候选集合”
  - 每层分叉体现 `branch factor B`
  - 深度体现 `depth D`

**树里 token 写什么：**
- 不要写真实词，写符号即可，例如：`a1, a2, a3` / `b1, b2` / `c1,...`  
  让读者看到“一个节点会扩展多个孩子”就够了。

**建议标注：**
- `top-B expansion`  
- `depth D`

---

### 模块 (3) 动态剪枝：Adaptive pruning（你们的关键点之一）

**你要画什么：**
- 一个框：`Adaptive pruning (ours)`
- 在上一模块的树上画“剪掉/变灰/打叉”的分支（被剪枝的候选）。

**剪枝规则要怎么写（非常重要）：**
在这个框里用两行短字说明剪枝依据（不要公式推导）：
- `prob. threshold  τ`（概率阈值剪枝）
- `node budget  N_max`（节点总数上限/预算）

**你要表达什么：**
- 树不会无限膨胀，验证开销是可控的。

---

### 模块 (4) 展平 + 树注意力 mask：Flatten + Tree Attention Mask（点到为止）

**你要画什么：**
这个模块建议拆成两个小子元素放在同一个大框里：

1) **Flatten（BFS 展平）**：画一个小条形列表：
- `Flattened nodes (BFS order): [u0, u1, u2, ...]`

2) **Mask（注意力掩码示意）**：画一个很小的矩阵格子（例如 4×4）：
- 黑格表示“允许注意”（可见）
- 白格表示“禁止注意”（不可见）
- 在矩阵旁写一句话：`tree causal mask`

**你要表达什么：**
- 我们把“树结构”编码到 attention mask 中，从而让目标模型能 **一次并行验证整棵树**。

> 注意：mask 的具体规则（祖先可见、兄弟不可见等）不要在 Fig1 解释太多，留给 Fig2。

---

### 模块 (5) 目标模型并行验证：Target model 一次 forward 验证所有节点

**你要画什么：**
- 一个大框：`Target model  M_T`
- 箭头输入写清楚：`prefix + flattened tree tokens + mask`
- 输出画成一条“多位置输出”的条带或小框：`logits for all nodes` 或 `verified signals`

**你要表达什么：**
- 关键加速点：目标模型只做 **一次 forward**，就能拿到树上所有候选位置的 logits（并行验证）。

---

### 模块 (6) 选择 + 提交：Greedy Path Selection & Commit

**你要画什么：**
- 一个框：`Greedy path selection`
- 在树上用高亮颜色把“被接受的最长路径”描粗（例如从根到某个叶子的一条路径）。
- 旁边写：`commit L tokens (+1 bonus)`

**你要表达什么：**
- 我们不会“随便接受”，而是选一条与目标模型 greedy 一致的最长有效路径，把这些 token 一次性提交（减少迭代次数）。

---

## 3. 两个必须出现的“辅助元素”

### 辅助元素 A：KV Cache 更新（正确性/工程关键点）

**你要画什么：**
在模块 (6) 旁边加一个小框或注释：
- `KV cache update: crop / rebuild for committed tokens`

**你要表达什么：**
- 提交后需要把 cache 与新 token 对齐，保证下一轮继续正确解码。

---

### 辅助元素 B：循环到下一轮（表明这是迭代算法）

**你要画什么：**
- 从 `commit` 画一条 **弯箭头/虚线箭头** 回到最左侧 `Prefix`：
  - 标注：`t ← t + L` 或 `next iteration`

---

## 4. 推荐版式（你可以照抄）

从左到右 6 个大块：

1. `Prefix + KV cache`
2. `Draft model M_D (build draft tree)`
3. `Adaptive pruning (τ, N_max)`
4. `Flatten (BFS) + tree attention mask`
5. `Target model M_T (parallel verify, 1 forward)`
6. `Greedy path selection + commit`

然后在 (6) 旁边放：
- `KV cache update`
并用弯箭头回到 (1)。

---

## 5. 图里每个符号/参数怎么写（避免写错）

- `M_D`：draft model（小模型，负责提案/扩展）
- `M_T`：target model（大模型，负责验证/最终输出）
- `D`：tree depth（树深）
- `B`：branch factor（每个节点保留 top-B 候选）
- `τ`：probability threshold（概率阈值剪枝）
- `N_max`：node budget（节点预算/上限）
- `L`：本轮提交的 token 数（被接受路径长度）

---

## 6. 常见误区（务必避免）

- 不要把 Fig1 画成“代码流程图”（太细、看不懂）。
- 不要在 Fig1 里放 attention mask 的严格定义（会挤爆）。
- 不要画很深的树（2–3 层足够表达“树”）。
- 不要在图里写实验配置（例如 Pythia-2.8B/70M、GPU 型号）——这属于表格/实验部分。
- 不要出现文件名、json、脚本路径等工程细节。

---

## 7. 你画完后自检（5 秒检查清单）

读者能否一眼回答：
- 输入是什么？（prefix+cache）
- 谁生成树？（M_D）
- 树为什么不会爆？（τ + N_max）
- 为什么能并行验证？（flatten + tree mask + M_T 一次 forward）
- 最终输出怎么保证是 greedy 的？（greedy path selection + commit）
- 下一轮怎么继续？（cache update + loop）

---

## 8. 可选增强（如果你时间够）

- 在模块 (5) 上方加一句小字：`1 target forward per iteration`
- 在剪枝模块旁加一个小对比标注：`without pruning → too many nodes`
- 用两种颜色区分“保留分支”和“被剪掉分支”



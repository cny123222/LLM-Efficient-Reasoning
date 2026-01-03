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

## 1.5 你可以“直接照抄”的最终版布局模板（最重要）

这一节不讲原理，只讲 **怎么摆、画什么框、写什么字、箭头怎么连**。你完全可以照着做。

### A. 画布与网格（Figma 里先把这个搭好）

- **画布尺寸**：1600 × 900 px（横向）。  
  - 说明：这个尺寸导出 PDF 很清晰，缩到 NeurIPS 单栏也不会糊。
- **外边距**：四周留 60 px 空白。
- **主流程区域**：从左到右排 6 个大模块，**每个模块同高**。

### B. 全局样式（所有模块统一用这一套）

下面这些参数，你在 Figma 里设置一次，后面复制粘贴即可：

- **模块外框形状**：圆角矩形（Rounded Rectangle）
  - 宽：220 px
  - 高：300 px
  - 圆角：16 px
  - 描边（Stroke）：2 px
  - 描边颜色：`#2F2F2F`
  - 填充（Fill）：浅色背景（见下方配色表）
- **模块标题字体**：
  - 字体：Inter / Helvetica
  - 字号：18
  - 字重：600
  - 颜色：`#111111`
- **模块正文字体**：
  - 字号：14
  - 字重：400
  - 行距：1.25
  - 颜色：`#222222`
- **箭头（主流程）**：
  - 线宽：3 px
  - 颜色：`#2F2F2F`
  - 箭头：Filled triangle
  - 线型：实线
- **箭头（循环回到下一轮）**：
  - 线宽：3 px
  - 颜色：`#6B7280`（灰）
  - 线型：虚线（Dash 8 / Gap 6）
  - 箭头：Filled triangle

### C. 推荐配色（你按这个上色就不会丑）

> 只用 3 种主色：Draft 蓝、Prune 橙、Verify 紫，其余用灰。

- **(1) Prefix/KV**：填充 `#F3F4F6`（浅灰）
- **(2) Draft (M_D)**：填充 `#E8F1FF`（浅蓝）
- **(3) Pruning (ours)**：填充 `#FFF4E6`（浅橙）
- **(4) Flatten+Mask**：填充 `#F1F5F9`（偏冷浅灰）
- **(5) Verify (M_T)**：填充 `#F3E8FF`（浅紫）
- **(6) Select/Commit**：填充 `#ECFDF5`（浅绿）

### D. 最终排版草图（你照这个“位置关系”摆）

每个模块宽 220、高 300；模块之间水平间距 30。

```
┌────────┐  →  ┌────────┐  →  ┌────────┐  →  ┌────────┐  →  ┌────────┐  →  ┌────────┐
│  (1)   │     │  (2)   │     │  (3)   │     │  (4)   │     │  (5)   │     │  (6)   │
│Prefix  │     │ Draft  │     │ Prune  │     │Flatten │     │ Verify │     │Commit  │
│ + KV   │     │  M_D   │     │ τ,Nmax │     │ +Mask  │     │  M_T   │     │ + KV   │
└────────┘     └────────┘     └────────┘     └────────┘     └────────┘     └────────┘
      ^                                                                                |
      |--------------------  dashed arrow: next iteration  (t ← t + L)  ---------------|
```

### E. “每个模块里面画什么”的逐字模板（直接复制到框里）

你在每个模块里都放：标题（上方）+ 内容（中间）+ 小注释（下方）。下面给你 **逐字**内容。

#### 模块 (1) Prefix + KV

- **标题**：`Prefix`
- **内容（两行）**：
  - `Prompt / prefix  x_{1:t}`
  - `KV cache (prefix)`
- **图形元素**：
  - 左侧画一个小条形文本框表示 prefix。
  - 右侧画一个“堆叠”小图标（3 层小矩形叠起来）表示 KV cache。

#### 模块 (2) Draft model：Tree drafting

- **标题**：`Draft model  M_D`
- **内容（三行）**：
  - `Tree drafting`
  - `top-B expansion`
  - `depth D`
- **图形元素**：
  - 在框中央画一棵 3 层小树：
    - 根节点写 `u0`
    - 第一层写 `u1 u2 u3`
    - 第二层只画 2–3 个叶子（不要画满），写 `u4 u5 u6`
  - 线条用灰色 `#6B7280`，节点用白底小圆（直径 18–22px）。

#### 模块 (3) Adaptive pruning（ours）

- **标题**：`Adaptive pruning`
- **内容（三行）**：
  - `prob. threshold  τ`
  - `node budget  N_max`
  - `prune low-prob branches`
- **图形元素**：
  - 从模块 (2) 的树复制一棵过来，**把 1–2 条分支变灰**（`#CBD5E1`），末端画一个小叉 `×` 或剪刀图标。
  - 保留路径用正常灰线 `#6B7280`。

#### 模块 (4) Flatten + tree mask

- **标题**：`Flatten + mask`
- **内容（两行）**：
  - `BFS flatten: [u0,u1,u2,...]`
  - `tree causal mask`
- **图形元素（两个并排小组件）**：
  1) 左边画一个小“列表条”（3–4 个小矩形堆成列表），旁边写 `u0,u1,u2...`
  2) 右边画一个 4×4 小方格矩阵：
     - 祖先可见的格子涂深色 `#111827`
     - 其余空白或浅灰 `#E5E7EB`
  - 注意：这里的 mask 只要看起来像“上三角 + 路径块”，不用严格正确。

#### 模块 (5) Target model：Parallel verification

- **标题**：`Target model  M_T`
- **内容（三行）**：
  - `Verify all nodes`
  - `one forward pass`
  - `logits / accept signals`
- **图形元素**：
  - 画一个“网络/Transformer”简图：用 3–4 层小矩形叠起来即可，写 `Transformer`.
  - 输出端画一条小条带（横条），分成 4–6 格表示 “logits for nodes”。

#### 模块 (6) Greedy select + Commit

- **标题**：`Select & commit`
- **内容（三行）**：
  - `longest valid path`
  - `commit L tokens (+1)`
  - `update KV cache`
- **图形元素**：
  - 画一棵小树（或从模块 2 复制），用**亮色描粗一条路径**：
    - 路径颜色：`#16A34A`（绿）
    - 线宽：4 px
  - 在路径末端加一个小标签：`L`
  - 右下角放一个小“cache”图标，旁边写 `crop / rebuild`.

### F. 箭头连接表（你按这个连，绝不会连错）

- 主流程实线箭头：
  - (1) → (2)
  - (2) → (3)
  - (3) → (4)
  - (4) → (5)
  - (5) → (6)
- 循环虚线箭头：
  - (6) → (1)，标注：`next iteration:  t ← t + L`

### G. Figma 逐步操作清单（你按步骤点就行）

1) 新建 Frame：1600×900，背景白色。  
2) 画一个模块框（220×300，圆角 16，描边 2），设定好字体与颜色。  
3) 复制这个框 6 次，水平排列：间距 30，对齐顶边（Align top）。  
4) 分别给 6 个框填充颜色（见上面的配色表）。  
5) 在每个框顶部放标题文字（18/600），中间放正文（14/400）。  
6) 先把模块 (2) 的小树画出来（圆点 + 线），然后复制到 (3) 和 (6)。  
7) 在 (3) 把 1–2 条分支改浅灰并加 `×`。  
8) 在 (4) 画“列表条 + 4×4 小矩阵”。  
9) 画主流程箭头：用 Line 工具画 5 条实线箭头连接模块。  
10) 画循环箭头：从 (6) 底部绕到 (1) 底部，设置虚线，写 `t ← t + L`。  
11) 检查缩放到 50% 时字仍清晰（不清晰就把正文字号调到 15）。  
12) 导出：选中整个 Frame → Export → PDF（矢量）。

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



### **CS2602 语言模型高效推理-个人作业需求文档**

**版本**: 1.0
**日期**: 2023-10-27

#### 1. 项目概述 (Project Overview)

本项目旨在通过**非训练式方法**，实现并评估一种或多种大型语言模型（LLM）的推理加速技术。项目的核心是**动手实践**，在不改变模型参数的前提下，通过优化推理过程（特别是对KV Cache的管理）来提升模型的推理速度并降低显存占用。

最终产出为一个公开的、可复现的GitHub仓库，其中包含你的算法实现、性能评测结果以及一份简短的报告。

#### 2. 核心要求 (Core Requirements)

根据作业要求，你的个人项目必须满足以下条件：

1.  **创建个人独立的公开GitHub仓库**:
    *   仓库命名清晰，例如 `CS2602-LLM-Inference-Acceleration`。
    *   包含一个内容详尽的 `README.md` 文件。

2.  **算法实践**:
    *   从课程提供的参考方法中（如StreamingLLM, SnapKV, PyramidKV, H2O等）**选择并复现**至少一种加速优化算法。
    *   所有代码实现应包含在仓库中，并建议有清晰的注释。

3.  **`README.md` 文件要求**:
    *   **如何运行你的代码**: 提供清晰的指令，包括环境设置、依赖安装和启动脚本的命令。
    *   **简短报告**: 展示你的优化效果，通常以表格形式对比优化前后的性能指标。

4.  **可复现性**:
    *   这是**个人部分的核心评分标准**。评估者应该能够根据你的`README`说明，成功运行你的代码，并得到与你报告中相符或接近的实验结果。

#### 3. 技术规格 (Technical Specifications)

为保证实验的公平性和可比性，所有实现和测试必须遵循以下规格：

*   **基础模型**: **Pythia-70M**
    *   Hugging Face模型ID: `EleutherAI/pythia-70m-deduped`
    *   应使用其完全训练好的最终版本（`step143000`或`main`分支）。
*   **核心库**: Hugging Face `transformers`
*   **测试数据集**: `pg-19` 和 `wikitext` (`wikitext-2-raw-v1`)。
    *   `pg-19`用于测试超长文本性能，可选取单个样本进行测试。
*   **性能评估指标**:
    *   **效果指标**: PPL (Perplexity)，确保优化没有严重损害模型性能。
    *   **速度指标**:
        *   TTFT (Time To First Token): 从发起请求到生成第一个token的时间。
        *   TPOT (Time Per Output Token): 平均生成每个token所需的时间。
        *   Throughput (吞吐量): 单位时间内处理的token总数。
    *   **资源指标**: Peak GPU Memory Usage (峰值显存占用)。

---

#### 4. 实施步骤 (Implementation Steps)

##### **阶段一：获取基线性能 (Baseline Performance Measurement)**

这是你所有工作的对照组，必须严谨、准确。

1.  **环境搭建**:
    *   创建一个新的Python虚拟环境。
    *   安装必要的库: `pip install torch transformers datasets accelerate`。

2.  **加载模型与数据**:
    *   使用`transformers`库加载`EleutherAI/pythia-70m-deduped`模型和对应的Tokenizer。
    *   使用`datasets`库加载`wikitext`或`pg-19`。

3.  **编写Baseline测试脚本**:
    *   创建一个Python脚本 (`baseline_test.py`)。
    *   在此脚本中，直接调用Hugging Face原生的 `model.generate()` 方法。
    *   **精确测量性能指标**:
        *   **TTFT/TPOT**: 在`generate`函数调用前后使用`time.time()`计时。为了测量TTFT，你可能需要使用`LogitsProcessor`或`StoppingCriteria`在生成第一个token后立即记录时间。
        *   **显存占用**: 在脚本开始时使用`torch.cuda.reset_peak_memory_stats()`重置计数器，在生成结束后使用`torch.cuda.max_memory_allocated()`获取峰值显存。
        *   **PPL**: Perplexity是一个评估指标，而非生成指标。你需要编写一个单独的函数来计算。它通常是通过计算模型在一段文本上的交叉熵损失（Cross-Entropy Loss）并取指数得到的。Hugging Face官网上有计算PPL的教程。

4.  **执行并记录结果**:
    *   在选定的数据集样本上运行你的`baseline_test.py`。
    *   将所有测得的性能指标记录在一个表格中。这将是你`README`报告的第一部分。

##### **阶段二：实现性能优化并进行对比 (Optimization & Comparison)**

这是你项目的核心工作。

1.  **选择并学习算法**:
    *   从PPT列出的“逐层KV Cache压缩层面”中选择一个算法，例如 **StreamingLLM**，因为它原理相对直观。
    *   仔细阅读该算法的原始论文，彻底理解其核心思想（例如，StreamingLLM是如何通过保留“起始token”和“滑动窗口”来工作的）。

2.  **实现自定义生成逻辑**:
    *   **关键**: **不要直接修改你本地安装的`transformers`库源码！**
    *   **推荐做法**:
        a. 在你的项目文件夹下创建一个新文件，如 `custom_generate.py`。
        b. 将`transformers`库中`generation/utils.py`文件里 `generate` 方法的核心循环逻辑复制到你的新文件中。
        c. 在这个循环内部，找到处理`past_key_values`（即KV Cache）的部分。
        d. 根据你所选算法的原理，**手动修改对`past_key_values`的处理方式**。例如，对于StreamingLLM，你需要在每一步生成后，将KV Cache裁剪到指定的窗口大小。

3.  **编写优化后测试脚本**:
    *   复制你的`baseline_test.py`为`optimized_test.py`。
    *   将其中调用`model.generate()`的部分，替换为调用你自己在`custom_generate.py`中实现的函数。
    *   确保所有测试条件（如输入文本、生成长度、硬件等）与Baseline测试完全一致。

4.  **执行、对比与分析**:
    *   运行`optimized_test.py`并记录所有性能指标。
    *   在`README.md`中创建第二个表格，或在第一个表格中增加一列，用于展示你的优化结果。
    *   **撰写分析**:
        *   对比两组数据，量化你的性能提升（例如：“通过实现StreamingLLM，在处理4096长度的上下文时，峰值显存占用降低了75%，TPOT提升了30%。”）。
        *   简要分析性能提升的原因（例如：“这是因为我们避免了存储完整的KV Cache，从而节省了大量显存并减少了Attention计算的复杂度。”）。
        *   如果优化效果不理想，也要进行分析和讨论，这是评分标准中“严谨”的一部分。

#### 5. 附录：有用资源

*   **Pythia-70M模型**: [https://huggingface.co/EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)
*   **参考实现仓库 (KVPress)**: [https://github.com/NVIDIA/kypress](https://github.com/NVIDIA/kypress) (这个仓库可以为你提供实现多种KV压缩算法的灵感和参考)。

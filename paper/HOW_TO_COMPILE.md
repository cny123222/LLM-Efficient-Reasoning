# 📄 LaTeX 编译说明

## 快速编译

### 方法1：命令行编译（推荐）

```bash
cd /root/LLM-Efficient-Reasoning/paper

# 编译一次（生成PDF，但引用可能不完整）
pdflatex dynatree.tex

# 编译两次（确保引用正确）
pdflatex dynatree.tex
pdflatex dynatree.tex

# 或者一行命令：
pdflatex dynatree.tex && pdflatex dynatree.tex
```

### 方法2：使用Make（如果有Makefile）

```bash
make
```

### 方法3：使用latexmk（自动处理引用）

```bash
latexmk -pdf dynatree.tex
```

---

## 🔍 查看生成的PDF

```bash
# Linux
xdg-open dynatree.pdf

# macOS
open dynatree.pdf

# 或者在IDE中直接打开
# 文件路径：/root/LLM-Efficient-Reasoning/paper/dynatree.pdf
```

---

## 🧹 清理临时文件

LaTeX编译会生成很多临时文件，可以用以下命令清理：

```bash
# 清理所有临时文件（保留 .tex 和 .pdf）
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls

# 或者使用latexmk清理
latexmk -c
```

---

## 📝 常见问题

### Q1: 编译报错 "undefined references"
**解决**：需要编译两次以解决交叉引用问题
```bash
pdflatex dynatree.tex
pdflatex dynatree.tex
```

### Q2: 引用显示为 [?]
**解决**：同样需要编译两次

### Q3: 找不到pdflatex命令
**解决**：需要安装TeX Live
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install mactex
```

### Q4: 中文显示问题
**解决**：本文档使用英文撰写，无需中文支持。如需中文，使用XeLaTeX：
```bash
xelatex dynatree.tex
```

---

## 📦 完整编译流程（带清理）

```bash
#!/bin/bash
cd /root/LLM-Efficient-Reasoning/paper

# 清理旧文件
rm -f *.aux *.log *.out *.toc

# 编译两次
pdflatex -interaction=nonstopmode dynatree.tex
pdflatex -interaction=nonstopmode dynatree.tex

# 检查结果
if [ -f dynatree.pdf ]; then
    echo "✓ 编译成功！PDF文件: dynatree.pdf"
    ls -lh dynatree.pdf
else
    echo "✗ 编译失败，请查看错误信息"
fi
```

保存为 `compile.sh` 并运行：
```bash
chmod +x compile.sh
./compile.sh
```

---

## 🎨 在IDE中编译

### VS Code
1. 安装插件：`LaTeX Workshop`
2. 打开 `dynatree.tex`
3. 按 `Ctrl+Alt+B` 或点击右上角的绿色播放按钮

### Overleaf（在线编辑）
1. 上传 `dynatree.tex` 和相关文件
2. 点击 "Recompile" 按钮
3. 自动生成PDF

---

## 📊 当前文档状态

- ✅ 标题：DynaTree: Dynamic Tree-based Speculative Decoding with Adaptive Pruning
- ✅ 作者：Nuoyan Chen, Jiamin Liu, Zhaocheng Li
- ✅ 单位：Shanghai Jiao Tong University
- ✅ Abstract：已完成
- ✅ Introduction：已完成
- ✅ Related Work：框架已完成
- ⏳ Method：待完成
- ⏳ Experiments：待完成
- ⏳ Conclusion：待完成

当前页数：13页（包含模板示例，实际论文内容约2-3页）

---

## 🔗 相关文件

- 主文件：`dynatree.tex`
- 样式文件：`neurips_2025.sty`
- 生成的PDF：`dynatree.pdf`
- 实验数据：`../papers/Tree_Speculative_Decoding_实验报告.md`
- 文献综述：`../related_work.md`


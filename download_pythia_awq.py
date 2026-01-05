#!/usr/bin/env python3
"""
下载 AWQ 量化版本的 Pythia-2.8B 模型

从魔塔社区 (ModelScope) 下载模型到本地。

Usage:
    python download_pythia_awq.py

注意:
    - 需要安装 modelscope: pip install modelscope
    - 模型大小约 1.5GB (AWQ INT4 量化)
    - 下载路径: /mnt/disk1/models/pythia-2.8b-awq
"""

import os
import sys

# 下载配置
MODEL_SAVE_PATH = "/mnt/disk1/models/pythia-2.8b-awq"

# 尝试多个可能的模型源
# 注意: TheBloke 的模型主要在 HuggingFace，魔塔社区可能没有直接镜像
# 如果魔塔没有，可以使用 HuggingFace 镜像或直接下载

def download_from_modelscope():
    """从魔塔社区下载"""
    try:
        from modelscope import snapshot_download
        
        print("=" * 60)
        print("从魔塔社区下载 AWQ 量化模型")
        print("=" * 60)
        
        # 魔塔社区可能的模型名称
        # 注意: 需要先在 https://modelscope.cn 上搜索确认模型是否存在
        possible_models = [
            "TheBloke/pythia-2.8B-AWQ",           # 原始名称
            "Pythia/pythia-2.8b-awq",             # 可能的变体
            "quantization/pythia-2.8b-awq",       # 可能的分类
        ]
        
        print(f"\n目标保存路径: {MODEL_SAVE_PATH}")
        print("\n尝试从魔塔社区下载...")
        print("注意: 如果模型在魔塔社区不存在，将自动尝试 HuggingFace 镜像\n")
        
        for model_id in possible_models:
            try:
                print(f"尝试下载: {model_id}")
                model_dir = snapshot_download(
                    model_id,
                    cache_dir=os.path.dirname(MODEL_SAVE_PATH),
                    local_dir=MODEL_SAVE_PATH,
                )
                print(f"\n✅ 下载成功!")
                print(f"模型路径: {model_dir}")
                return True
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                continue
        
        return False
        
    except ImportError:
        print("❌ modelscope 未安装")
        print("请运行: pip install modelscope")
        return False


def download_from_huggingface():
    """从 HuggingFace (使用镜像) 下载"""
    try:
        # 设置 HuggingFace 镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        from huggingface_hub import snapshot_download
        
        print("\n" + "=" * 60)
        print("从 HuggingFace 镜像下载 AWQ 量化模型")
        print("=" * 60)
        
        model_id = "TheBloke/pythia-2.8b-AWQ"
        
        print(f"\n模型: {model_id}")
        print(f"目标路径: {MODEL_SAVE_PATH}")
        print(f"使用镜像: https://hf-mirror.com")
        print("\n开始下载...\n")
        
        model_dir = snapshot_download(
            repo_id=model_id,
            local_dir=MODEL_SAVE_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"\n✅ 下载成功!")
        print(f"模型路径: {model_dir}")
        return True
        
    except ImportError:
        print("❌ huggingface_hub 未安装")
        print("请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ HuggingFace 下载失败: {e}")
        return False


def download_from_huggingface_direct():
    """直接从 HuggingFace 下载 (不使用镜像)"""
    try:
        # 清除镜像设置
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        
        from huggingface_hub import snapshot_download
        
        print("\n" + "=" * 60)
        print("直接从 HuggingFace 下载 (需要网络畅通)")
        print("=" * 60)
        
        model_id = "TheBloke/pythia-2.8b-AWQ"
        
        print(f"\n模型: {model_id}")
        print(f"目标路径: {MODEL_SAVE_PATH}")
        print("\n开始下载...\n")
        
        model_dir = snapshot_download(
            repo_id=model_id,
            local_dir=MODEL_SAVE_PATH,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"\n✅ 下载成功!")
        print(f"模型路径: {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 直接下载失败: {e}")
        return False


def verify_model():
    """验证下载的模型"""
    print("\n" + "=" * 60)
    print("验证模型文件")
    print("=" * 60)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ 模型目录不存在: {MODEL_SAVE_PATH}")
        return False
    
    # 检查关键文件
    required_files = ["config.json"]
    awq_files = ["model.safetensors", "pytorch_model.bin", "quant_config.json"]
    
    files = os.listdir(MODEL_SAVE_PATH)
    print(f"\n模型文件列表:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(MODEL_SAVE_PATH, f))
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
        print(f"  - {f} ({size_str})")
    
    # 检查必需文件
    missing = [f for f in required_files if f not in files]
    if missing:
        print(f"\n⚠️  缺少文件: {missing}")
    
    # 检查 AWQ 文件
    has_awq = any(f in files for f in awq_files)
    if has_awq:
        print("\n✅ AWQ 模型文件验证通过")
    else:
        print("\n⚠️  未找到 AWQ 权重文件，可能下载不完整")
    
    return has_awq


def main():
    print("\n" + "=" * 60)
    print("   Pythia-2.8B AWQ 量化模型下载脚本")
    print("=" * 60)
    print(f"\n目标路径: {MODEL_SAVE_PATH}")
    
    # 检查目录是否已存在
    if os.path.exists(MODEL_SAVE_PATH):
        files = os.listdir(MODEL_SAVE_PATH)
        if files:
            print(f"\n⚠️  目录已存在且不为空，包含 {len(files)} 个文件")
            response = input("是否继续下载 (会覆盖)? [y/N]: ").strip().lower()
            if response != 'y':
                print("取消下载")
                return
    
    # 创建父目录
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 依次尝试不同的下载源
    success = False
    
    # 1. 尝试魔塔社区
    print("\n[1/3] 尝试从魔塔社区下载...")
    success = download_from_modelscope()
    
    # 2. 尝试 HuggingFace 镜像
    if not success:
        print("\n[2/3] 尝试从 HuggingFace 镜像下载...")
        success = download_from_huggingface()
    
    # 3. 尝试直接从 HuggingFace 下载
    if not success:
        print("\n[3/3] 尝试直接从 HuggingFace 下载...")
        success = download_from_huggingface_direct()
    
    # 验证下载
    if success:
        verify_model()
        
        print("\n" + "=" * 60)
        print("📝 使用方法")
        print("=" * 60)
        print(f"""
# 安装 autoawq
pip install autoawq

# 加载 AWQ 模型
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "{MODEL_SAVE_PATH}",
    fuse_layers=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{MODEL_SAVE_PATH}")

# 生成
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
""")
    else:
        print("\n" + "=" * 60)
        print("❌ 所有下载方式都失败了")
        print("=" * 60)
        print("""
可能的解决方案:

1. 手动从 HuggingFace 下载:
   https://huggingface.co/TheBloke/pythia-2.8b-AWQ

2. 使用 git lfs:
   git lfs install
   git clone https://huggingface.co/TheBloke/pythia-2.8b-AWQ /mnt/disk1/models/pythia-2.8b-awq

3. 使用 huggingface-cli:
   pip install huggingface_hub
   huggingface-cli download TheBloke/pythia-2.8b-AWQ --local-dir /mnt/disk1/models/pythia-2.8b-awq

4. 检查网络连接或使用代理
""")


if __name__ == "__main__":
    main()







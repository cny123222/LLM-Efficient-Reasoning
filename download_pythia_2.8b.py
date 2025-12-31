#!/usr/bin/env python3
"""
下载 Pythia-2.8B 模型到指定目录

使用方法:
    python download_pythia_2.8b.py
"""

import os
import sys
from pathlib import Path

# 使用 ModelScope 下载模型
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    from modelscope import AutoModel, AutoTokenizer as MSTokenizer
    USE_MODELSCOPE = True
    print("✓ 使用 ModelScope 下载模型")
except ImportError:
    print("⚠ ModelScope 未安装，尝试使用 Hugging Face Hub")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    USE_MODELSCOPE = False

def download_pythia_2_8b(model_dir="/mnt/disk1/models/pythia-2.8b"):
    """
    下载 Pythia-2.8B 模型
    
    Args:
        model_dir: 模型保存目录
    """
    model_name = "EleutherAI/pythia-2.8b"
    
    print(f"开始下载模型: {model_name}")
    print(f"目标目录: {model_dir}")
    
    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    
    if USE_MODELSCOPE:
        try:
            # 方法1: 使用 ModelScope 的 snapshot_download（主要方法）
            print("\n方法1: 使用 ModelScope snapshot_download 下载模型...")
            ms_snapshot_download(
                model_id=model_name,
                local_dir=model_dir,  # 直接下载到指定目录
                local_files_only=False,
                revision='master',
            )
            print(f"✓ 模型已下载到: {model_dir}")
            
            # 验证下载
            print("\n验证模型文件...")
            model_files = list(Path(model_dir).glob("*.bin")) + list(Path(model_dir).glob("*.safetensors"))
            if model_files:
                print(f"✓ 找到 {len(model_files)} 个模型文件")
            else:
                print("⚠ 警告: 未找到模型权重文件，检查子目录...")
                # ModelScope 可能将文件保存在子目录中
                for subdir in Path(model_dir).iterdir():
                    if subdir.is_dir():
                        sub_files = list(subdir.glob("*.bin")) + list(subdir.glob("*.safetensors"))
                        if sub_files:
                            print(f"✓ 在 {subdir} 中找到 {len(sub_files)} 个模型文件")
            
            # 尝试加载 tokenizer 验证
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                print(f"✓ Tokenizer 验证成功")
            except:
                print("⚠ Tokenizer 验证跳过（可能需要从子目录加载）")
            
            print(f"\n✓ 下载完成！模型保存在: {model_dir}")
            
        except Exception as e:
            print(f"\n✗ ModelScope 下载失败: {e}")
            print("\n尝试方法2: 使用 ModelScope AutoModel...")
            
            try:
                # 方法2: 使用 transformers + ModelScope（通过环境变量）
                print("\n方法2: 使用 transformers 通过 ModelScope 下载模型...")
                # 设置环境变量强制使用 ModelScope
                os.environ['USE_MODELSCOPE'] = '1'
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                print("正在下载模型权重...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=None,  # 不使用缓存，直接下载
                    resume_download=True,
                )
                
                print("正在下载tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=None,
                    resume_download=True,
                )
                
                # 保存模型和tokenizer到指定目录
                print(f"正在保存模型到: {model_dir}")
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                
                print(f"✓ 模型已下载并保存到: {model_dir}")
                
            except Exception as e2:
                print(f"✗ 方法2 也失败: {e2}")
                print("\n请检查:")
                print("1. 网络连接是否正常")
                print("2. 是否有足够的磁盘空间")
                print("3. ModelScope 模型名称是否正确")
                print("4. 是否安装了 modelscope: pip install modelscope")
                sys.exit(1)
    else:
        # 回退到 Hugging Face Hub（如果 ModelScope 不可用）
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("\n使用 Hugging Face Hub 下载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=model_dir,
                resume_download=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=model_dir,
                resume_download=True,
            )
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"✓ 模型已下载到: {model_dir}")
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 Pythia-2.8B 模型")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/disk1/models/pythia-2.8b",
        help="模型保存目录 (默认: /mnt/disk1/models/pythia-2.8b)"
    )
    
    args = parser.parse_args()
    
    download_pythia_2_8b(args.model_dir)


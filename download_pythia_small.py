#!/usr/bin/env python3
"""
下载 Pythia-70M 和 Pythia-160M 模型到指定目录

使用方法:
    python download_pythia_small.py
    python download_pythia_small.py --base_dir /mnt/disk1/models
"""

import os
import sys
from pathlib import Path

# 使用 ModelScope 下载模型
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    USE_MODELSCOPE = True
    print("✓ 使用 ModelScope 下载模型")
except ImportError:
    print("⚠ ModelScope 未安装，尝试使用 Hugging Face Hub")
    from huggingface_hub import snapshot_download
    USE_MODELSCOPE = False

def download_model(model_name, model_dir, use_modelscope=True):
    """
    下载单个模型
    
    Args:
        model_name: 模型名称（如 "EleutherAI/pythia-70m"）
        model_dir: 模型保存目录
        use_modelscope: 是否使用 ModelScope
    """
    print(f"\n{'='*60}")
    print(f"开始下载模型: {model_name}")
    print(f"目标目录: {model_dir}")
    print(f"{'='*60}")
    
    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    
    if use_modelscope:
        try:
            # 方法1: 使用 ModelScope 的 snapshot_download（主要方法）
            print(f"\n使用 ModelScope snapshot_download 下载 {model_name}...")
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
            except Exception as e:
                print(f"⚠ Tokenizer 验证跳过: {e}")
            
            print(f"\n✓ {model_name} 下载完成！模型保存在: {model_dir}")
            return True
            
        except Exception as e:
            print(f"\n✗ ModelScope 下载失败: {e}")
            print("\n尝试使用 transformers 通过 ModelScope 下载...")
            
            try:
                # 方法2: 使用 transformers + ModelScope（通过环境变量）
                print(f"\n使用 transformers 通过 ModelScope 下载 {model_name}...")
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
                
                print(f"✓ {model_name} 已下载并保存到: {model_dir}")
                return True
                
            except Exception as e2:
                print(f"✗ 方法2 也失败: {e2}")
                return False
    else:
        # 回退到 Hugging Face Hub（如果 ModelScope 不可用）
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"\n使用 Hugging Face Hub 下载 {model_name}...")
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
            print(f"✓ {model_name} 已下载到: {model_dir}")
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False


def download_pythia_small_models(base_dir="/mnt/disk1/models"):
    """
    下载 Pythia-70M 和 Pythia-160M 模型
    
    Args:
        base_dir: 基础目录，模型将保存在 base_dir/pythia-70m 和 base_dir/pythia-160m
    """
    models = [
        ("EleutherAI/pythia-70m", "pythia-70m"),
        ("EleutherAI/pythia-160m", "pythia-160m"),
    ]
    
    print(f"\n{'#'*60}")
    print(f"开始下载 Pythia 小模型")
    print(f"基础目录: {base_dir}")
    print(f"{'#'*60}\n")
    
    results = {}
    
    for model_name, model_subdir in models:
        model_dir = os.path.join(base_dir, model_subdir)
        success = download_model(model_name, model_dir, USE_MODELSCOPE)
        results[model_subdir] = success
    
    # 总结
    print(f"\n{'#'*60}")
    print("下载总结:")
    print(f"{'#'*60}")
    for model_subdir, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {model_subdir}: {status}")
    
    failed_models = [name for name, success in results.items() if not success]
    if failed_models:
        print(f"\n⚠ 以下模型下载失败: {', '.join(failed_models)}")
        print("请检查:")
        print("1. 网络连接是否正常")
        print("2. 是否有足够的磁盘空间")
        print("3. ModelScope 模型名称是否正确")
        print("4. 是否安装了 modelscope: pip install modelscope")
        return False
    else:
        print(f"\n✓ 所有模型下载完成！")
        print(f"模型保存在:")
        for model_subdir in results.keys():
            print(f"  - {os.path.join(base_dir, model_subdir)}")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 Pythia-70M 和 Pythia-160M 模型")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/disk1/models",
        help="基础目录，模型将保存在 base_dir/pythia-70m 和 base_dir/pythia-160m (默认: /mnt/disk1/models)"
    )
    
    args = parser.parse_args()
    
    success = download_pythia_small_models(args.base_dir)
    sys.exit(0 if success else 1)


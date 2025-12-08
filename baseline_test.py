from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
import torch
import time
from datasets import load_dataset
import numpy as np
from typing import Optional, List, Tuple

# 指定模型ID
model_id = "EleutherAI/pythia-70m-deduped"

# 加载最终的、完全训练好的模型和tokenizer
# AutoModelForCausalLM 会自动根据 config.json 中的 model_type 选择正确的模型类
# 对于 Pythia 系列，会自动加载 GPTNeoXForCausalLM
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # 如果需要，可以强制指定最终版本
    # revision="step143000", 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 设置pad_token（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_optimal_device():
    """
    Checks for available hardware accelerators and returns the most optimal one.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    # 1. 检查是否有可用的NVIDIA CUDA GPU
    if torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) is available. Using cuda.")
        return torch.device("cuda")
    
    # 2. 检查是否在macOS上并且MPS (Apple Silicon GPU) 可用
    # torch.backends.mps.is_available() 可以在未来的PyTorch版本中替代部分检查
    # 但目前的标准做法是检查 .is_built()
    if torch.backends.mps.is_available():
        # 这个检查确保你的PyTorch版本支持MPS
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
            print("Falling back to CPU.")
            return torch.device("cpu")
        else:
            print("MPS (Apple Silicon GPU) is available. Using mps.")
            return torch.device("mps")
            
    # 3. 如果以上两者都不可用，则回退到CPU
    print("No GPU acceleration available. Falling back to CPU.")
    return torch.device("cpu")

# 在你的主程序中使用这个函数
device = get_optimal_device()
model.to(device)
model.eval()  # 设置为评估模式

print(f"Model loaded on {device}")


class FirstTokenTimer(LogitsProcessor):
    """用于测量第一个token生成时间的LogitsProcessor"""
    def __init__(self):
        self.first_token_time = None
        self.start_time = None
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.start_time is None:
            self.start_time = time.time()
        if self.first_token_time is None and input_ids.shape[1] > 0:
            self.first_token_time = time.time()
        return scores


def measure_generation_performance(
    model, 
    tokenizer, 
    input_text: str, 
    max_new_tokens: int = 100,
    device: torch.device = None
) -> dict:
    """
    测量生成性能指标：TTFT, TPOT, Throughput
    
    Returns:
        dict: 包含TTFT, TPOT, Throughput, total_time, num_tokens等指标
    """
    # 编码输入文本
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    # 重置显存统计（仅对CUDA有效）
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # 创建FirstTokenTimer来测量TTFT
    first_token_timer = FirstTokenTimer()
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行生成
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪心解码
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[first_token_timer],
        )
    
    # 记录结束时间
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    num_generated_tokens = output.shape[1] - input_length
    
    # TTFT: Time To First Token
    if first_token_timer.first_token_time is not None:
        ttft = first_token_timer.first_token_time - start_time
    else:
        ttft = total_time  # 如果没有生成任何token，使用总时间
    
    # TPOT: Time Per Output Token (平均每个输出token的时间)
    if num_generated_tokens > 0:
        tpot = total_time / num_generated_tokens
    else:
        tpot = float('inf')
    
    # Throughput: tokens per second
    throughput = num_generated_tokens / total_time if total_time > 0 else 0
    
    # 峰值显存占用（仅对CUDA有效）
    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为MB
    
    return {
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "num_generated_tokens": num_generated_tokens,
        "peak_memory_mb": peak_memory,
        "input_length": input_length,
        "output_length": output.shape[1]
    }


def calculate_perplexity(
    model, 
    tokenizer, 
    text: str, 
    device: torch.device = None,
    max_length: int = 512
) -> float:
    """
    计算文本的困惑度（Perplexity）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        device: 设备
        max_length: 最大序列长度
    
    Returns:
        float: 困惑度值
    """
    # 编码文本
    encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = encodings.input_ids.to(device)
    
    # 计算损失
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # 困惑度 = exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity


def load_test_data():
    """加载测试数据集"""
    print("\nLoading test datasets...")
    
    # 加载wikitext-2-raw-v1
    print("Loading wikitext-2-raw-v1...")
    wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # 加载pg-19（用于超长文本测试）
    print("Loading pg-19...")
    try:
        pg19_dataset = load_dataset("pg19", split="test")
    except Exception as e:
        print(f"Warning: Could not load pg-19 dataset: {e}")
        pg19_dataset = None
    
    return wikitext_dataset, pg19_dataset


def run_baseline_tests():
    """运行基线性能测试"""
    print("\n" + "="*60)
    print("Baseline Performance Measurement")
    print("="*60)
    
    # 加载测试数据
    wikitext_dataset, pg19_dataset = load_test_data()
    
    results = []
    
    # 测试1: wikitext数据集（选择前几个样本）
    print("\n" + "-"*60)
    print("Test 1: Wikitext Dataset")
    print("-"*60)
    
    num_wikitext_samples = min(5, len(wikitext_dataset))
    for i in range(num_wikitext_samples):
        sample = wikitext_dataset[i]
        text = sample.get("text", "")
        
        # 跳过空文本或过短的文本
        if not text or len(text.strip()) < 50:
            continue
        
        print(f"\nProcessing wikitext sample {i+1}/{num_wikitext_samples}...")
        print(f"Input text length: {len(text)} characters")
        
        # 截取前512个字符作为输入（避免输入过长）
        input_text = text[:512]
        
        # 测量生成性能
        gen_metrics = measure_generation_performance(
            model, tokenizer, input_text, 
            max_new_tokens=50,  # 生成50个新token
            device=device
        )
        
        # 计算困惑度（使用原始文本的前1024个字符）
        ppl_text = text[:1024] if len(text) >= 1024 else text
        try:
            ppl = calculate_perplexity(model, tokenizer, ppl_text, device=device)
        except Exception as e:
            print(f"Warning: Could not calculate perplexity: {e}")
            ppl = None
        
        result = {
            "dataset": "wikitext",
            "sample_id": i,
            "input_length": gen_metrics["input_length"],
            "output_length": gen_metrics["output_length"],
            "ttft_seconds": gen_metrics["ttft"],
            "tpot_seconds": gen_metrics["tpot"],
            "throughput_tokens_per_sec": gen_metrics["throughput"],
            "peak_memory_mb": gen_metrics["peak_memory_mb"],
            "perplexity": ppl
        }
        results.append(result)
        
        # 打印结果
        print(f"  TTFT: {gen_metrics['ttft']:.4f} seconds")
        print(f"  TPOT: {gen_metrics['tpot']:.4f} seconds")
        print(f"  Throughput: {gen_metrics['throughput']:.2f} tokens/sec")
        if gen_metrics['peak_memory_mb']:
            print(f"  Peak Memory: {gen_metrics['peak_memory_mb']:.2f} MB")
        if ppl:
            print(f"  Perplexity: {ppl:.2f}")
    
    # 测试2: pg-19数据集（选择一个样本进行超长文本测试）
    if pg19_dataset is not None:
        print("\n" + "-"*60)
        print("Test 2: PG-19 Dataset (Long Context)")
        print("-"*60)
        
        # 选择第一个非空样本
        for i in range(min(10, len(pg19_dataset))):
            sample = pg19_dataset[i]
            text = sample.get("text", "")
            
            if not text or len(text.strip()) < 100:
                continue
            
            print(f"\nProcessing pg-19 sample {i+1}...")
            print(f"Input text length: {len(text)} characters")
            
            # 对于pg-19，使用更长的输入（前1024个字符）
            input_text = text[:1024]
            
            # 测量生成性能
            gen_metrics = measure_generation_performance(
                model, tokenizer, input_text,
                max_new_tokens=100,  # 生成100个新token
                device=device
            )
            
            # 计算困惑度（使用原始文本的前2048个字符）
            ppl_text = text[:2048] if len(text) >= 2048 else text
            try:
                ppl = calculate_perplexity(model, tokenizer, ppl_text, device=device)
            except Exception as e:
                print(f"Warning: Could not calculate perplexity: {e}")
                ppl = None
            
            result = {
                "dataset": "pg-19",
                "sample_id": i,
                "input_length": gen_metrics["input_length"],
                "output_length": gen_metrics["output_length"],
                "ttft_seconds": gen_metrics["ttft"],
                "tpot_seconds": gen_metrics["tpot"],
                "throughput_tokens_per_sec": gen_metrics["throughput"],
                "peak_memory_mb": gen_metrics["peak_memory_mb"],
                "perplexity": ppl
            }
            results.append(result)
            
            # 打印结果
            print(f"  TTFT: {gen_metrics['ttft']:.4f} seconds")
            print(f"  TPOT: {gen_metrics['tpot']:.4f} seconds")
            print(f"  Throughput: {gen_metrics['throughput']:.2f} tokens/sec")
            if gen_metrics['peak_memory_mb']:
                print(f"  Peak Memory: {gen_metrics['peak_memory_mb']:.2f} MB")
            if ppl:
                print(f"  Perplexity: {ppl:.2f}")
            
            break  # 只测试一个pg-19样本
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("Baseline Performance Summary")
    print("="*60)
    print(f"{'Dataset':<12} {'Sample':<8} {'Input Len':<10} {'TTFT(s)':<10} {'TPOT(s)':<10} {'Throughput':<12} {'Memory(MB)':<12} {'PPL':<10}")
    print("-"*60)
    
    for r in results:
        memory_str = f"{r['peak_memory_mb']:.2f}" if r['peak_memory_mb'] else "N/A"
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] else "N/A"
        print(f"{r['dataset']:<12} {r['sample_id']:<8} {r['input_length']:<10} "
              f"{r['ttft_seconds']:<10.4f} {r['tpot_seconds']:<10.4f} "
              f"{r['throughput_tokens_per_sec']:<12.2f} {memory_str:<12} {ppl_str:<10}")
    
    # 计算平均值
    if results:
        print("\n" + "-"*60)
        print("Average Metrics:")
        avg_ttft = np.mean([r['ttft_seconds'] for r in results])
        avg_tpot = np.mean([r['tpot_seconds'] for r in results])
        avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in results])
        
        # 修复：检查是否有有效的内存数据
        memory_values = [r['peak_memory_mb'] for r in results if r['peak_memory_mb']]
        avg_memory = np.mean(memory_values) if memory_values else None
        
        # 修复：检查是否有有效的困惑度数据
        ppl_values = [r['perplexity'] for r in results if r['perplexity']]
        avg_ppl = np.mean(ppl_values) if ppl_values else None
        
        print(f"  Average TTFT: {avg_ttft:.4f} seconds")
        print(f"  Average TPOT: {avg_tpot:.4f} seconds")
        print(f"  Average Throughput: {avg_throughput:.2f} tokens/sec")
        if avg_memory is not None:
            print(f"  Average Peak Memory: {avg_memory:.2f} MB")
        else:
            print(f"  Average Peak Memory: N/A (not supported on this device)")
        if avg_ppl is not None:
            print(f"  Average Perplexity: {avg_ppl:.2f}")
    
    return results


if __name__ == "__main__":
    # 运行基线测试
    results = run_baseline_tests()
    
    print("\n" + "="*60)
    print("Baseline testing completed!")
    print("="*60)
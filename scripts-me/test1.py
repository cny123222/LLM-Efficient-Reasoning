import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_decode.core import SpeculativeGenerator, TreeSpeculativeGeneratorV2, StreamingSpeculativeGenerator
import time
import gc
import warnings
warnings.filterwarnings('ignore')

device = 'cuda'
target_path = '/mnt/disk1/models/pythia-2.8b'
draft_path = '/mnt/disk1/models/pythia-70m'

tokenizer = AutoTokenizer.from_pretrained(target_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch.float16, device_map=device)
draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=torch.float16, device_map=device)

prompt = 'Write a detailed explanation about the development of large language models:'
max_new_tokens = 500

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# 充分 Warmup
print('Warming up (3 runs)...')
for _ in range(3):
    cleanup()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.inference_mode():
        _ = target_model.generate(input_ids, max_new_tokens=100, do_sample=False, 
                                  eos_token_id=None, assistant_model=draft_model)
    torch.cuda.synchronize()

print('\\n' + '='*70)
print(f'准确性能测试: {max_new_tokens} tokens (3次取平均, 跳过首次)')
print('='*70)

results = []

# 1. Baseline
print('\\n1. Baseline (纯自回归)...')
tps = []
for run in range(4):  # 4次，跳过首次
    cleanup()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = target_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, 
                                    eos_token_id=None, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = out.shape[1] - input_ids.shape[1]
    tp = tokens / elapsed
    if run > 0:  # 跳过首次
        tps.append(tp)
    print(f'   Run {run+1}: {tokens} tokens, {tp:.1f} t/s' + (' (warmup)' if run==0 else ''))
baseline_tp = sum(tps) / len(tps)
results.append(('Baseline (AR)', baseline_tp, 1.0))
print(f'   >>> 平均: {baseline_tp:.1f} t/s')

# 2. HF Assisted
print('\\n2. HuggingFace Assisted...')
tps = []
for run in range(4):
    cleanup()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = target_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, 
                                    eos_token_id=None, assistant_model=draft_model, 
                                    pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = out.shape[1] - input_ids.shape[1]
    tp = tokens / elapsed
    if run > 0:
        tps.append(tp)
    print(f'   Run {run+1}: {tokens} tokens, {tp:.1f} t/s' + (' (warmup)' if run==0 else ''))
hf_tp = sum(tps) / len(tps)
results.append(('HF Assisted', hf_tp, hf_tp/baseline_tp))
print(f'   >>> 平均: {hf_tp:.1f} t/s ({hf_tp/baseline_tp:.2f}x)')

# 3. Linear Spec Decode (多个 K 值)
print('\\n3. Linear Spec Decode...')
for K in [5, 6, 7, 8]:
    cleanup()
    gen = SpeculativeGenerator(target_model, draft_model, tokenizer, K=K, max_len=8192, device=device, use_compile=False)
    tps = []
    for run in range(4):
        gen.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = gen.generate(prompt, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        stats = gen.get_stats()
        tp = stats['total_tokens'] / elapsed
        if run > 0:
            tps.append(tp)
    avg_tp = sum(tps) / len(tps)
    results.append((f'Linear K={K}', avg_tp, avg_tp/baseline_tp))
    print(f'   K={K}: {avg_tp:.1f} t/s ({avg_tp/baseline_tp:.2f}x)')

# 4. Tree V2 (最优配置 D=8 B=3 t=0.03)
print('\\n4. Tree V2 (D=8 B=3 t=0.03)...')
cleanup()
gen = TreeSpeculativeGeneratorV2(
    target_model, draft_model, tokenizer,
    tree_depth=8, branch_factor=3, probability_threshold=0.03,
    max_tree_nodes=128, device=device, use_compile=False
)
tps = []
for run in range(4):
    gen.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = gen.generate(prompt, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = gen.get_stats()
    tp = stats['total_tokens'] / elapsed
    if run > 0:
        tps.append(tp)
    print(f'   Run {run+1}: {tp:.1f} t/s' + (' (warmup)' if run==0 else ''))
tree_tp = sum(tps) / len(tps)
results.append(('Tree V2 D=8B=3t=0.03', tree_tp, tree_tp/baseline_tp))
print(f'   >>> 平均: {tree_tp:.1f} t/s ({tree_tp/baseline_tp:.2f}x)')

# 5. Streaming
print('\\n5. Streaming Spec Decode K=6...')
cleanup()
gen = StreamingSpeculativeGenerator(
    target_model, draft_model, tokenizer,
    K=6, max_len=8192, max_cache_len=1024, start_size=4, recent_size=1020,
    device=device, use_compile=False
)
tps = []
for run in range(4):
    gen.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = gen.generate(prompt, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats = gen.get_stats()
    tp = stats['total_tokens'] / elapsed
    if run > 0:
        tps.append(tp)
stream_tp = sum(tps) / len(tps)
results.append(('Streaming K=6 c=1024', stream_tp, stream_tp/baseline_tp))
print(f'   >>> 平均: {stream_tp:.1f} t/s ({stream_tp/baseline_tp:.2f}x)')

# 排序输出
print('\\n' + '='*70)
print('最终结果 (按加速比排序)')
print('='*70)
print(f'{\"方法\":<25} {\"吞吐量\":>12} {\"加速比\":>10}')
print('-'*50)
for name, tp, speedup in sorted(results, key=lambda x: x[2], reverse=True):
    print(f'{name:<25} {tp:>10.1f} t/s {speedup:>8.2f}x')
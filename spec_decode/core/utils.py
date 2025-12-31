"""
Utility functions for speculative decoding.
"""

import torch
import time
from typing import Callable, Any, Dict
from contextlib import contextmanager


@contextmanager
def timer(name: str = ""):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name:
        print(f"{name}: {elapsed*1000:.2f}ms")


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2
    }


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class ThroughputMeter:
    """Measure throughput (tokens per second)."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.start_time = None
    
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def stop(self, num_tokens: int):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        self.total_tokens += num_tokens
        self.total_time += elapsed
        self.start_time = None
    
    def get_throughput(self) -> float:
        """Get throughput in tokens/second."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time
    
    def get_stats(self) -> Dict[str, float]:
        return {
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "throughput": self.get_throughput()
        }


def warmup_model(model: torch.nn.Module, tokenizer, device: str = "cuda", num_warmup: int = 3):
    """Warm up model with dummy inputs to initialize CUDA kernels."""
    dummy_input = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()



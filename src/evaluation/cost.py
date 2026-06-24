"""
Computational cost measurement (DEVELOPMENT.md §7).

Cost metrics must be measured identically across all model families to be
comparable: wall-clock training time, peak process RAM, inference latency,
parameter count, and (for GPU models) GPU memory.
"""
from __future__ import annotations

import os
import time

import psutil


class CostTracker:
    """Context manager that tracks training wall-clock time and peak RAM.

    Call `update_ram` periodically during long training loops to capture the peak.

    Example:
        >>> with CostTracker() as tracker:
        ...     model.fit(X_train, y_train)
        ...     tracker.update_ram()
        >>> tracker.elapsed_s, tracker.peak_ram_mb
    """

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.elapsed_s: float = 0.0
        self.peak_ram_mb: float = 0.0
        self._process = psutil.Process(os.getpid())

    def __enter__(self) -> "CostTracker":
        self.start_time = time.perf_counter()
        self.peak_ram_mb = self._process.memory_info().rss / 1024 / 1024
        return self

    def update_ram(self) -> None:
        """Update the running peak RAM measurement."""
        current = self._process.memory_info().rss / 1024 / 1024
        if current > self.peak_ram_mb:
            self.peak_ram_mb = current

    def __exit__(self, *args) -> None:
        self.elapsed_s = time.perf_counter() - self.start_time


def measure_inference_time(model, X_test, n_warmup: int = 10, n_max: int = 1000) -> float:
    """Average inference time in milliseconds per sample, with warmup (§7).

    Args:
        model: A fitted estimator with a ``predict`` method.
        X_test: Test features (array-like, sliceable).
        n_warmup: Warmup prediction rounds (excluded from timing).
        n_max: Cap on the number of samples timed.

    Returns:
        Mean inference time in milliseconds per sample.
    """
    n = min(n_max, len(X_test))
    warm = X_test[:32]
    for _ in range(n_warmup):
        model.predict(warm)
    start = time.perf_counter()
    model.predict(X_test[:n])
    elapsed = time.perf_counter() - start
    return (elapsed / n) * 1000.0


def count_parameters(model) -> int | None:
    """Trainable parameter count for PyTorch models; ``None`` for non-torch models."""
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except ImportError:
        pass
    return None


def gpu_used_mb() -> float | None:
    """Current GPU-0 memory used in MB, or ``None`` if no GPU/NVML is available."""
    try:
        import py3nvml.py3nvml as nvml

        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        used = nvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
        nvml.nvmlShutdown()
        return float(used)
    except Exception:
        return None

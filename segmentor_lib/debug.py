"""
Memory debugging utilities for SAM3-RS.

Provides tools to track tensor memory usage and log debug information.
"""

import torch
from typing import Optional


class MemoryDebugger:
    """Memory debugging utility for tensor analysis."""

    def __init__(self, enabled: bool = False, log_file: Optional[str] = None):
        """
        Args:
            enabled: Enable memory tracking
            log_file: Path to log file for debug output
        """
        self.enabled = enabled
        self.log_handle = None  # 先初始化，避免 __del__ 访问失败

        if log_file:
            import os
            # 自动创建日志目录
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            try:
                self.log_handle = open(log_file, 'w', encoding='utf-8')
            except Exception as e:
                print(f"[Warning] Failed to open log file {log_file}: {e}")
                self.log_handle = None

    def __del__(self):
        """Close log file on cleanup."""
        if hasattr(self, 'log_handle') and self.log_handle:
            self.log_handle.close()

    def debug_print(self, message: str):
        """Print debug message to log file only (not to console)."""
        if not self.enabled or not self.log_handle:
            return
        self.log_handle.write(message + '\n')
        self.log_handle.flush()

    def log_cuda_memory(self, message: str):
        """Log current CUDA memory allocation and reservation.

        Args:
            message: Context message describing the operation
        """
        if not self.enabled:
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        msg = f"[MEMORY] {message}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
        self.debug_print(msg)
        # Ensure flush after memory log
        if self.log_handle:
            self.log_handle.flush()

    def log_tensor_memory(self, name: str, tensor: torch.Tensor, detail: bool = False):
        """Log tensor memory usage.

        Args:
            name: Tensor name for identification
            tensor: Tensor to analyze
            detail: Whether to log statistics (min/max/mean) for 3D+ tensors
        """
        if not self.enabled or tensor is None:
            return

        numel = tensor.numel()
        element_size = tensor.element_size()
        memory_mb = numel * element_size / (1024 * 1024)

        msg = f"[MEMORY] {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, " \
              f"memory={memory_mb:.2f} MB ({numel:,} elements)"
        self.debug_print(msg)

        if detail and tensor.dim() > 2:
            # Log statistics for 3D+ tensors
            detail_msg = f"        min={tensor.min():.4f}, max={tensor.max():.4f}, " \
                         f"mean={tensor.mean():.4f}"
            self.debug_print(detail_msg)

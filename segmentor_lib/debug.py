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
        self.log_handle = open(log_file, 'w', encoding='utf-8') if log_file else None

    def __del__(self):
        """Close log file on cleanup."""
        if self.log_handle:
            self.log_handle.close()

    def debug_print(self, message: str):
        """Print debug message to log file only (not to console)."""
        if self.log_handle:
            self.log_handle.write(message + '\n')
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

"""
SAM3-RS: SAM3 adapted for Remote Sensing Zero-shot Segmentation

Lightweight wrapper around SAM3 with remote sensing specific optimizations:
- Sliding window inference for large images
- Multi-class text prompt handling
- Dual-head fusion (instance + semantic)
- Presence score filtering
- Remote sensing dataset support
"""

from .segmentor import SAM3RSSegmentor
from .data import RSDataLoader
from .prompts import PromptManager
from .metrics import RSEvaluator
from .utils import visualize_results

__all__ = [
    'SAM3RSSegmentor',
    'RSDataLoader',
    'PromptManager',
    'RSEvaluator',
    'visualize_results',
]

__version__ = '0.1.0'

"""SAM3-RS Segmentor Library.

This package provides modular components for SAM3-based remote sensing segmentation.
"""

# Core components (stable)
from .core import InferenceEngine
from .prompts import load_prompts, precompute_text_features
from .postprocess import fuse_prompts_to_classes, logits_to_pred, resize_logits
from .sliding_window import SlidingWindowInference
from .multi_scale import MultiScaleInference
from .config_loader import load_inference_config, save_inference_config
from .analyzers import BaseAnalyzer, PresenceScoreAnalyzer, StatisticsAnalyzer
from .debug import MemoryDebugger

__all__ = [
    # Core
    "InferenceEngine",
    "load_prompts",
    "precompute_text_features",
    "fuse_prompts_to_classes",
    "logits_to_pred",
    "resize_logits",
    "SlidingWindowInference",
    "MultiScaleInference",
    # Config
    "load_inference_config",
    "save_inference_config",
    # Analyzers
    "BaseAnalyzer",
    "PresenceScoreAnalyzer",
    "StatisticsAnalyzer",
    # Debug
    "MemoryDebugger",
]

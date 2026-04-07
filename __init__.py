"""SAM3-RS: Lightweight SAM3 wrapper for remote sensing open-vocabulary semantic segmentation."""

from .segmentor import SAM3RSSegmentor, InferenceConfig, SegmentationResult

__version__ = "0.2.0"
__all__ = [
    "SAM3RSSegmentor",
    "InferenceConfig",
    "SegmentationResult",
]

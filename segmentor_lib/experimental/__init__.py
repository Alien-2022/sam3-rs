"""Experimental features for SAM3-RS.

This module contains experimental features that may be removed or significantly
changed in future versions. Use with caution.

Available features:
- semantic_enhancement: Synonym-based prompt enhancement
- unsupervised_threshold: Unsupervised threshold calibration based on confidence distributions
"""

from .semantic_enhancement import (
    apply_semantic_enhancement,
    apply_semantic_enhancement_with_avg_embedding,
    get_semantic_enhancer,
)
from .unsupervised_threshold import UnsupervisedThresholdCalibration

__all__ = [
    # Semantic enhancement
    "apply_semantic_enhancement",
    "apply_semantic_enhancement_with_avg_embedding",
    "get_semantic_enhancer",
    # Unsupervised threshold calibration
    "UnsupervisedThresholdCalibration",
]

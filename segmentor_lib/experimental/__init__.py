"""Experimental features for SAM3-RS.

This module contains experimental features that may be removed or significantly
changed in future versions. Use with caution.

Available features:
- unsupervised_threshold: Unsupervised threshold calibration based on confidence distributions
"""

from .unsupervised_threshold import UnsupervisedThresholdCalibration

__all__ = [
    "UnsupervisedThresholdCalibration",
]

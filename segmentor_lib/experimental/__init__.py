"""Experimental features for SAM3-RS.

This module contains experimental features that may be removed or significantly
changed in future versions. Use with caution.

Available features:
- semantic_enhancement: Synonym-based prompt enhancement
- adaptive_threshold: Dynamic threshold adjustment strategies
"""

from .semantic_enhancement import (
    apply_semantic_enhancement,
    apply_semantic_enhancement_with_avg_embedding,
    get_semantic_enhancer,
)
from .adaptive_threshold import (
    get_adaptive_threshold_strategy,
    AdaptiveThresholdBase,
    PresenceScoreAdaptiveThreshold,
    ClassSpecificAdaptiveThreshold,
    HybridAdaptiveThreshold,
    ImageFeatureAdaptiveThreshold,
    logits_to_pred_adaptive,
)

__all__ = [
    # Semantic enhancement
    "apply_semantic_enhancement",
    "apply_semantic_enhancement_with_avg_embedding",
    "get_semantic_enhancer",
    # Adaptive threshold
    "get_adaptive_threshold_strategy",
    "AdaptiveThresholdBase",
    "PresenceScoreAdaptiveThreshold",
    "ClassSpecificAdaptiveThreshold",
    "HybridAdaptiveThreshold",
    "ImageFeatureAdaptiveThreshold",
    "logits_to_pred_adaptive",
]

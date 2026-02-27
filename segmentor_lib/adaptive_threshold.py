"""Adaptive threshold strategies for SAM3-RS inference.

This module provides various strategies for dynamically adjusting
prob_threshold and confidence_threshold based on image characteristics,
class properties, and presence scores.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class AdaptiveThresholdBase:
    """Base class for adaptive threshold strategies."""

    def __init__(self, base_confidence: float, base_prob: float):
        self.base_confidence = base_confidence
        self.base_prob = base_prob

    def compute_thresholds(
        self,
        presence_scores: torch.Tensor,
        class_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Compute adaptive thresholds for a given prompt/class.

        Args:
            presence_scores: [batch_size] or [batch_size, 1] presence scores
            class_id: Optional class ID for class-specific adjustments
            **kwargs: Additional context information

        Returns:
            (confidence_threshold, prob_threshold)
        """
        raise NotImplementedError


class PresenceScoreAdaptiveThreshold(AdaptiveThresholdBase):
    """
    Adjust thresholds based on presence score.

    Logic:
    - High presence score (> 0.7): use stricter thresholds (reduce false positives)
    - Medium presence score (0.3-0.7): use base thresholds
    - Low presence score (< 0.3): use looser thresholds (reduce false negatives)
    """

    def __init__(
        self,
        base_confidence: float = 0.5,
        base_prob: float = 0.5,
        strict_factor: float = 1.3,
        loose_factor: float = 0.7,
        **kwargs
    ):
        super().__init__(base_confidence, base_prob)
        self.strict_factor = strict_factor
        self.loose_factor = loose_factor

    def compute_thresholds(
        self,
        presence_scores: torch.Tensor,
        class_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        # Average presence score across batch
        avg_presence = presence_scores.mean().item()

        if avg_presence > 0.5:
            # High confidence: use stricter thresholds
            confidence = self.base_confidence * self.strict_factor
            prob = self.base_prob * self.strict_factor
        elif avg_presence < 0.2:
            # Low confidence: use looser thresholds
            confidence = self.base_confidence * self.loose_factor
            prob = self.base_prob * self.loose_factor
        else:
            # Medium confidence: use base thresholds
            confidence = self.base_confidence
            prob = self.base_prob

        # Clamp to valid range
        confidence = min(max(confidence, 0.05), 0.95)
        prob = min(max(prob, 0.01), 0.5)

        return float(confidence), float(prob)


class ClassSpecificAdaptiveThreshold(AdaptiveThresholdBase):
    """
    Use different base thresholds for different classes.

    Classes can be configured with their own threshold adjustments.
    """

    def __init__(
        self,
        base_confidence: float = 0.25,
        base_prob: float = 0.05,
        class_configs: Optional[Dict[int, Dict[str, float]]] = None
    ):
        super().__init__(base_confidence, base_prob)
        self.class_configs = class_configs or {}

    def compute_thresholds(
        self,
        presence_scores: torch.Tensor,
        class_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        confidence = self.base_confidence
        prob = self.base_prob

        if class_id is not None and class_id in self.class_configs:
            config = self.class_configs[class_id]
            confidence = config.get('confidence_threshold', confidence)
            prob = config.get('prob_threshold', prob)

        return float(confidence), float(prob)


class HybridAdaptiveThreshold(AdaptiveThresholdBase):
    """
    Combine presence score adaptation with class-specific adjustments.

    First apply class-specific base thresholds, then adjust based on presence score.
    """

    def __init__(
        self,
        base_confidence: float = 0.25,
        base_prob: float = 0.05,
        class_configs: Optional[Dict[int, Dict[str, float]]] = None,
        use_presence_adaptation: bool = True
    ):
        super().__init__(base_confidence, base_prob)
        self.class_configs = class_configs or {}
        self.use_presence_adaptation = use_presence_adaptation

    def compute_thresholds(
        self,
        presence_scores: torch.Tensor,
        class_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        # Step 1: Get class-specific base thresholds
        confidence = self.base_confidence
        prob = self.base_prob

        if class_id is not None and class_id in self.class_configs:
            config = self.class_configs[class_id]
            confidence = config.get('confidence_threshold', confidence)
            prob = config.get('prob_threshold', prob)

        # Step 2: Adjust based on presence score
        if self.use_presence_adaptation:
            avg_presence = presence_scores.mean().item()

            if avg_presence > 0.7:
                confidence *= 1.2
                prob *= 1.2
            elif avg_presence < 0.3:
                confidence *= 0.8
                prob *= 0.8

        # Clamp to valid range
        confidence = min(max(confidence, 0.05), 0.95)
        prob = min(max(prob, 0.01), 0.5)

        return float(confidence), float(prob)


class ImageFeatureAdaptiveThreshold(AdaptiveThresholdBase):
    """
    Adjust thresholds based on image feature characteristics.

    Uses statistics of the image features (e.g., variance, mean)
    to determine if the image is "easy" or "hard".
    """

    def __init__(
        self,
        base_confidence: float = 0.25,
        base_prob: float = 0.05,
        hard_image_factor: float = 0.7,
        easy_image_factor: float = 1.1,
        **kwargs
    ):
        super().__init__(base_confidence, base_prob)
        self.hard_image_factor = hard_image_factor
        self.easy_image_factor = easy_image_factor

    def compute_thresholds(
        self,
        presence_scores: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        class_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[float, float]:
        confidence = self.base_confidence
        prob = self.base_prob

        # If image features are provided, compute complexity metric
        if image_features is not None:
            # Compute feature variance as a proxy for image complexity
            # image_features: [B, C, H, W]
            feat_std = image_features.std(dim=[2, 3]).mean().item()

            # Normalize std (typical range is 0.1-1.0)
            normalized_std = min(feat_std / 0.5, 1.0)

            if normalized_std > 0.7:
                # High complexity: use looser thresholds
                confidence = self.base_confidence * self.hard_image_factor
                prob = self.base_prob * self.hard_image_factor
            elif normalized_std < 0.3:
                # Low complexity: use stricter thresholds
                confidence = self.base_confidence * self.easy_image_factor
                prob = self.base_prob * self.easy_image_factor

        # Clamp to valid range
        confidence = min(max(confidence, 0.05), 0.95)
        prob = min(max(prob, 0.01), 0.5)

        return float(confidence), float(prob)


def get_adaptive_threshold_strategy(
    strategy_name: str,
    base_confidence: float = 0.25,
    base_prob: float = 0.05,
    **kwargs
) -> AdaptiveThresholdBase:
    """
    Factory function to create adaptive threshold strategy.

    Args:
        strategy_name: One of 'presence', 'class_specific', 'hybrid', 'image_feature'
        base_confidence: Base confidence threshold
        base_prob: Base prob threshold
        **kwargs: Strategy-specific arguments

    Returns:
        AdaptiveThresholdBase instance
    """
    strategies = {
        'presence': PresenceScoreAdaptiveThreshold,
        'class_specific': ClassSpecificAdaptiveThreshold,
        'hybrid': HybridAdaptiveThreshold,
        'image_feature': ImageFeatureAdaptiveThreshold,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available strategies: {list(strategies.keys())}"
        )

    strategy_cls = strategies[strategy_name]
    return strategy_cls(base_confidence=base_confidence,
                     base_prob=base_prob, **kwargs)

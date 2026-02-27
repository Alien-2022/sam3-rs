"""Lightweight streaming metrics for semantic segmentation."""

from __future__ import annotations

import numpy as np
from typing import Dict


class SegmentationMetric:
    """Accumulates confusion matrix and computes common segmentation metrics."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        use_prompted_background: bool = False,
        bg_idx: int = 0
    ):
        """
        Args:
            num_classes: Number of semantic classes
            ignore_index: Label value for ignored pixels (e.g., no-data regions)
            use_prompted_background: If True, background is in prompts (GT has background class).
                                    If False, background is injected by segmentor (GT has no background),
                                    and background predictions should be filtered.
            bg_idx: Index of background class in predictions
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_prompted_background = use_prompted_background
        self.bg_idx = bg_idx
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _fast_hist(self, label: np.ndarray, pred: np.ndarray) -> np.ndarray:
        # Layer 1: Filter out no-data regions based on ignore_index
        mask = label != self.ignore_index
        label = label[mask]
        pred = pred[mask]

        # Layer 2: Filter out background predictions if GT has no background class
        # When use_prompted_background=False, background is injected by segmentor
        # and should be filtered since GT doesn't have a true background class
        if not self.use_prompted_background:
            # Keep only pixels where prediction is not background
            mask_bg = pred != self.bg_idx
            label = label[mask_bg]
            pred = pred[mask_bg]

        # Validate that labels and preds are in expected range
        # For use_prompted_background=False: valid labels/preds are [1, num_classes]
        # For use_prompted_background=True: valid labels/preds are [0, num_classes-1]
        if not self.use_prompted_background:
            if np.any((label < 0) | (label > self.num_classes)):
                raise ValueError(f"GT labels out of range: min={label.min()}, max={label.max()}, expected [1, {self.num_classes}]")
            if np.any((pred < 0) | (pred > self.num_classes)):
                raise ValueError(f"Predictions out of range: min={pred.min()}, max={pred.max()}, expected [1, {self.num_classes}]")
            # Map [1, num_classes] to [0, num_classes-1] for confusion matrix indexing
            label = label - 1
            pred = pred - 1
        else:
            if np.any((label < 0) | (label >= self.num_classes)):
                raise ValueError(f"GT labels out of range: min={label.min()}, max={label.max()}, expected [0, {self.num_classes-1}]")
            if np.any((pred < 0) | (pred >= self.num_classes)):
                raise ValueError(f"Predictions out of range: min={pred.min()}, max={pred.max()}, expected [0, {self.num_classes-1}]")

        # Handle empty case
        if label.size == 0:
            return np.zeros_like(self.confusion_matrix)

        idx = self.num_classes * label + pred
        hist = np.bincount(idx, minlength=self.num_classes ** 2)
        hist = hist.reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        self.confusion_matrix += self._fast_hist(label, pred)

    def compute(self) -> Dict[str, float]:
        hist = self.confusion_matrix
        tp = np.diag(hist).astype(np.float64)
        pos_gt = hist.sum(axis=1).astype(np.float64)
        pos_pred = hist.sum(axis=0).astype(np.float64)

        eps = 1e-10
        iou = tp / (pos_gt + pos_pred - tp + eps)
        acc = tp / (pos_gt + eps)

        m_iou = float(np.nanmean(iou))
        m_acc = float(np.nanmean(acc))
        a_acc = float(tp.sum() / (hist.sum() + eps))

        return {
            "mIoU": m_iou,
            "mAcc": m_acc,
            "aAcc": a_acc,
        }

    def reset(self) -> None:
        self.confusion_matrix[...] = 0

    def per_class_iou(self) -> np.ndarray:
        hist = self.confusion_matrix
        tp = np.diag(hist).astype(np.float64)
        pos_gt = hist.sum(axis=1).astype(np.float64)
        pos_pred = hist.sum(axis=0).astype(np.float64)
        eps = 1e-10
        return tp / (pos_gt + pos_pred - tp + eps)

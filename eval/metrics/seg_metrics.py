"""Lightweight streaming metrics for semantic segmentation."""

from __future__ import annotations

import numpy as np
from typing import Dict


class SegmentationMetric:
    """Accumulates confusion matrix and computes common segmentation metrics."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _fast_hist(self, label: np.ndarray, pred: np.ndarray) -> np.ndarray:
        mask = label != self.ignore_index
        label = label[mask]
        pred = pred[mask]

        # Drop any labels/preds outside [0, num_classes-1]
        valid = (label >= 0) & (label < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        if not np.any(valid):
            return np.zeros_like(self.confusion_matrix)

        label = label[valid]
        pred = pred[valid]

        idx = self.num_classes * label + pred
        hist = np.bincount(idx, minlength=self.num_classes ** 2)
        if hist.size < self.num_classes ** 2:
            padded = np.zeros(self.num_classes ** 2, dtype=hist.dtype)
            padded[: hist.size] = hist
            hist = padded
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

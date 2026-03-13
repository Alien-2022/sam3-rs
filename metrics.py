"""
Evaluation metrics for remote sensing semantic segmentation.

Implements common metrics:
- Pixel Accuracy
- mIoU (Mean Intersection over Union)
- Class-wise IoU
- F1-score

No heavy dependencies, just numpy!
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result container for metrics"""
    pixel_accuracy: float
    mean_iou: float
    class_iou: Dict[int, float]
    f1_score: float

    def print_summary(self, class_names: Optional[Dict[int, str]] = None):
        """Pretty print metrics"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"Pixel Accuracy: {self.pixel_accuracy:.4f}")
        print(f"Mean IoU (mIoU): {self.mean_iou:.4f}")
        print(f"F1-Score: {self.f1_score:.4f}")

        print(f"\nClass-wise IoU:")
        for class_id in sorted(self.class_iou.keys()):
            class_name = class_names.get(class_id, f"Class_{class_id}") if class_names else f"Class_{class_id}"
            iou = self.class_iou[class_id]
            print(f"  {class_id:2d} ({class_name:20s}): {iou:.4f}")

        print(f"{'='*60}\n")


class RSEvaluator:
    """
    Remote sensing segmentation evaluator.

    Lightweight, no MMSegmentation dependency.

    Usage:
        evaluator = RSEvaluator(num_classes=9)

        # Accumulate predictions
        for pred_path, gt_path in zip(preds, gts):
            evaluator.update(pred_path, gt_path)

        # Get metrics
        results = evaluator.compute()
        results.print_summary(class_names)
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        """
        Args:
            num_classes: Number of classes (including background)
            ignore_index: Label to ignore in evaluation (e.g., 255)
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Accumulators
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0

    def _calculate_iou(self, pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
        """Calculate IoU for a single class."""
        # True positives, False positives, False negatives
        tp = np.sum((pred == class_id) & (gt == class_id))
        fp = np.sum((pred == class_id) & (gt != class_id))
        fn = np.sum((pred != class_id) & (gt == class_id))

        # Intersection over Union
        union = tp + fp + fn
        if union == 0:
            return 1.0  # Both empty, perfect match
        return tp / union

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1-score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def update(self, pred_path: str, gt_path: str):
        """
        Update confusion matrix with a single prediction.

        Args:
            pred_path: Path to prediction image (should be class IDs)
            gt_path: Path to ground truth image (should be class IDs)
        """
        # Load images
        from PIL import Image
        import warnings

        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))

        # Handle ignore index
        valid_mask = gt != self.ignore_index

        # Flatten
        pred = pred[valid_mask].astype(np.int64)
        gt = gt[valid_mask].astype(np.int64)

        # Warn and clip out-of-range class IDs to avoid silent index errors
        pred_oob = (pred < 0) | (pred >= self.num_classes)
        gt_oob   = (gt  < 0) | (gt  >= self.num_classes)
        if pred_oob.any():
            warnings.warn(
                f"Prediction contains {pred_oob.sum()} pixel(s) with class ID "
                f"outside [0, {self.num_classes - 1}] (min={pred[pred_oob].min()}, "
                f"max={pred[pred_oob].max()}). They will be clipped.",
                UserWarning,
                stacklevel=2,
            )
        if gt_oob.any():
            warnings.warn(
                f"Ground truth contains {gt_oob.sum()} pixel(s) with class ID "
                f"outside [0, {self.num_classes - 1}] (min={gt[gt_oob].min()}, "
                f"max={gt[gt_oob].max()}). They will be clipped.",
                UserWarning,
                stacklevel=2,
            )

        pred = np.clip(pred, 0, self.num_classes - 1)
        gt   = np.clip(gt,   0, self.num_classes - 1)

        # Vectorized confusion matrix update using np.bincount (O(n) vs O(n²))
        idx = self.num_classes * gt + pred
        hist = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes)

        self.total_pixels += len(pred)
        self.correct_pixels += int(np.sum(pred == gt))

    def update_batch(self, pred_paths: List[str], gt_paths: List[str]):
        """
        Update with batch of predictions.

        Args:
            pred_paths: List of prediction paths
            gt_paths: List of ground truth paths
        """
        for pred_path, gt_path in zip(pred_paths, gt_paths):
            self.update(pred_path, gt_path)

    def compute(self) -> MetricResult:
        """
        Compute all metrics from accumulated confusion matrix.

        Returns:
            MetricResult with all metrics
        """
        # Pixel accuracy
        pixel_acc = self.correct_pixels / self.total_pixels

        # Class-wise IoU
        class_iou = {}
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp
            union = tp + fp + fn

            if union == 0:
                iou = 1.0
            else:
                iou = tp / union

            class_iou[i] = iou

        # Mean IoU (mIoU)
        mean_iou = np.mean(list(class_iou.values()))

        # F1-Score (macro average)
        # Calculate precision and recall per class
        precisions = []
        recalls = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i, :].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        # Macro F1
        f1_scores = [
            self._calculate_f1(p, r)
            for p, r in zip(precisions, recalls)
        ]
        f1_score = np.mean(f1_scores)

        return MetricResult(
            pixel_accuracy=pixel_acc,
            mean_iou=mean_iou,
            class_iou=class_iou,
            f1_score=f1_score
        )

    def save_results(self, results: MetricResult, output_path: str, class_names: Optional[Dict[int, str]] = None):
        """
        Save results to JSON file.

        Args:
            results: MetricResult from compute()
            output_path: Path to save JSON
            class_names: Optional mapping from class ID to name
        """
        import json

        # Prepare data
        data = {
            'pixel_accuracy': float(results.pixel_accuracy),
            'mean_iou': float(results.mean_iou),
            'f1_score': float(results.f1_score),
            'class_iou': {int(k): float(v) for k, v in results.class_iou.items()}
        }

        if class_names:
            data['class_names'] = {
                int(k): v for k, v in class_names.items()
            }

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved results to {output_path}")


# ============ Utility Functions ============

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.

    Args:
        mask1: First binary mask (0 or class ID)
        mask2: Second binary mask (0 or class ID)

    Returns:
        IoU value
    """
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))

    if union == 0:
        return 1.0
    return intersection / union


def compute_metrics_for_directory(
    pred_dir: str,
    gt_dir: str,
    num_classes: int,
    ignore_index: int = 255,
    output_path: Optional[str] = None
) -> MetricResult:
    """
    Compute metrics for all predictions in a directory.

    Convenience function for quick evaluation.

    Args:
        pred_dir: Directory with prediction images
        gt_dir: Directory with ground truth images
        num_classes: Number of classes
        ignore_index: Label to ignore
        output_path: Optional path to save results JSON

    Returns:
        MetricResult

    Example:
        results = compute_metrics_for_directory(
            pred_dir='outputs/preds/',
            gt_dir='data/OpenEarthMap/val/masks/',
            num_classes=9,
            output_path='outputs/metrics.json'
        )
    """
    evaluator = RSEvaluator(num_classes=num_classes, ignore_index=ignore_index)

    # Get matching files
    import os
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('_pred.png')])

    for pred_file in pred_files:
        # Get base name
        base_name = pred_file.replace('_pred.png', '')
        pred_path = os.path.join(pred_dir, pred_file)

        # Find corresponding GT
        gt_file = base_name + '.tif'  # Adjust based on dataset
        gt_path = os.path.join(gt_dir, gt_file)

        if os.path.exists(gt_path):
            evaluator.update(pred_path, gt_path)
        else:
            print(f"Warning: GT not found for {base_name}")

    results = evaluator.compute()

    if output_path:
        evaluator.save_results(results, output_path)

    return results

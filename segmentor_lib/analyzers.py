"""Analyzer hooks for SAM3-RS inference.

This module provides analyzer classes for collecting and reporting
statistics during inference. Analyzers follow a hook-based design
and can be extended for various analysis tasks.

Available analyzers:
- PresenceScoreAnalyzer: Collects presence score statistics
- StatisticsAnalyzer: Comprehensive evaluation statistics analysis
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict


class BaseAnalyzer(ABC):
    """Base class for inference analyzers."""

    @abstractmethod
    def on_before_inference(self, context: Dict[str, Any]):
        """Called before starting inference on an image/batch."""
        pass

    @abstractmethod
    def on_after_prompt(self, context: Dict[str, Any]):
        """Called after processing each prompt."""
        pass

    @abstractmethod
    def on_after_inference(self, context: Dict[str, Any]):
        """Called after completing inference on an image/batch."""
        pass

    def on_after_eval(self, context: Dict[str, Any]):
        """Called after evaluation of a single image (with GT and prediction).
        
        This hook is called by run_eval.py after computing metrics for each image.
        Context should contain:
        - image_name: str
        - gt_mask: np.ndarray (ground truth mask)
        - pred_mask: np.ndarray (prediction mask)
        - logits: np.ndarray (optional, raw logits)
        - class_names: List[str]
        - ignore_index: int
        """
        pass

    def report(self):
        """Print analysis results."""
        pass

    def reset(self):
        """Reset all collected statistics."""
        pass


class PresenceScoreAnalyzer(BaseAnalyzer):
    """Analyzer for collecting and reporting presence score statistics."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stats = []

    def on_before_inference(self, context: Dict[str, Any]):
        """No action needed before inference."""
        pass

    def on_after_prompt(self, context: Dict[str, Any]):
        """Collect presence score after each prompt."""
        if not self.enabled:
            return

        output = context.get("output")
        image_name = context.get("image_name", "unknown")
        prompt_word = context.get("prompt_word", "unknown")

        # Extract presence score from output
        if output is not None:
            if "presence_score" in output:
                # Single view mode
                ps = output["presence_score"]
                self.stats.append({
                    "image": image_name,
                    "prompt": prompt_word,
                    "presence_score": ps
                })
            elif "presence_logit_dec" in output:
                # Batch mode
                img_presence = output["presence_logit_dec"].sigmoid().max(dim=1)[0]
                for img_idx, ps in enumerate(img_presence):
                    image_names = context.get("image_names")
                    img_name = image_names[img_idx] if image_names else f"img_{img_idx}"
                    self.stats.append({
                        "image": img_name,
                        "prompt": prompt_word,
                        "presence_score": ps.item()
                    })

    def on_after_inference(self, context: Dict[str, Any]):
        """No action needed after inference."""
        pass

    def report(self):
        """Print presence score statistics."""
        if not self.enabled or not self.stats:
            print("[DEBUG] Presence score analysis disabled or no data collected")
            return

        import numpy as np

        all_scores = [s["presence_score"] for s in self.stats]

        print("=" * 80)
        print("PRESENCE SCORE ANALYSIS")
        print("=" * 80)

        # Overall statistics
        print(f"\nTotal samples: {len(self.stats)}")
        print(f"Mean: {np.mean(all_scores):.4f}")
        print(f"Median: {np.median(all_scores):.4f}")
        print(f"Std: {np.std(all_scores):.4f}")
        print(f"Min: {np.min(all_scores):.4f}")
        print(f"Max: {np.max(all_scores):.4f}")

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95]
        print("\nPercentiles:")
        for p in percentiles:
            print(f"  {p}%: {np.percentile(all_scores, p):.4f}")

        # Distribution
        bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        print("\nDistribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in all_scores if bins[i] <= s < bins[i+1])
            ratio = count / len(all_scores) if all_scores else 0
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} ({ratio:6.2%})")
        count = sum(1 for s in all_scores if s == 1.0)
        ratio = count / len(all_scores) if all_scores else 0
        print(f"  [1.0, 1.0]: {count:4d} ({ratio:6.2%})")

        # Per-prompt statistics
        prompt_stats = {}
        for s in self.stats:
            prompt = s["prompt"]
            if prompt not in prompt_stats:
                prompt_stats[prompt] = []
            prompt_stats[prompt].append(s["presence_score"])

        print("\nPer-prompt statistics (sorted by mean):")
        prompt_means = [(p, np.mean(v), len(v), np.min(v), np.max(v), np.std(v))
                        for p, v in prompt_stats.items()]
        prompt_means.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Prompt':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
        print("-" * 80)
        for p, mean, count, pmin, pmax, pstd in prompt_means:
            print(f"{p:<30} {mean:8.4f} {pstd:8.4f} {pmin:8.4f} {pmax:8.4f} {count:6d}")

        print("=" * 80)


class StatisticsAnalyzer(BaseAnalyzer):
    """Comprehensive statistics analyzer for evaluation analysis.
    
    This analyzer collects and analyzes:
    1. Per-class IoU, Precision, Recall, F1 distributions
    2. Confusion matrix
    3. Size-stratified performance
    4. Error patterns (FN vs FP)
    5. Class co-occurrence patterns
    
    Usage:
        analyzer = StatisticsAnalyzer(
            num_classes=7,
            class_names=['bg', 'building', 'road', ...],
            ignore_index=255
        )
        
        # After each image evaluation
        analyzer.on_after_eval({
            'image_name': 'xxx.png',
            'gt_mask': gt_array,
            'pred_mask': pred_array,
            ...
        })
        
        # After all images processed
        analyzer.report()
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        enabled: bool = True,
        compute_boundary_iou: bool = False,
        size_bins: Optional[List[float]] = None
    ):
        """
        Args:
            num_classes: Number of classes (excluding ignore_index)
            class_names: List of class names (optional)
            ignore_index: Index to ignore in evaluation
            enabled: Whether to enable analysis
            compute_boundary_iou: Whether to compute boundary IoU (slower)
            size_bins: Bins for size-stratified analysis [0, 0.01, 0.05, 0.1, 1.0]
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.enabled = enabled
        self.compute_boundary_iou = compute_boundary_iou
        self.size_bins = size_bins or [0.0, 0.01, 0.05, 0.1, 0.2, 1.0]
        
        self.reset()

    def reset(self):
        """Reset all collected statistics."""
        # Per-image results
        self.image_results = []
        
        # Confusion matrix: [num_classes, num_classes]
        # confusion_matrix[gt, pred] = count
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        # Per-class statistics accumulation
        self.class_tp = np.zeros(self.num_classes, dtype=np.int64)  # True Positives
        self.class_fp = np.zeros(self.num_classes, dtype=np.int64)  # False Positives
        self.class_fn = np.zeros(self.num_classes, dtype=np.int64)  # False Negatives
        self.class_tn = np.zeros(self.num_classes, dtype=np.int64)  # True Negatives
        
        # Per-class pixel counts
        self.class_pixel_counts = np.zeros(self.num_classes, dtype=np.int64)
        
        # Size-stratified: {class_id: {size_bin: {'tp', 'fp', 'fn', 'count'}}}
        self.size_stratified = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}))
        
        # Class co-occurrence: counts of images where both classes appear
        self.cooccurrence_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        # Boundary IoU (if enabled)
        if self.compute_boundary_iou:
            self.boundary_tp = np.zeros(self.num_classes, dtype=np.int64)
            self.boundary_fp = np.zeros(self.num_classes, dtype=np.int64)
            self.boundary_fn = np.zeros(self.num_classes, dtype=np.int64)

    def on_before_inference(self, context: Dict[str, Any]):
        """No action needed before inference."""
        pass

    def on_after_prompt(self, context: Dict[str, Any]):
        """No action needed after prompt."""
        pass

    def on_after_inference(self, context: Dict[str, Any]):
        """No action needed after inference."""
        pass

    def on_after_eval(self, context: Dict[str, Any]):
        """Called after evaluation of a single image.
        
        Args:
            context: Dict containing:
                - image_name: str
                - gt_mask: np.ndarray (H, W)
                - pred_mask: np.ndarray (H, W)
                - logits: np.ndarray (optional, C, H, W)
        """
        if not self.enabled:
            return
        
        image_name = context.get("image_name", "unknown")
        gt_mask = context.get("gt_mask")
        pred_mask = context.get("pred_mask")
        
        if gt_mask is None or pred_mask is None:
            return
        
        # Ensure masks are numpy arrays
        if not isinstance(gt_mask, np.ndarray):
            gt_mask = np.array(gt_mask)
        if not isinstance(pred_mask, np.ndarray):
            pred_mask = np.array(pred_mask)
        
        # Create valid mask (ignore ignore_index)
        valid_mask = (gt_mask != self.ignore_index)
        gt_valid = gt_mask[valid_mask]
        pred_valid = pred_mask[valid_mask]
        
        # 1. Update confusion matrix
        for gt_cls in range(self.num_classes):
            for pred_cls in range(self.num_classes):
                count = np.sum((gt_valid == gt_cls) & (pred_valid == pred_cls))
                self.confusion_matrix[gt_cls, pred_cls] += count
        
        # 2. Update per-class statistics
        for cls in range(self.num_classes):
            gt_mask_cls = (gt_valid == cls)
            pred_mask_cls = (pred_valid == cls)
            
            tp = np.sum(gt_mask_cls & pred_mask_cls)
            fp = np.sum(~gt_mask_cls & pred_mask_cls)
            fn = np.sum(gt_mask_cls & ~pred_mask_cls)
            tn = np.sum(~gt_mask_cls & ~pred_mask_cls)
            
            self.class_tp[cls] += tp
            self.class_fp[cls] += fp
            self.class_fn[cls] += fn
            self.class_tn[cls] += tn
            
            self.class_pixel_counts[cls] += np.sum(gt_mask_cls)
        
        # 3. Size-stratified analysis (per connected component)
        self._update_size_stratified(gt_mask, pred_mask, valid_mask)
        
        # 4. Class co-occurrence
        present_classes = np.unique(gt_valid)
        # Filter out class IDs that are out of range
        present_classes = present_classes[present_classes < self.num_classes]
        for c1 in present_classes:
            for c2 in present_classes:
                self.cooccurrence_matrix[c1, c2] += 1
        
        # 5. Per-image result for later analysis
        self._collect_image_result(image_name, gt_mask, pred_mask, valid_mask)
        
        # 6. Boundary IoU (if enabled)
        if self.compute_boundary_iou:
            self._update_boundary_stats(gt_mask, pred_mask, valid_mask)

    def _update_size_stratified(self, gt_mask: np.ndarray, pred_mask: np.ndarray, valid_mask: np.ndarray):
        """Update size-stratified statistics for each class."""
        try:
            from scipy import ndimage
        except ImportError:
            return  # Skip if scipy not available
        
        for cls in range(self.num_classes):
            # Get connected components in GT
            gt_class_mask = (gt_mask == cls) & valid_mask
            labeled, num_components = ndimage.label(gt_class_mask)
            
            for comp_id in range(1, num_components + 1):
                component_mask = (labeled == comp_id)
                component_size = np.sum(component_mask)
                total_pixels = np.sum(valid_mask)
                size_ratio = component_size / total_pixels
                
                # Find which size bin this component belongs to
                bin_idx = np.searchsorted(self.size_bins, size_ratio, side='right') - 1
                bin_idx = max(0, min(bin_idx, len(self.size_bins) - 2))
                size_bin = f"{self.size_bins[bin_idx]:.2f}-{self.size_bins[bin_idx+1]:.2f}"
                
                # Calculate metrics for this component
                pred_class_mask = (pred_mask == cls)
                tp = np.sum(component_mask & pred_class_mask)
                fp = np.sum(component_mask & ~pred_class_mask)  # Actually FN for this component
                fn = np.sum(~component_mask & pred_class_mask & valid_mask)  # Not counting this
                
                # For this component: TP = correctly predicted as this class within component
                # FN = pixels in GT component not predicted as this class
                fn_component = np.sum(component_mask & ~pred_class_mask)
                
                self.size_stratified[cls][size_bin]['tp'] += tp
                self.size_stratified[cls][size_bin]['fn'] += fn_component
                self.size_stratified[cls][size_bin]['count'] += 1

    def _collect_image_result(self, image_name: str, gt_mask: np.ndarray, 
                               pred_mask: np.ndarray, valid_mask: np.ndarray):
        """Collect per-image statistics for correlation analysis."""
        gt_valid = gt_mask[valid_mask]
        pred_valid = pred_mask[valid_mask]
        
        # Compute per-class IoU for this image
        per_class_iou = {}
        per_class_acc = {}
        
        for cls in range(self.num_classes):
            gt_cls = (gt_valid == cls)
            pred_cls = (pred_valid == cls)
            
            intersection = np.sum(gt_cls & pred_cls)
            union = np.sum(gt_cls | pred_cls)
            
            if union > 0:
                per_class_iou[cls] = intersection / union
            else:
                per_class_iou[cls] = float('nan')  # Class not present
            
            # Accuracy for this class
            correct = np.sum(gt_cls & pred_cls)
            total = np.sum(gt_cls)
            per_class_acc[cls] = correct / total if total > 0 else float('nan')
        
        # Overall metrics
        valid_pixels = np.sum(valid_mask)
        correct_pixels = np.sum(gt_valid == pred_valid)
        
        result = {
            'image_name': image_name,
            'per_class_iou': per_class_iou,
            'per_class_acc': per_class_acc,
            'pixel_acc': correct_pixels / valid_pixels if valid_pixels > 0 else 0,
            'num_classes_present': len([c for c in range(self.num_classes) if np.sum(gt_valid == c) > 0]),
            'classes_present': [c for c in range(self.num_classes) if np.sum(gt_valid == c) > 0],
            'image_size': gt_mask.shape
        }
        
        self.image_results.append(result)

    def _update_boundary_stats(self, gt_mask: np.ndarray, pred_mask: np.ndarray, valid_mask: np.ndarray):
        """Update boundary IoU statistics."""
        try:
            from scipy import ndimage
        except ImportError:
            return
        
        # Define boundary width (1-2 pixels)
        boundary_width = 2
        
        for cls in range(self.num_classes):
            gt_class = (gt_mask == cls) & valid_mask
            pred_class = (pred_mask == cls) & valid_mask
            
            # Get boundaries using morphological operations
            gt_eroded = ndimage.binary_erosion(gt_class, iterations=boundary_width)
            gt_boundary = gt_class & ~gt_eroded
            
            pred_eroded = ndimage.binary_erosion(pred_class, iterations=boundary_width)
            pred_boundary = pred_class & ~pred_eroded
            
            # Boundary IoU
            self.boundary_tp[cls] += np.sum(gt_boundary & pred_boundary)
            self.boundary_fp[cls] += np.sum(~gt_boundary & pred_boundary)
            self.boundary_fn[cls] += np.sum(gt_boundary & ~pred_boundary)

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics from collected statistics."""
        metrics = {}
        
        # 1. Per-class metrics
        per_class_iou = []
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for cls in range(self.num_classes):
            tp = self.class_tp[cls]
            fp = self.class_fp[cls]
            fn = self.class_fn[cls]
            
            # IoU
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            per_class_iou.append(iou)
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            per_class_precision.append(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class_recall.append(recall)
            
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class_f1.append(f1)
        
        metrics['per_class_iou'] = {self.class_names[i]: per_class_iou[i] for i in range(self.num_classes)}
        metrics['per_class_precision'] = {self.class_names[i]: per_class_precision[i] for i in range(self.num_classes)}
        metrics['per_class_recall'] = {self.class_names[i]: per_class_recall[i] for i in range(self.num_classes)}
        metrics['per_class_f1'] = {self.class_names[i]: per_class_f1[i] for i in range(self.num_classes)}
        
        # 2. Overall metrics
        metrics['mIoU'] = np.nanmean(per_class_iou)
        metrics['mPrecision'] = np.nanmean(per_class_precision)
        metrics['mRecall'] = np.nanmean(per_class_recall)
        metrics['mF1'] = np.nanmean(per_class_f1)
        
        # Pixel accuracy
        total_pixels = np.sum(self.class_tp) + np.sum(self.class_fp)
        metrics['aAcc'] = np.sum(self.class_tp) / total_pixels if total_pixels > 0 else 0.0
        
        # 3. Confusion matrix (normalized by GT)
        confusion_normalized = self.confusion_matrix.astype(np.float64)
        row_sums = confusion_normalized.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion_normalized = confusion_normalized / row_sums
        metrics['confusion_matrix'] = self.confusion_matrix.tolist()
        metrics['confusion_matrix_normalized'] = confusion_normalized.tolist()
        
        # 4. Size-stratified IoU
        size_metrics = {}
        for cls in range(self.num_classes):
            size_metrics[self.class_names[cls]] = {}
            for size_bin, stats in self.size_stratified[cls].items():
                tp, fn = stats['tp'], stats['fn']
                count = stats['count']
                # Simplified IoU approximation for components
                iou = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                size_metrics[self.class_names[cls]][size_bin] = {
                    'iou': iou,
                    'count': count
                }
        metrics['size_stratified'] = size_metrics
        
        # 5. Top confusions
        confusions = []
        for gt_cls in range(self.num_classes):
            for pred_cls in range(self.num_classes):
                if gt_cls != pred_cls:
                    count = self.confusion_matrix[gt_cls, pred_cls]
                    if count > 0:
                        confusions.append({
                            'gt_class': self.class_names[gt_cls],
                            'pred_class': self.class_names[pred_cls],
                            'count': int(count),
                            'error_rate': count / (self.confusion_matrix[gt_cls, :].sum() + 1e-8)
                        })
        confusions.sort(key=lambda x: x['count'], reverse=True)
        metrics['top_confusions'] = confusions[:10]
        
        # 6. Class co-occurrence
        metrics['cooccurrence_matrix'] = self.cooccurrence_matrix.tolist()
        
        # 7. Boundary IoU (if enabled)
        if self.compute_boundary_iou:
            boundary_iou = []
            for cls in range(self.num_classes):
                tp = self.boundary_tp[cls]
                fp = self.boundary_fp[cls]
                fn = self.boundary_fn[cls]
                biou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                boundary_iou.append(biou)
            metrics['per_class_boundary_iou'] = {self.class_names[i]: boundary_iou[i] for i in range(self.num_classes)}
            metrics['mBoundaryIoU'] = np.nanmean(boundary_iou)
        
        return metrics

    def report(self, detailed: bool = True):
        """Print comprehensive analysis report."""
        if not self.enabled or len(self.image_results) == 0:
            print("[StatisticsAnalyzer] Disabled or no data collected")
            return
        
        metrics = self.compute_metrics()
        
        print("=" * 80)
        print("                    SAM3-RS EVALUATION ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\nDataset Statistics:")
        print(f"  Images analyzed: {len(self.image_results)}")
        print(f"  Number of classes: {self.num_classes}")
        
        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  mIoU:       {metrics['mIoU']:.4f}")
        print(f"  mPrecision: {metrics['mPrecision']:.4f}")
        print(f"  mRecall:    {metrics['mRecall']:.4f}")
        print(f"  mF1:        {metrics['mF1']:.4f}")
        print(f"  aAcc:       {metrics['aAcc']:.4f}")
        
        if self.compute_boundary_iou:
            print(f"  mBoundaryIoU: {metrics['mBoundaryIoU']:.4f}")
        
        # Per-class analysis
        print("\n" + "-" * 80)
        print("1. CLASS-LEVEL ANALYSIS")
        print("-" * 80)
        
        print(f"\n{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Pixels':>12} {'Samples':>8}")
        print("-" * 80)
        
        for cls in range(self.num_classes):
            name = self.class_names[cls]
            iou = metrics['per_class_iou'][name]
            prec = metrics['per_class_precision'][name]
            rec = metrics['per_class_recall'][name]
            f1 = metrics['per_class_f1'][name]
            pixels = self.class_pixel_counts[cls]
            samples = self.cooccurrence_matrix[cls, cls]
            
            flag = " ⚠️" if iou < 0.4 else ""
            print(f"{name:<20} {iou:8.4f} {prec:8.4f} {rec:8.4f} {f1:8.4f} {pixels:12d} {samples:8d}{flag}")
        
        # Error type analysis (FN vs FP)
        print(f"\n{'Class':<20} {'FN Rate':>10} {'FP Rate':>10} {'Error Type':<20}")
        print("-" * 80)
        
        for cls in range(self.num_classes):
            name = self.class_names[cls]
            tp = self.class_tp[cls]
            fp = self.class_fp[cls]
            fn = self.class_fn[cls]
            
            fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
            fp_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
            
            if fn_rate > fp_rate + 0.1:
                error_type = "Miss-heavy (FN)"
            elif fp_rate > fn_rate + 0.1:
                error_type = "Hallucination (FP)"
            else:
                error_type = "Balanced"
            
            print(f"{name:<20} {fn_rate:10.2%} {fp_rate:10.2%} {error_type:<20}")
        
        # Confusion analysis
        print("\n" + "-" * 80)
        print("2. CONFUSION MATRIX (Top Confusions)")
        print("-" * 80)
        
        print(f"\n{'GT Class':<20} → {'Predicted As':<20} {'Count':>10} {'Error Rate':>12}")
        print("-" * 80)
        
        for conf in metrics['top_confusions'][:10]:
            print(f"{conf['gt_class']:<20} → {conf['pred_class']:<20} {conf['count']:10d} {conf['error_rate']:11.2%}")
        
        # Size-stratified analysis
        print("\n" + "-" * 80)
        print("3. SIZE-STRATIFIED PERFORMANCE")
        print("-" * 80)
        
        print(f"\n{'Size Range':<15}", end="")
        for cls in range(min(5, self.num_classes)):  # Show first 5 classes
            print(f" {self.class_names[cls][:8]:>10}", end="")
        print("  Count")
        print("-" * 80)
        
        all_bins = set()
        for cls in range(self.num_classes):
            all_bins.update(self.size_stratified[cls].keys())
        
        for size_bin in sorted(all_bins):
            print(f"{size_bin:<15}", end="")
            for cls in range(min(5, self.num_classes)):
                stats = self.size_stratified[cls].get(size_bin, {'tp': 0, 'fn': 0, 'count': 0})
                tp, fn = stats['tp'], stats['fn']
                iou = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                print(f" {iou:10.3f}", end="")
            total_count = sum(self.size_stratified[cls].get(size_bin, {}).get('count', 0) for cls in range(self.num_classes))
            print(f"  {total_count:6d}")
        
        # Key findings
        print("\n" + "-" * 80)
        print("4. KEY FINDINGS")
        print("-" * 80)
        
        findings = self._generate_findings(metrics)
        for i, finding in enumerate(findings, 1):
            print(f"\n{i}. {finding['title']}")
            print(f"   {finding['detail']}")
            if finding.get('suggestion'):
                print(f"   → 建议: {finding['suggestion']}")
        
        print("\n" + "=" * 80)

    def _generate_findings(self, metrics: Dict) -> List[Dict]:
        """Generate key findings from metrics."""
        findings = []
        
        # 1. Low IoU classes
        low_iou_classes = [
            (name, iou) for name, iou in metrics['per_class_iou'].items() 
            if iou < 0.4
        ]
        if low_iou_classes:
            low_iou_classes.sort(key=lambda x: x[1])
            worst = low_iou_classes[0]
            findings.append({
                'title': f"【低性能类别】'{worst[0]}' IoU仅 {worst[1]:.2f}",
                'detail': f"低IoU类别: {', '.join([f'{n}({i:.2f})' for n, i in low_iou_classes])}",
                'suggestion': "检查提示词是否足够区分这些类别，或考虑样本不平衡问题"
            })
        
        # 2. Major confusions
        if metrics['top_confusions']:
            top_conf = metrics['top_confusions'][0]
            findings.append({
                'title': f"【主要混淆】'{top_conf['gt_class']}' → '{top_conf['pred_class']}'",
                'detail': f"错误率 {top_conf['error_rate']:.1%} ({top_conf['count']} 像素)",
                'suggestion': f"检查这两个类别的语义相似性，可能需要更明确的提示词区分"
            })
        
        # 3. Error type analysis
        fn_heavy_classes = []
        fp_heavy_classes = []
        for cls in range(self.num_classes):
            name = self.class_names[cls]
            tp = self.class_tp[cls]
            fp = self.class_fp[cls]
            fn = self.class_fn[cls]
            
            fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
            fp_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
            
            if fn_rate > 0.6:
                fn_heavy_classes.append((name, fn_rate))
            elif fp_rate > 0.6:
                fp_heavy_classes.append((name, fp_rate))
        
        if fn_heavy_classes:
            findings.append({
                'title': f"【漏检严重】{len(fn_heavy_classes)} 个类别漏检率超过60%",
                'detail': f"类别: {', '.join([f'{n}({r:.1%})' for n, r in fn_heavy_classes])}",
                'suggestion': "考虑降低 prob_threshold 或使用语义增强提升召回率"
            })
        
        if fp_heavy_classes:
            findings.append({
                'title': f"【误检严重】{len(fp_heavy_classes)} 个类别误检率超过60%",
                'detail': f"类别: {', '.join([f'{n}({r:.1%})' for n, r in fp_heavy_classes])}",
                'suggestion': "考虑提高 confidence_threshold 或 prob_threshold 降低误检"
            })
        
        # 4. Size analysis
        small_perf = []
        for cls in range(self.num_classes):
            small_bin = self.size_stratified[cls].get('0.00-0.01', {})
            if small_bin.get('count', 0) > 0:
                tp, fn = small_bin.get('tp', 0), small_bin.get('fn', 0)
                iou = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                small_perf.append((self.class_names[cls], iou, small_bin['count']))
        
        if small_perf:
            avg_small_iou = np.mean([p[1] for p in small_perf])
            if avg_small_iou < 0.3:
                findings.append({
                    'title': f"【小目标检测困难】小目标平均IoU仅 {avg_small_iou:.2f}",
                    'detail': f"共 {sum(p[2] for p in small_perf)} 个小目标组件",
                    'suggestion': "考虑使用多尺度推理或调整模型参数"
                })
        
        # 5. Class imbalance
        pixel_counts = self.class_pixel_counts
        if pixel_counts.max() > 0:
            max_ratio = pixel_counts.max() / (pixel_counts.min() + 1)
            if max_ratio > 100:
                findings.append({
                    'title': f"【类别不平衡】最多与最少类别像素比 {max_ratio:.0f}:1",
                    'detail': f"像素数: {dict(zip(self.class_names, pixel_counts.tolist()))}",
                    'suggestion': "考虑使用加权损失或过采样策略"
                })
        
        return findings

    def save_report(self, output_path: str):
        """Save analysis report to JSON file."""
        import json
        
        metrics = self.compute_metrics()
        
        report = {
            'summary': {
                'num_images': len(self.image_results),
                'num_classes': self.num_classes,
                'mIoU': metrics['mIoU'],
                'mPrecision': metrics['mPrecision'],
                'mRecall': metrics['mRecall'],
                'mF1': metrics['mF1'],
                'aAcc': metrics['aAcc'],
            },
            'per_class_metrics': {
                'iou': metrics['per_class_iou'],
                'precision': metrics['per_class_precision'],
                'recall': metrics['per_class_recall'],
                'f1': metrics['per_class_f1'],
                'pixel_counts': dict(zip(self.class_names, self.class_pixel_counts.tolist())),
            },
            'confusion_matrix': metrics['confusion_matrix'],
            'top_confusions': metrics['top_confusions'],
            'size_stratified': metrics['size_stratified'],
            'findings': self._generate_findings(metrics),
        }
        
        if self.compute_boundary_iou:
            report['per_class_boundary_iou'] = metrics['per_class_boundary_iou']
            report['summary']['mBoundaryIoU'] = metrics['mBoundaryIoU']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to: {output_path}")

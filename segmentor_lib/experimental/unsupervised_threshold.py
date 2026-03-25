"""
Unsupervised Threshold Calibration for SAM3-RS

This module implements threshold calibration based on confidence score distributions
from model inference on unlabeled data.

The approach:
1. Collect confidence statistics from a sample of validation/test images (no labels needed)
2. Estimate confidence_threshold from the distribution of instance confidence scores
3. Estimate prob_threshold from the distribution of pixel-level segmentation logits
4. Use percentile-based heuristics to determine appropriate thresholds

改进：增加数据分布特征分析，帮助理解不同数据集的统计特性
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from scipy import stats


class DatasetStatistics:
    """数据集统计特征，用于分析不同数据集的分布特性"""
    
    def __init__(self):
        self.confidence_stats: Dict[str, float] = {}
        self.logit_stats: Dict[int, Dict[str, float]] = {}  # per-class logit stats (all pixels)
        self.global_logit_stats: Dict[str, float] = {}  # global stats (all pixels)
        self.nonzero_logit_stats: Dict[int, Dict[str, float]] = {}  # per-class logit stats (non-zero only)
        self.nonzero_global_logit_stats: Dict[str, float] = {}  # global stats (non-zero only)
        self.distribution_metrics: Dict[str, float] = {}
    
    def to_dict(self) -> Dict:
        """转换为字典格式用于保存"""
        return {
            "confidence": self.confidence_stats,
            "logits_per_class": self.logit_stats,
            "logits_global": self.global_logit_stats,
            "nonzero_logits_per_class": self.nonzero_logit_stats,
            "nonzero_logits_global": self.nonzero_global_logit_stats,
            "distribution": self.distribution_metrics
        }
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 70)
        print("[Dataset Statistics] Data Distribution Analysis")
        print("=" * 70)
        
        # Confidence score statistics
        print("\n--- Confidence Score Distribution ---")
        if self.confidence_stats:
            print(f"  Count:       {self.confidence_stats.get('count', 0):,}")
            print(f"  Min:         {self.confidence_stats.get('min', 0):.4f}")
            print(f"  Mean:        {self.confidence_stats.get('mean', 0):.4f}")
            print(f"  Median:      {self.confidence_stats.get('median', 0):.4f}")
            print(f"  Std:         {self.confidence_stats.get('std', 0):.4f}")
            print(f"  P5:          {self.confidence_stats.get('p5', 0):.4f}")
            print(f"  P25:         {self.confidence_stats.get('p25', 0):.4f}")
            print(f"  P50:         {self.confidence_stats.get('p50', 0):.4f}")
            print(f"  P75:         {self.confidence_stats.get('p75', 0):.4f}")
            print(f"  P95:         {self.confidence_stats.get('p95', 0):.4f}")
            print(f"  Max:         {self.confidence_stats.get('max', 0):.4f}")
            print(f"  Skewness:    {self.confidence_stats.get('skewness', 0):.4f}")
            print(f"  Kurtosis:    {self.confidence_stats.get('kurtosis', 0):.4f}")
        
        # Global logit statistics
        print("\n--- Global Logit Distribution (All Pixels) ---")
        if self.global_logit_stats:
            print(f"  Count:       {self.global_logit_stats.get('count', 0):,}")
            print(f"  Min:         {self.global_logit_stats.get('min', 0):.4f}")
            print(f"  Mean:        {self.global_logit_stats.get('mean', 0):.4f}")
            print(f"  Median:      {self.global_logit_stats.get('median', 0):.4f}")
            print(f"  Std:         {self.global_logit_stats.get('std', 0):.4f}")
            print(f"  P5:          {self.global_logit_stats.get('p5', 0):.4f}")
            print(f"  P25:         {self.global_logit_stats.get('p25', 0):.4f}")
            print(f"  P50:         {self.global_logit_stats.get('p50', 0):.4f}")
            print(f"  P75:         {self.global_logit_stats.get('p75', 0):.4f}")
            print(f"  P95:         {self.global_logit_stats.get('p95', 0):.4f}")
            print(f"  Max:         {self.global_logit_stats.get('max', 0):.4f}")
            print(f"  Skewness:    {self.global_logit_stats.get('skewness', 0):.4f}")
            print(f"  Kurtosis:    {self.global_logit_stats.get('kurtosis', 0):.4f}")
        
        # Non-zero global logit statistics
        print("\n--- Non-Zero Logit Distribution (>0 pixels only) ---")
        if self.nonzero_global_logit_stats:
            print(f"  Count:       {self.nonzero_global_logit_stats.get('count', 0):,}")
            print(f"  Min:         {self.nonzero_global_logit_stats.get('min', 0):.4f}")
            print(f"  Mean:        {self.nonzero_global_logit_stats.get('mean', 0):.4f}")
            print(f"  Median:      {self.nonzero_global_logit_stats.get('median', 0):.4f}")
            print(f"  Std:         {self.nonzero_global_logit_stats.get('std', 0):.4f}")
            print(f"  P5:          {self.nonzero_global_logit_stats.get('p5', 0):.4f}")
            print(f"  P25:         {self.nonzero_global_logit_stats.get('p25', 0):.4f}")
            print(f"  P50:         {self.nonzero_global_logit_stats.get('p50', 0):.4f}")
            print(f"  P75:         {self.nonzero_global_logit_stats.get('p75', 0):.4f}")
            print(f"  P95:         {self.nonzero_global_logit_stats.get('p95', 0):.4f}")
            print(f"  Max:         {self.nonzero_global_logit_stats.get('max', 0):.4f}")
            print(f"  Skewness:    {self.nonzero_global_logit_stats.get('skewness', 0):.4f}")
            print(f"  Kurtosis:    {self.nonzero_global_logit_stats.get('kurtosis', 0):.4f}")
            # 计算非零比例
            if self.global_logit_stats and self.global_logit_stats.get('count', 0) > 0:
                nonzero_ratio = self.nonzero_global_logit_stats.get('count', 0) / self.global_logit_stats.get('count', 1)
                print(f"  Non-zero Ratio: {nonzero_ratio:.2%}")
        
        # Per-class logit statistics
        print("\n--- Per-Class Logit Distribution (All Pixels vs Non-Zero) ---")
        if self.logit_stats:
            print(f"{'Class':<8} {'Type':<12} {'Count':<10} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'NonZero%':<10}")
            print("-" * 100)
            for class_idx in sorted(self.logit_stats.keys()):
                # All pixels
                stats_dict = self.logit_stats[class_idx]
                nonzero_stats = self.nonzero_logit_stats.get(class_idx, {})
                total_count = stats_dict['count']
                nonzero_count = nonzero_stats.get('count', 0)
                nonzero_ratio = nonzero_count / total_count * 100 if total_count > 0 else 0
                
                print(f"{class_idx:<8} {'All':<12} {stats_dict['count']:<10,} "
                      f"{stats_dict['mean']:<10.4f} {stats_dict['median']:<10.4f} "
                      f"{stats_dict['min']:<10.4f} {stats_dict['max']:<10.4f} {nonzero_ratio:<10.2f}")
                
                # Non-zero pixels
                if nonzero_count > 0:
                    print(f"{'':8} {'NonZero':<12} {nonzero_count:<10,} "
                          f"{nonzero_stats.get('mean', 0):<10.4f} {nonzero_stats.get('median', 0):<10.4f} "
                          f"{nonzero_stats.get('min', 0):<10.4f} {nonzero_stats.get('max', 0):<10.4f}")
        
        # Distribution metrics
        print("\n--- Key Distribution Metrics ---")
        if self.distribution_metrics:
            for key, value in self.distribution_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, bool):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        
        print("=" * 70)


class UnsupervisedThresholdCalibration:
    """
    无监督阈值校准：基于验证集的统计分布

    该方法无需标注数据，仅依赖模型输出的置信度分布来估计合理的阈值。
    
    改进：增加详细的数据分布特征分析，帮助理解不同数据集的特性
    """

    def __init__(self, num_samples: int = 50):
        """
        Args:
            num_samples: 用于统计的验证样本数（无需标注）
        """
        self.num_samples = num_samples
        self.confidence_scores: List[float] = []
        self.prob_logits_per_class: Dict[int, List[float]] = {}
        self.dataset_stats = DatasetStatistics()

    def _compute_distribution_stats(self, data: np.ndarray) -> Dict[str, float]:
        """计算详细的分布统计特征"""
        if len(data) == 0:
            return {}
        
        return {
            'count': len(data),
            'min': float(np.min(data)),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'p5': float(np.percentile(data, 5)),
            'p25': float(np.percentile(data, 25)),
            'p50': float(np.percentile(data, 50)),
            'p75': float(np.percentile(data, 75)),
            'p95': float(np.percentile(data, 95)),
            'max': float(np.max(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }

    def collect_statistics(
        self,
        segmentor,
        image_paths: List[str],
        prompt_names: List[str],
        num_classes: int,
        relaxed_threshold: float = 0.1,
        exclude_background: bool = True,
        background_names: List[str] = None,
        analyze_distribution: bool = True,
        batch_size: int = 4,
        force_single_view: bool = False
    ) -> None:
        """
        对验证集的部分图像进行推理，收集统计信息（无需标注）

        Args:
            segmentor: SAM3RSSegmentor 实例
            image_paths: 测试图像路径列表（可以是任意图像，无需标注）
            prompt_names: 提示词名称列表
            num_classes: 类别数
            relaxed_threshold: 临时宽松阈值（默认 0.1）
            exclude_background: 是否排除背景类进行统计（默认 True）
            background_names: 背景类的提示词名称列表（默认 ["background", "bg"]）
            analyze_distribution: 是否进行详细的分布分析（默认 True）
            batch_size: batch推理的批量大小（默认 4）
            force_single_view: 强制使用单图模式（适用于大图需要sliding window的情况）
        """
        segmentor.processor.model.eval()  # Set model to eval mode
        num_samples = min(self.num_samples, len(image_paths))

        # 检测是否需要使用单图模式（大图需要sliding window）
        use_single_view = force_single_view
        if not use_single_view and num_samples > 0:
            # 检查第一张图像的尺寸
            sample_image = Image.open(image_paths[0]).convert("RGB")
            img_w, img_h = sample_image.size
            # 如果配置了sliding window且图像尺寸大于crop_size，使用单图模式
            if hasattr(segmentor.config, 'slide_crop_size') and segmentor.config.slide_crop_size > 0:
                if img_w > segmentor.config.slide_crop_size or img_h > segmentor.config.slide_crop_size:
                    use_single_view = True
                    print(f"  [Auto-detect] Large image ({img_w}x{img_h}) detected, using single-view mode with sliding window")

        # 确定要排除的背景类
        if background_names is None:
            # background_names = ["background", "clutter"]
            background_names = []

        print(f"\n[UnsupervisedCalibration] Collecting statistics from {num_samples} images...")
        print(f"  Batch size: {batch_size}")
        print(f"  Using relaxed threshold ({relaxed_threshold}) for statistics collection")
        if analyze_distribution:
            print(f"  Distribution analysis: enabled")
        if exclude_background:
            print(f"  Excluding background classes from confidence statistics: {background_names}")

        # 保存原始阈值以便后续恢复
        original_confidence_threshold = segmentor.processor.confidence_threshold
        original_prob_threshold = segmentor.config.prob_threshold
        original_prob_thresholds = segmentor.config.prob_thresholds
        # engine.config 和 segmentor.config 是同一个对象，但为了保险也保存一下
        original_engine_confidence = segmentor.engine.config.confidence_threshold

        # ⭐ 第一阶段：使用宽松的临时阈值收集统计
        segmentor.processor.confidence_threshold = relaxed_threshold
        # 设置宽松的prob_threshold以收集更多前景像素的logits
        segmentor.config.prob_threshold = 0.01  # 非常宽松的阈值
        segmentor.config.prob_thresholds = None  # 不使用per-class阈值
        # batch模式使用 engine.config，需要同时修改
        segmentor.engine.config.confidence_threshold = relaxed_threshold

        # 用于分布分析的原始 logits（未经筛选）
        all_raw_logits_per_class: Dict[int, List[float]] = {i: [] for i in range(num_classes)}

        try:
            if use_single_view:
                # ===== 单图模式（适用于大图，会自动使用sliding window） =====
                for img_idx in range(num_samples):
                    image_path = image_paths[img_idx]
                    image_name = f"calib_{img_idx}"

                    if (img_idx + 1) % 10 == 0 or img_idx == 0:
                        print(f"  Progress: {img_idx + 1}/{num_samples}")

                    # Load and predict using predict_single (supports sliding window)
                    with torch.no_grad():
                        result = segmentor.predict_single(
                            image_path,
                            detailed=True  # Need detailed to get per-instance scores
                        )

                    # Collect confidence scores from instances
                    if hasattr(result, 'per_class_results') and result.per_class_results:
                        for prompt_name in prompt_names:
                            # Skip background class if exclude_background is True
                            if exclude_background and prompt_name.lower() in [bg.lower() for bg in background_names]:
                                continue

                            if prompt_name in result.per_class_results:
                                class_data = result.per_class_results[prompt_name]
                                if "scores" in class_data and class_data["scores"] is not None:
                                    # Collect instance confidence scores
                                    scores_cpu = class_data["scores"].float().cpu().numpy()
                                    self.confidence_scores.extend(scores_cpu.tolist())

                    # Collect per-class logits for prob_threshold estimation
                    if result.seg_logits is not None:
                        seg_logits = result.seg_logits.float().cpu().numpy()  # [num_prompts, H, W]

                        # Map prompts to classes (handle synonyms)
                        num_prompts = seg_logits.shape[0]

                        for prompt_idx in range(min(num_prompts, num_classes)):
                            # Skip background class if exclude_background is True
                            if exclude_background and prompt_idx < len(prompt_names):
                                prompt_name = prompt_names[prompt_idx]
                                if prompt_name.lower() in [bg.lower() for bg in background_names]:
                                    continue

                            class_logits = seg_logits[prompt_idx].flatten()

                            # 收集原始 logits 用于分布分析
                            if analyze_distribution:
                                # 随机采样一部分原始 logits（避免内存爆炸）
                                if len(class_logits) > 10000:
                                    sampled_raw = np.random.choice(class_logits, size=10000, replace=False)
                                else:
                                    sampled_raw = class_logits
                                all_raw_logits_per_class[prompt_idx].extend(sampled_raw.tolist())

                            # Sample top 5% logits for threshold estimation
                            # Note: This collects high-confidence regions only (top 5% per class)
                            # which explains why Prob Threshold mean is higher than Global Logit mean
                            p95 = np.percentile(class_logits, 95)
                            high_logits = class_logits[class_logits >= p95]

                            if len(high_logits) > 0:
                                # Sample up to 1000 values randomly from top 5%
                                sample_size = min(1000, len(high_logits))
                                if len(high_logits) > sample_size:
                                    sampled_logits = np.random.choice(high_logits, size=sample_size, replace=False)
                                else:
                                    sampled_logits = high_logits

                                if prompt_idx not in self.prob_logits_per_class:
                                    self.prob_logits_per_class[prompt_idx] = []
                                self.prob_logits_per_class[prompt_idx].extend(sampled_logits.tolist())
            else:
                # ===== Batch模式（适用于小图） =====
                num_batches = (num_samples + batch_size - 1) // batch_size
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    current_batch_size = end_idx - start_idx

                    # 每个batch都更新进度
                    processed = min(end_idx, num_samples)
                    progress_pct = processed / num_samples * 100
                    print(f"  Progress: {processed}/{num_samples} ({progress_pct:.1f}%) - batch {batch_idx + 1}/{num_batches}")

                    # 加载当前 batch 的图像
                    batch_image_paths = image_paths[start_idx:end_idx]
                    batch_images = [Image.open(p).convert("RGB") for p in batch_image_paths]
                    batch_image_names = [f"calib_{i}" for i in range(start_idx, end_idx)]

                    # Run inference with detailed mode using batch_view
                    with torch.no_grad():
                        batch_seg_logits, batch_per_class_results, _, _, _ = segmentor.inference_batch_view(
                            images=batch_images,
                            detailed=True,  # Need detailed to get per-instance scores
                            image_names=batch_image_names
                        )

                    # batch_seg_logits: [B, num_prompts, H, W]
                    # batch_per_class_results: List[Dict] of length B

                    # Process each image in the batch
                    for b in range(current_batch_size):
                        img_idx = start_idx + b
                        per_class_results = batch_per_class_results[b]
                        seg_logits = batch_seg_logits[b]  # [num_prompts, H, W]

                        # Collect confidence scores from instances
                        if per_class_results:
                            for prompt_name in prompt_names:
                                # Skip background class if exclude_background is True
                                if exclude_background and prompt_name.lower() in [bg.lower() for bg in background_names]:
                                    continue

                                if prompt_name in per_class_results:
                                    class_data = per_class_results[prompt_name]
                                    if "scores" in class_data and class_data["scores"] is not None and len(class_data["scores"]) > 0:
                                        # Collect instance confidence scores
                                        scores_cpu = class_data["scores"].float().cpu().numpy()
                                        self.confidence_scores.extend(scores_cpu.tolist())

                        # Collect per-class logits for prob_threshold estimation
                        if seg_logits is not None:
                            seg_logits_np = seg_logits.float().cpu().numpy()  # [num_prompts, H, W]
                            num_prompts = seg_logits_np.shape[0]

                            for prompt_idx in range(min(num_prompts, num_classes)):
                                # Skip background class if exclude_background is True
                                if exclude_background and prompt_idx < len(prompt_names):
                                    prompt_name = prompt_names[prompt_idx]
                                    if prompt_name.lower() in [bg.lower() for bg in background_names]:
                                        continue

                                class_logits = seg_logits_np[prompt_idx].flatten()

                                # 收集原始 logits 用于分布分析
                                if analyze_distribution:
                                    # 随机采样一部分原始 logits（避免内存爆炸）
                                    if len(class_logits) > 10000:
                                        sampled_raw = np.random.choice(class_logits, size=10000, replace=False)
                                    else:
                                        sampled_raw = class_logits
                                    all_raw_logits_per_class[prompt_idx].extend(sampled_raw.tolist())

                                # Sample top 5% logits for threshold estimation
                                # Note: This collects high-confidence regions only (top 5% per class)
                                # which explains why Prob Threshold mean is higher than Global Logit mean
                                p95 = np.percentile(class_logits, 95)
                                high_logits = class_logits[class_logits >= p95]

                                if len(high_logits) > 0:
                                    # Sample up to 1000 values randomly from top 5%
                                    sample_size = min(1000, len(high_logits))
                                    if len(high_logits) > sample_size:
                                        sampled_logits = np.random.choice(high_logits, size=sample_size, replace=False)
                                    else:
                                        sampled_logits = high_logits

                                    if prompt_idx not in self.prob_logits_per_class:
                                        self.prob_logits_per_class[prompt_idx] = []
                                    self.prob_logits_per_class[prompt_idx].extend(sampled_logits.tolist())

                    # 清理 batch 数据以释放内存
                    del batch_images, batch_seg_logits, batch_per_class_results
                    if batch_idx % 2 == 0:  # 每2个batch清理一次
                        torch.cuda.empty_cache()

        finally:
            # ⭐ 恢复原始阈值
            segmentor.processor.confidence_threshold = original_confidence_threshold
            segmentor.config.prob_threshold = original_prob_threshold
            segmentor.config.prob_thresholds = original_prob_thresholds
            segmentor.engine.config.confidence_threshold = original_engine_confidence

        print(f"[UnsupervisedCalibration] Collection complete!")
        print(f"  - Total confidence scores collected: {len(self.confidence_scores)}")
        print(f"  - Classes with logit data: {len(self.prob_logits_per_class)}")

        # 计算分布统计特征
        if analyze_distribution:
            self._compute_distribution_statistics(all_raw_logits_per_class)
            self.dataset_stats.print_summary()

    def _compute_distribution_statistics(self, raw_logits_per_class: Dict[int, List[float]]):
        """计算数据集分布统计特征（包括 all 和 non-zero 两种统计）"""
        # Confidence score 统计
        if self.confidence_scores:
            conf_array = np.array(self.confidence_scores)
            self.dataset_stats.confidence_stats = self._compute_distribution_stats(conf_array)

        # Per-class logit 统计 (all pixels)
        for class_idx, logits_list in raw_logits_per_class.items():
            if len(logits_list) > 0:
                logits_array = np.array(logits_list)
                self.dataset_stats.logit_stats[class_idx] = self._compute_distribution_stats(logits_array)
                
                # Non-zero logit 统计 (>0.01，筛选有实际响应的像素，避免大量接近0的背景像素淹没统计)
                # Note: sigmoid 输出范围是(0,1)，理论上不会严格为0，但大部分背景像素值极低(<0.01)
                nonzero_threshold = 0.01
                nonzero_logits = logits_array[logits_array > nonzero_threshold]
                if len(nonzero_logits) > 0:
                    self.dataset_stats.nonzero_logit_stats[class_idx] = self._compute_distribution_stats(nonzero_logits)

        # Global logit 统计 (all pixels)
        all_logits = []
        for logits_list in raw_logits_per_class.values():
            all_logits.extend(logits_list)
        if all_logits:
            all_logits_array = np.array(all_logits)
            self.dataset_stats.global_logit_stats = self._compute_distribution_stats(all_logits_array)
            
            # Non-zero global logit 统计 (>0.01，筛选有实际响应的像素)
            # Note: sigmoid 输出范围是(0,1)，理论上不会严格为0，但大部分背景像素值极低(<0.01)
            nonzero_threshold = 0.01
            nonzero_all_logits = all_logits_array[all_logits_array > nonzero_threshold]
            if len(nonzero_all_logits) > 0:
                self.dataset_stats.nonzero_global_logit_stats = self._compute_distribution_stats(nonzero_all_logits)

            # 计算关键分布指标（优先使用 non-zero 统计，更准确）
            stats_dict = None
            if self.dataset_stats.nonzero_global_logit_stats:
                stats_dict = self.dataset_stats.nonzero_global_logit_stats
                stats_source = "nonzero"
            elif self.dataset_stats.global_logit_stats:
                stats_dict = self.dataset_stats.global_logit_stats
                stats_source = "all"
            
            if stats_dict:
                # Top 5% vs 整体分布的差异
                # 接近 1 表示 top 5% 范围很小（分布集中），接近 0 表示范围很大（分布分散）
                top5_percent = 1.0 - (stats_dict['p95'] - stats_dict['min']) / (stats_dict['max'] - stats_dict['min'] + 1e-8)
                
                # 分布集中度（标准差/均值）
                # 值越小表示分布越集中，值越大表示分布越分散
                # LoveDA 通常较大（分布分散），Potsdam 通常较小（分布集中）
                concentration = stats_dict['std'] / (stats_dict['mean'] + 1e-8)
                
                # 分布偏态（负值左偏，正值右偏）
                # > 0: 右偏（长尾在右侧），适合 top 5% 采样（如 LoveDA）
                # < 0: 左偏（长尾在左侧），不适合 top 5% 采样
                # ≈ 0: 对称分布
                skewness = stats_dict['skewness']
                
                # 分布峰态（正值更尖峭，负值更平坦）
                # > 3: 尖峭分布（数据集中在均值附近）
                # < 3: 平坦分布（数据比较分散）
                # scipy 默认返回 excess kurtosis（相对于正态分布的超出量）
                kurtosis = stats_dict['kurtosis']
                
                # 低置信度占比标志（P5 < 0.1）
                # True: 有大量低置信度像素（适合 top 5% 采样，如 LoveDA）
                # False: 低置信度像素少（不适合 top 5% 采样，如 Potsdam）
                low_conf_ratio = (stats_dict['p5'] < 0.1)
                
                # 中位数相对于均值的偏差
                # 值越大表示分布越不对称
                # > 0.5: 中位数和均值差异大（分布不对称）
                # < 0.5: 中位数和均值接近（分布较对称）
                median_deviation = abs(stats_dict['median'] - stats_dict['mean']) / (stats_dict['mean'] + 1e-8)
                
                # 非零像素比例（如果适用）
                if self.dataset_stats.global_logit_stats and self.dataset_stats.nonzero_global_logit_stats:
                    nonzero_ratio = self.dataset_stats.nonzero_global_logit_stats['count'] / self.dataset_stats.global_logit_stats['count']
                else:
                    nonzero_ratio = 1.0
                
                self.dataset_stats.distribution_metrics = {
                    'stats_source': stats_source,  # 指示使用的是哪种统计
                    'top5_percent_range': top5_percent,
                    'concentration': concentration,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'low_conf_ratio': low_conf_ratio,
                    'median_deviation': median_deviation,
                    'nonzero_ratio': nonzero_ratio,
                    # 判断数据集类型
                    'is_sparse_annotation': skewness > 0 and low_conf_ratio and nonzero_ratio < 0.5,  # 右偏且低分位数低且稀疏 → 稀疏标注
                    'is_dense_annotation': concentration < 0.5 and stats_dict['median'] > 0.3 and nonzero_ratio > 0.5,  # 集中且中位数高且密集 → 密集标注
                }

    def estimate_confidence_threshold(
        self,
        percentile: float = 30.0,
        min_threshold: float = 0.1,
        max_threshold: float = 0.8
    ) -> float:
        """
        基于 confidence score 分布估计 confidence_threshold

        Args:
            percentile: 使用哪个百分位数（默认 30% 表示保留前 70% 的高置信度实例）
            min_threshold: 最小阈值限制
            max_threshold: 最大阈值限制

        Returns:
            estimated_threshold: float

        原理:
          - 低置信度实例往往是误检
          - 取 30 百分位意味着"至少 30% 的实例被过滤"
          - 这在无标注情况下是合理的保守估计
        """
        if len(self.confidence_scores) == 0:
            print("  [Warning] No confidence scores collected. Using default 0.1")
            return 0.1

        confidence_array = np.array(self.confidence_scores)

        # Compute statistics
        min_val = confidence_array.min()
        mean_val = confidence_array.mean()
        median_val = np.median(confidence_array)
        p30_val = np.percentile(confidence_array, percentile)
        max_val = confidence_array.max()

        # Apply threshold
        threshold = np.clip(p30_val, min_threshold, max_threshold)

        print(f"\n[UnsupervisedCalibration] Confidence Score Statistics:")
        print(f"  Min:       {min_val:.4f}")
        print(f"  Mean:      {mean_val:.4f}")
        print(f"  Median:    {median_val:.4f}")
        print(f"  {percentile:.0f}%:      {p30_val:.4f} ← threshold")
        print(f"  Max:       {max_val:.4f}")
        print(f"  Final:     {threshold:.4f}")

        return float(threshold)

    def estimate_prob_threshold_per_class(
        self,
        percentile_low: float = 40.0,
        percentile_high: float = 60.0,
        min_threshold: float = 0.01,
        max_threshold: float = 0.95
    ) -> Dict[int, float]:
        """
        为每个类别单独估计 prob_threshold

        基于该类别在验证集上的 logit 分布

        Args:
            percentile_low: 低置信度百分位
            percentile_high: 高置信度百分位
            min_threshold: 最小阈值限制
            max_threshold: 最大阈值限制

        Returns:
            thresholds_dict: {class_idx: threshold}
        """
        if not self.prob_logits_per_class:
            print("  [Warning] No logit data collected. Returning empty dict")
            return {}

        thresholds = {}

        print(f"\n[UnsupervisedCalibration] Per-Class Prob Threshold Statistics:")
        print(f"{'Class':<8} {'Count':<8} {'Min':<10} {'P40':<10} {'Median':<10} {'P60':<10} {'Max':<10} {'Threshold':<10}")
        print("-" * 90)

        for class_idx in sorted(self.prob_logits_per_class.keys()):
            logits_list = self.prob_logits_per_class[class_idx]

            if len(logits_list) == 0:
                thresholds[class_idx] = 0.5
                continue

            logits_array = np.array(logits_list)

            # Compute statistics
            min_val = logits_array.min()
            mean_val = logits_array.mean()
            p40 = np.percentile(logits_array, percentile_low)
            median = np.median(logits_array)
            p60 = np.percentile(logits_array, percentile_high)
            max_val = logits_array.max()

            # Use median as threshold (balances false positives and false negatives)
            threshold = np.clip(median, min_threshold, max_threshold)
            thresholds[class_idx] = threshold

            print(f"{class_idx:<8} {len(logits_list):<8} {min_val:<10.4f} {p40:<10.4f} "
                  f"{median:<10.4f} {p60:<10.4f} {max_val:<10.4f} {threshold:<10.4f}")

        return thresholds

    def estimate_global_prob_threshold(
        self,
        percentile: float = 50.0,
        min_threshold: float = 0.01,
        max_threshold: float = 0.95
    ) -> float:
        """
        估计全局 prob_threshold（所有类别共用一个）

        Args:
            percentile: 百分位（50 = 中位数）
            min_threshold: 最小阈值限制
            max_threshold: 最大阈值限制

        Returns:
            threshold: float
        """
        if not self.prob_logits_per_class:
            print("  [Warning] No logit data collected. Using default 0.5")
            return 0.5

        # Merge all class logits
        all_logits = []
        for logits_list in self.prob_logits_per_class.values():
            all_logits.extend(logits_list)

        if len(all_logits) == 0:
            print("  [Warning] No logit data collected. Using default 0.5")
            return 0.5

        all_logits_array = np.array(all_logits)
        threshold = np.percentile(all_logits_array, percentile)
        threshold = np.clip(threshold, min_threshold, max_threshold)

        print(f"\n[UnsupervisedCalibration] Global Prob Threshold Statistics (Top 5% sampled):")
        print(f"  Count:     {len(all_logits)}")
        print(f"  Min:       {all_logits_array.min():.4f}")
        print(f"  Mean:      {all_logits_array.mean():.4f}")
        print(f"  Median:    {np.median(all_logits_array):.4f}")
        print(f"  {percentile:.0f}%:      {threshold:.4f} ← threshold")
        print(f"  Max:       {all_logits_array.max():.4f}")

        return float(threshold)

    def calibrate(
        self,
        segmentor,
        image_paths: List[str],
        prompt_names: List[str],
        num_classes: int,
        confidence_percentile: float = 30.0,
        prob_percentile: float = 50.0,
        use_per_class_prob: bool = False,
        exclude_background: bool = True,
        background_names: List[str] = None,
        relaxed_threshold: float = 0.01,
        analyze_distribution: bool = True,
        batch_size: int = 4,
        force_single_view: bool = False
    ) -> Tuple[float, Optional[Dict[int, float]]]:
        """
        一步完成所有校准

        Args:
            segmentor: SAM3RSSegmentor 实例
            image_paths: 校准图像路径列表
            prompt_names: 提示词名称列表
            num_classes: 类别数
            confidence_percentile: confidence threshold 的百分位
            prob_percentile: prob threshold 的百分位
            use_per_class_prob: 是否为每个类别单独估计 prob threshold
            exclude_background: 是否排除背景类进行统计（默认 True）
            background_names: 背景类的提示词名称列表（默认 ["background", "bg"]）
            relaxed_threshold: 统计收集时的临时宽松阈值（默认 0.1）
            analyze_distribution: 是否进行详细的分布分析（默认 True）
            batch_size: batch推理的批量大小（默认 4，小图如LoveDA适用）
            force_single_view: 强制使用单图模式（适用于大图需要sliding window的情况，如Potsdam）

        Returns:
            (confidence_threshold, prob_thresholds)
            - confidence_threshold: float
            - prob_thresholds: None (if use_per_class_prob=False) or Dict[int, float]
        """
        print("=" * 70)
        print("[UnsupervisedCalibration] Starting threshold calibration...")
        print("=" * 70)

        # Collect statistics
        self.collect_statistics(
            segmentor, image_paths, prompt_names, num_classes,
            relaxed_threshold=relaxed_threshold,
            exclude_background=exclude_background,
            background_names=background_names,
            analyze_distribution=analyze_distribution,
            batch_size=batch_size,
            force_single_view=force_single_view
        )

        # Estimate confidence threshold
        confidence_threshold = self.estimate_confidence_threshold(
            percentile=confidence_percentile
        )

        # Estimate prob threshold(s)
        if use_per_class_prob:
            prob_thresholds = self.estimate_prob_threshold_per_class()
        else:
            prob_threshold = self.estimate_global_prob_threshold(
                percentile=prob_percentile
            )
            prob_thresholds = None  # Use global threshold

        print("=" * 70)
        print("[UnsupervisedCalibration] Calibration complete!")
        print(f"  Confidence Threshold: {confidence_threshold:.4f}")
        if use_per_class_prob:
            print(f"  Prob Thresholds (per-class): {prob_thresholds}")
        else:
            print(f"  Prob Threshold (global): {prob_threshold:.4f}")
        print("=" * 70)

        return confidence_threshold, prob_thresholds

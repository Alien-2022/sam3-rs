"""
Lightweight SAM3 wrapper for remote sensing semantic segmentation.

Core features:
- No heavy dependencies (no MMSegmentation required)
- Flexible prompt management
- Sliding window for large images
- Easy to extend and modify
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from segmentor_lib.core import InferenceEngine
from segmentor_lib.prompts import load_prompts, precompute_text_features
from segmentor_lib.postprocess import fuse_prompts_to_classes, logits_to_pred, resize_logits
from segmentor_lib.sliding_window import SlidingWindowInference
from segmentor_lib.debug import MemoryDebugger


@dataclass
class InferenceConfig:
    """Configuration for SAM3-RS inference"""

    # Model paths
    checkpoint_path: str
    bpe_path: str

    # Inference parameters
    device: str = "cuda"
    confidence_threshold: float = 0.5
    # To return a mask, prob_threshold must be >0 to have effect
    prob_threshold: float = 0.1
    # Background class index in the output (should match GT mask labels)
    # Default: 0. Set this to match your dataset's background class ID.
    bg_idx: Optional[int] = 0
    # Whether SAM prompts include background as a separate category.
    # - False (default): Background is handled via prob_threshold filtering.
    #   SAM only segments foreground categories; pixels below prob_threshold are assigned to bg_idx.
    # - True: Background is also prompted to SAM (e.g., "background" in prompts.txt).
    #   SAM will explicitly segment background as a category. bg_idx should match the line number.
    use_prompted_background: bool = False

    # Head selection (SegEarthOV3 innovation)
    use_semantic_head: bool = True
    use_instance_head: bool = True
    use_presence_score: bool = False

    # Presence score application mode
    # - "after_fusion" (default): apply to fused result after dual-head fusion
    # - "before_fusion": apply to semantic head separately before fusion with instance head
    # Note: instance head already includes presence_score in SAM3's internal computation
    presence_score_mode: str = "after_fusion"
    # presence_score_mode: str = "before_fusion"

    # Semantic enhancement in text space
    # - False (default): use all synonyms separately and fuse results (original behavior)
    # - "select_word": select the most representative synonym for each class
    # - "avg_embedding": use average embedding of synonyms for each class
    # If True (legacy): equivalent to "select_word" mode
    semantic_enhancement_mode: str = "false"  # "false", "select_word", "avg_embedding"
    # Legacy alias for backward compatibility
    use_semantic_enhancement: bool = False

    # Adaptive threshold strategy
    # - None (default): use fixed thresholds (confidence_threshold, prob_threshold)
    # - 'presence': adjust thresholds based on presence score
    # - 'class_specific': use class-specific base thresholds
    # - 'hybrid': combine class-specific + presence score adaptation
    # - 'image_feature': adjust based on image feature complexity
    adaptive_threshold_strategy: Optional[str] = None

    # Class-specific threshold configurations (used with 'class_specific' or 'hybrid' strategies)
    # Format: {class_id: {'confidence_threshold': float, 'prob_threshold': float}}
    # Example: {1: {'confidence_threshold': 0.15, 'prob_threshold': 0.03}}
    class_thresholds: Optional[Dict[int, Dict[str, float]]] = None

    # Large image handling
    slide_crop_size: int = 0  # 0 means no sliding
    slide_stride: int = 512

    # Text prompts
    prompts_file: Optional[str] = None  # Path to prompts config

    # Debug options
    debug_memory: bool = False  # Print memory usage information
    debug_log_file: Optional[str] = None  # Path to log file for debug output
    analyze_presence_score: bool = False  # Analyze and print presence score statistics


@dataclass
class SegmentationResult:
    """Result structure for each image"""

    image_path: str
    seg_logits: torch.Tensor  # [num_classes, H, W]
    seg_pred: torch.Tensor  # [H, W] with class IDs
    # {prompt_name: {masks, boxes, scores, ...}}
    per_class_results: Dict[str, Dict]

    # Individual head logits (for analysis / visualization)
    semantic_logits: Optional[torch.Tensor] = None  # [num_classes, H, W]
    instance_logits: Optional[torch.Tensor] = None  # [num_classes, H, W]


class SAM3RSSegmentor:
    """
    Lightweight SAM3 wrapper for remote sensing zero-shot segmentation.

    Key improvements over vanilla SAM3:
    1. Dual-head fusion (instance + semantic)
    2. Presence score filtering
    3. Sliding window for large images
    4. Multi-class prompt batching
    5. Remote sensing specific optimizations
    """

    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: InferenceConfig object with all parameters
        """
        self.config = config
        self.device = torch.device(config.device)

        # Check config validity
        if not (0.0 < config.prob_threshold < 1.0):
            raise ValueError("prob_threshold must be between 0 and 1.")

        # Initialize memory debugger
        self.debugger = MemoryDebugger(
            enabled=config.debug_memory if hasattr(config, 'debug_memory') else False,
            log_file=config.debug_log_file
        )

        # Initialize SAM3 model (follow SegEarthOV3's approach)
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            bpe_path=config.bpe_path,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
        )
        # Ensure model weights on target device to avoid CPU/GPU dtype mismatches
        model = model.to(self.device)
        self.processor = Sam3Processor(
            model,
            confidence_threshold=config.confidence_threshold,
            device=self.device,
        )

        # Load prompts if provided
        self.prompts = load_prompts(config.prompts_file)
        self.num_classes = max(self.prompts["indices"]) + 1 if self.prompts else 0
        self.num_prompts = len(self.prompts["names"]) if self.prompts else 0

        # Convert class indices to tensor
        if self.prompts:
            self.query_indices = torch.tensor(
                self.prompts["indices"], dtype=torch.int64, device=self.device
            )
            print(
                f"✓ SAM3-RS initialized with {self.num_classes} classes, {self.num_prompts} prompts"
            )

        # Apply semantic enhancement if enabled
        # Support legacy use_semantic_enhancement flag and new semantic_enhancement_mode
        enhancement_mode = getattr(config, 'semantic_enhancement_mode', 'false').lower()
        if config.use_semantic_enhancement and enhancement_mode == 'false':
            # Legacy mode: default to select_word
            enhancement_mode = 'select_word'

        if enhancement_mode == 'select_word':
            print("Using semantic enhancement: select representative word mode")
            from segmentor_lib.experimental.semantic_enhancement import apply_semantic_enhancement
            self.prompts = apply_semantic_enhancement(
                self.processor, self.prompts, self.num_classes, self.device
            )
            self.num_prompts = len(self.prompts["names"])
            # Pre-compute text features for the selected representative words
            self.text_features_cache = precompute_text_features(
                self.processor, self.prompts, self.device, self.num_prompts
            )
        elif enhancement_mode == 'avg_embedding':
            print("Using semantic enhancement: average embedding mode")
            from segmentor_lib.experimental.semantic_enhancement import apply_semantic_enhancement_with_avg_embedding
            self.prompts = apply_semantic_enhancement_with_avg_embedding(
                self.processor, self.prompts, self.num_classes, self.device
            )
            self.num_prompts = len(self.prompts["names"])
            # Skip precompute_text_features since we already have avg_embeddings
            self.text_features_cache = None
        else:
            # No semantic enhancement: pre-compute text features normally
            self.text_features_cache = precompute_text_features(
                self.processor, self.prompts, self.device, self.num_prompts
            )

        # Initialize inference engine with analyzers
        self.engine = InferenceEngine(
            processor=self.processor,
            prompts=self.prompts,
            config=config,
            device=self.device
        )
        self.engine.text_features_cache = self.text_features_cache


    def _inference_single_view(
        self, image: Image.Image, detailed: bool = False, image_name: str = "unknown"
    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Inference on a single image (or crop patch).

        Returns:
            seg_logits: [num_prompts, H, W] fused segmentation logits
            per_class_results: {prompt_name: {masks, boxes, scores, ...}} (empty if not detailed)
            semantic_logits_only: [num_prompts, H, W] from semantic head only (None if not enabled)
            instance_logits_only: [num_prompts, H, W] from instance head only (None if not enabled)
        """
        return self.engine.inference_single_view(image, detailed, image_name)

    def _inference_batch_view(self, images: List[Image.Image], detailed: bool = False, image_names: Optional[List[str]] = None):
        """
        高性能批量推理实现 (True Batch Inference)。
        直接调用底层 model.backbone 和 model.forward_grounding，绕过处理器内部的单图限制。
        """
        return self.engine.inference_batch_view(images, detailed, image_names)


    def _sliding_window_inference(self, image: Image.Image, detailed: bool = False,
                                image_name: str = "unknown"):
        """Run sliding window inference."""
        sliding_window = SlidingWindowInference(
            inference_func=self._inference_single_view,
            crop_size=self.config.slide_crop_size,
            stride=self.config.slide_stride,
            device=self.device
        )
        return sliding_window(image, detailed, image_name)

    def predict_single(
        self, image_path: str, detailed: bool = False
    ) -> SegmentationResult:
        """
        Predict segmentation for a single image.

        Args:
            image_path: Path to input image
            detailed: Whether to return per-instance results (per_class_results)

        Returns:
            SegmentationResult with predictions, including separate semantic_pred and instance_pred
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_shape = (image.height, image.width)
        image_name = os.path.basename(image_path)

        # Choose inference mode
        per_class_results = None
        adaptive_prob_thresholds = None
        if self.config.slide_crop_size > 0 and (
            self.config.slide_crop_size < image.width
            or self.config.slide_crop_size < image.height
        ):
            # Use sliding window for large images
            seg_logits, per_class_results, semantic_logits_only, instance_logits_only, adaptive_prob_thresholds = self._sliding_window_inference(
                image, detailed=detailed, image_name=image_name
            )
        else:
            # Single view inference
            seg_logits, per_class_results, semantic_logits_only, instance_logits_only, adaptive_prob_thresholds = self._inference_single_view(
                image, detailed=detailed, image_name=image_name
            )

        # Resize logits to original shape
        seg_logits = resize_logits(seg_logits, original_shape)
        semantic_logits_only = resize_logits(semantic_logits_only, original_shape) if semantic_logits_only is not None else None
        instance_logits_only = resize_logits(instance_logits_only, original_shape) if instance_logits_only is not None else None

        # Clear cache after sliding window (helps with large images)
        torch.cuda.empty_cache()

        # Map prompts to actual class IDs (handle synonyms)
        if self.num_classes != self.num_prompts:
            seg_logits = fuse_prompts_to_classes(seg_logits, self.query_indices, self.num_classes, self.num_prompts)

        # Get final prediction (argmax)
        bg_idx = 0 if self.config.bg_idx is None else self.config.bg_idx

        # Note: When use_prompted_background=True, background class is treated
        # equally with other classes in argmax (no longer forced to 0).
        # This allows the model to predict background as a valid class.

        # Use adaptive prob_threshold if available
        if adaptive_prob_thresholds is not None:
            from segmentor_lib.experimental.adaptive_threshold import logits_to_pred_adaptive
            seg_pred = logits_to_pred_adaptive(
                seg_logits, self.config.use_prompted_background,
                adaptive_prob_thresholds, self.config.prob_threshold, bg_idx
            )
        else:
            seg_pred = logits_to_pred(seg_logits, self.config.use_prompted_background, self.config.prob_threshold, bg_idx)

        # Fuse individual head logits
        semantic_logits = None
        instance_logits = None

        if semantic_logits_only is not None:
            if self.num_classes != self.num_prompts:
                semantic_logits_only = fuse_prompts_to_classes(semantic_logits_only, self.query_indices, self.num_classes, self.num_prompts)
            semantic_logits = semantic_logits_only

        if instance_logits_only is not None:
            if self.num_classes != self.num_prompts:
                instance_logits_only = fuse_prompts_to_classes(instance_logits_only, self.query_indices, self.num_classes, self.num_prompts)
            instance_logits = instance_logits_only

        # Clear intermediate tensors to save memory
        del semantic_logits_only, instance_logits_only
        torch.cuda.empty_cache()

        # Prepare result
        result = SegmentationResult(
            image_path=image_path,
            seg_logits=seg_logits,
            seg_pred=seg_pred,
            per_class_results=per_class_results,
            semantic_logits=semantic_logits,
            instance_logits=instance_logits,
        )

        return result

    def predict_batch(
        self,
        image_paths: List[str],
        detailed: bool = False,
    ) -> List[SegmentationResult]:
        """
        Predict segmentation for a batch of images.

        Args:
            image_paths: List of image paths
            detailed: Whether to return detailed per-instance results

        Returns:
            List of SegmentationResult
        """
        # Load images
        images = [Image.open(p).convert("RGB") for p in image_paths]
        image_names = [os.path.basename(p) for p in image_paths]
        # Batch reference for sliding windows is not currently supported.
        try:
            seg_logits, _, _, _, adaptive_prob_thresholds = self._inference_batch_view(
                images, detailed=detailed, image_names=image_names
            )
        finally:
            # 确保即使推理失败也清理 images 中的显存引用
            del images

        # Post-process for Each Batch Item
        batch_size = len(image_paths)
        results = []

        # 1. Map prompts to actual class IDs (handle synonyms)
        if self.num_classes != self.num_prompts:
            # seg_logits: [B, num_prompts, H, W]
            # cls_index: [num_cls, num_prompts, 1, 1]
            cls_index = F.one_hot(self.query_indices, num_classes=self.num_classes).T
            cls_index = cls_index.view(self.num_classes, self.num_prompts, 1, 1)

            # (B, 1, num_prompts, H, W) * (1, num_cls, num_prompts, 1, 1) -> max(dim=2)
            # Result: [B, num_cls, H, W]
            seg_logits = (seg_logits.unsqueeze(1) * cls_index.unsqueeze(0)).max(2)[0]

        # 2. Get final prediction (argmax)
        bg_idx = 0 if self.config.bg_idx is None else self.config.bg_idx

        # If background is in prompts but we want to exclude it from metric,
        # set its logits to 0 so argmax won't select it
        if self.config.use_prompted_background:
            seg_logits = seg_logits.clone()
            seg_logits[:, bg_idx] = 0

        # Unify prediction logic across batch
        # seg_logits: [B, num_classes, H, W]
        if not self.config.use_prompted_background:
            bg_pad = torch.zeros(
                (batch_size, 1, *seg_logits.shape[2:]),
                device=seg_logits.device,
                dtype=seg_logits.dtype,
            )
            logits_for_argmax = torch.cat([bg_pad, seg_logits], dim=1)
            seg_pred = torch.argmax(logits_for_argmax, dim=1)
        else:
            seg_pred = torch.argmax(seg_logits, dim=1)

        # Apply prob_threshold filtering (adaptive or fixed)
        if adaptive_prob_thresholds is not None:
            # Use per-class adaptive thresholds
            for class_id in range(self.num_classes):
                threshold = adaptive_prob_thresholds.get(class_id, self.config.prob_threshold)
                class_logits = seg_logits[:, class_id]
                # If argmax selected this class but logits are below threshold, assign to background
                mask = (seg_pred == class_id) & (class_logits < threshold)
                seg_pred[mask] = bg_idx
        else:
            # Use fixed threshold for all classes
            max_vals = seg_logits.max(1)[0]
            seg_pred[max_vals < self.config.prob_threshold] = bg_idx

        # 3. Package results
        for b in range(batch_size):
            results.append(
                SegmentationResult(
                    image_path=image_paths[b],
                    seg_logits=seg_logits[b],
                    seg_pred=seg_pred[b],
                    per_class_results={},  # Detailed results not supported in batch mode yet
                    semantic_logits=None,
                    instance_logits=None,
                )
            )

        return results
"""Multi-scale inference module for SAM3-RS.

Remote sensing images contain objects at highly varying scales (e.g., small cars
vs. large croplands). Running inference at multiple resolutions and merging the
logits improves segmentation quality across all object sizes.

Usage:
    from segmentor_lib.multi_scale import MultiScaleInference

    multi_scale = MultiScaleInference(
        inference_func=segmentor._inference_single_view,
        scales=[0.5, 1.0, 1.5],
        merge_mode="avg",
        device=device,
    )
    seg_logits, per_class, semantic, instance, adaptive = multi_scale(image)
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Callable, Dict, List, Optional, Tuple


class MultiScaleInference:
    """Multi-scale test-time inference for remote sensing segmentation.

    Runs inference at multiple image scales, resizes each output back to the
    original resolution, and merges them with a configurable strategy.

    Attributes:
        inference_func: Callable with the same signature as
            ``InferenceEngine.inference_single_view``.
        scales: Relative scale factors (1.0 = original size).
        merge_mode: How to combine multi-scale logits – ``"avg"`` (default)
            or ``"max"``.
        device: Target torch device.
    """

    MERGE_MODES = ("avg", "max")

    def __init__(
        self,
        inference_func: Callable,
        scales: List[float] = (0.5, 1.0, 1.5),
        merge_mode: str = "avg",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            inference_func: Function that takes ``(image, detailed, image_name)``
                and returns
                ``(seg_logits, per_class_results, semantic_logits, instance_logits,
                adaptive_prob_thresholds)``.
            scales: Scale factors to use.  Must include 1.0 to always process
                the original resolution.
            merge_mode: ``"avg"`` – average logits across scales (default);
                ``"max"`` – element-wise maximum (sharpens boundaries).
            device: torch device for accumulator tensors.

        Raises:
            ValueError: If an unsupported merge_mode is supplied.
        """
        if merge_mode not in self.MERGE_MODES:
            raise ValueError(
                f"Unknown merge_mode '{merge_mode}'. "
                f"Supported modes: {self.MERGE_MODES}"
            )

        self.inference_func = inference_func
        self.scales = list(scales)
        self.merge_mode = merge_mode
        self.device = device or torch.device("cpu")

    def __call__(
        self,
        image: Image.Image,
        detailed: bool = False,
        image_name: str = "unknown",
    ) -> Tuple[
        torch.Tensor,
        Dict,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict],
    ]:
        """Run multi-scale inference on a single image.

        Args:
            image: PIL Image (any size).
            detailed: Whether to return per-instance per-class results.
            image_name: Identifier string for logging.

        Returns:
            Tuple of:
                seg_logits: ``[num_prompts, H, W]`` merged logits.
                per_class_results: From the 1.0-scale inference (or empty dict).
                semantic_logits: Merged semantic-head logits, or ``None``.
                instance_logits: Merged instance-head logits, or ``None``.
                adaptive_prob_thresholds: From the 1.0-scale inference, or ``None``.
        """
        orig_w, orig_h = image.size
        orig_shape = (orig_h, orig_w)

        seg_accum: Optional[torch.Tensor] = None
        semantic_accum: Optional[torch.Tensor] = None
        instance_accum: Optional[torch.Tensor] = None
        scale_count = 0

        # Results from the 1.0 (or closest) scale kept for detailed output
        canonical_per_class: Dict = {}
        canonical_adaptive: Optional[Dict] = None

        for scale in self.scales:
            # Resize image to target scale
            if scale == 1.0:
                scaled_image = image
            else:
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                scaled_image = image.resize((new_w, new_h), Image.BILINEAR)

            scale_name = f"{image_name}_scale{scale:.2f}"
            logits, per_class, sem_logits, inst_logits, adaptive = self.inference_func(
                scaled_image, detailed=(detailed and scale == 1.0), image_name=scale_name
            )

            # Keep canonical outputs from the 1.0-scale run
            if scale == 1.0:
                canonical_per_class = per_class or {}
                canonical_adaptive = adaptive

            # Up/down-sample logits back to original resolution
            logits = self._resize(logits, orig_shape)
            sem_logits = self._resize(sem_logits, orig_shape) if sem_logits is not None else None
            inst_logits = self._resize(inst_logits, orig_shape) if inst_logits is not None else None

            if self.merge_mode == "avg":
                seg_accum = logits if seg_accum is None else seg_accum + logits
                if sem_logits is not None:
                    semantic_accum = sem_logits if semantic_accum is None else semantic_accum + sem_logits
                if inst_logits is not None:
                    instance_accum = inst_logits if instance_accum is None else instance_accum + inst_logits
            else:  # "max"
                seg_accum = logits if seg_accum is None else torch.maximum(seg_accum, logits)
                if sem_logits is not None:
                    semantic_accum = sem_logits if semantic_accum is None else torch.maximum(semantic_accum, sem_logits)
                if inst_logits is not None:
                    instance_accum = inst_logits if instance_accum is None else torch.maximum(instance_accum, inst_logits)

            scale_count += 1

            # Free scaled image to conserve memory
            if scale != 1.0:
                del scaled_image

        # Release GPU cache once after all scales are processed
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Normalise average
        if self.merge_mode == "avg" and scale_count > 0:
            seg_accum = seg_accum / scale_count
            if semantic_accum is not None:
                semantic_accum = semantic_accum / scale_count
            if instance_accum is not None:
                instance_accum = instance_accum / scale_count

        return (
            seg_accum,
            canonical_per_class,
            semantic_accum,
            instance_accum,
            canonical_adaptive,
        )

    @staticmethod
    def _resize(
        tensor: torch.Tensor, target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Bilinear-resize a ``[C, H, W]`` tensor to *target_shape* ``(H, W)``."""
        if tensor.shape[-2:] == target_shape:
            return tensor
        return F.interpolate(
            tensor.unsqueeze(0),
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

"""Sliding window inference module for large images."""

import torch
from PIL import Image
from typing import Tuple, Dict, Optional
from tqdm import tqdm


class SlidingWindowInference:
    """Sliding window inference handler for large images."""

    def __init__(self, inference_func, crop_size: int, stride: int, device, memory_debugger=None):
        """
        Args:
            inference_func: Function that takes (image, detailed, image_name) and returns
                          (seg_logits, per_class_results, semantic_logits, instance_logits, adaptive_prob_thresholds)
            crop_size: Size of sliding window crop
            stride: Stride for sliding window
            device: torch.device
            memory_debugger: Optional MemoryDebugger instance for memory profiling
        """
        self.inference_func = inference_func
        self.crop_size = crop_size
        self.stride = stride
        self.device = device
        self.memory_debugger = memory_debugger

    def __call__(self, image: Image.Image, detailed: bool = False,
                 image_name: str = "unknown") -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
        """
        Run sliding window inference on image.

        Args:
            image: PIL Image
            detailed: Whether to return detailed results
            image_name: Image identifier

        Returns:
            Tuple of (seg_logits, per_class_results, semantic_logits, instance_logits, adaptive_prob_thresholds)
        """
        w, h = image.size

        # Calculate number of crops (ensure full coverage)
        num_crops_x = max(w - self.crop_size + self.stride - 1, 0) // self.stride + 1
        num_crops_y = max(h - self.crop_size + self.stride - 1, 0) // self.stride + 1

        # Initialize accumulation tensors
        seg_logits_sum = None
        count_map = None
        semantic_logits_sum = None
        instance_logits_sum = None
        per_class_results = {}

        # Collect boxes and scores for iterative visual prompt refinement
        # Format: {prompt_name: {"boxes": [N, 4], "scores": [N]}}
        boxes_by_prompt = None
        # Always collect boxes when detailed=True (for visualization/debugging)
        # Only use for iterative refinement when enabled in config
        if detailed:
            boxes_by_prompt = {}

        # Process each crop
        adaptive_prob_thresholds_list = []
        total_crops = num_crops_y * num_crops_x

        crop_idx = 0
        with tqdm(total=total_crops, desc=f"Sliding {image_name[:20]}",
                  leave=False, unit="crop") as pbar:
            for crop_y in range(num_crops_y):
                for crop_x in range(num_crops_x):
                    if self.memory_debugger:
                        self.memory_debugger.debug_print(f"\n[CROP {crop_idx+1}/{total_crops}] y={crop_y}/{num_crops_y-1}, x={crop_x}/{num_crops_x-1}")
                        self.memory_debugger.log_cuda_memory(f"Crop {crop_idx+1} start")

                    # Calculate crop boundaries
                    x1 = crop_x * self.stride
                    y1 = crop_y * self.stride
                    x2 = min(x1 + self.crop_size, w)
                    y2 = min(y1 + self.crop_size, h)

                    # Adjust start points to ensure crop size at boundaries
                    # (Matches official SegEarth-OV3 behavior)
                    x1 = max(x2 - self.crop_size, 0)
                    y1 = max(y2 - self.crop_size, 0)

                    # Extract crop
                    crop = image.crop((x1, y1, x2, y2))

                    # Run inference on crop (always collect detailed results when boxes_by_prompt is set)
                    collect_detailed = (boxes_by_prompt is not None)
                    crop_seg_logits, crop_per_class, crop_semantic, crop_instance, crop_adaptive = self.inference_func(
                        crop, detailed=collect_detailed, image_name=f"{image_name}_crop_{crop_y}_{crop_x}"
                    )

                    if self.memory_debugger:
                        self.memory_debugger.log_cuda_memory(f"Crop {crop_idx+1} after inference")

                    # Collect adaptive thresholds from first crop (they should be similar across crops)
                    if crop_adaptive is not None and not adaptive_prob_thresholds_list:
                        adaptive_prob_thresholds_list.append(crop_adaptive)

                    # Resize crop result to match original crop size (in case padding was used)
                    if crop_seg_logits.shape[-2:] != (y2 - y1, x2 - x1):
                        import torch.nn.functional as F
                        crop_seg_logits = F.interpolate(
                            crop_seg_logits.unsqueeze(0),
                            size=(y2 - y1, x2 - x1),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)

                    # Initialize accumulators on first crop
                    if seg_logits_sum is None:
                        num_classes = crop_seg_logits.shape[0]
                        seg_logits_sum = torch.zeros((num_classes, h, w), device=self.device, dtype=torch.bfloat16)
                        count_map = torch.zeros((h, w), device=self.device, dtype=torch.float32)

                        if crop_semantic is not None:
                            semantic_logits_sum = torch.zeros((num_classes, h, w), device=self.device, dtype=torch.bfloat16)
                        if crop_instance is not None:
                            instance_logits_sum = torch.zeros((num_classes, h, w), device=self.device, dtype=torch.bfloat16)

                    # Accumulate logits
                    seg_logits_sum[:, y1:y2, x1:x2] += crop_seg_logits
                    count_map[y1:y2, x1:x2] += 1

                    if crop_semantic is not None:
                        semantic_logits_sum[:, y1:y2, x1:x2] += crop_semantic
                    if crop_instance is not None:
                        instance_logits_sum[:, y1:y2, x1:x2] += crop_instance

                    # Collect boxes from this crop and transform to global coordinates (only if needed)
                    if boxes_by_prompt is not None and crop_per_class:
                        for prompt_name, class_data in crop_per_class.items():
                            if prompt_name not in boxes_by_prompt:
                                boxes_by_prompt[prompt_name] = {"boxes": [], "scores": []}

                            crop_boxes = class_data["boxes"]  # [N, 4] in crop coordinates
                            crop_scores = class_data["scores"]  # [N]

                            # Transform boxes from crop coordinates to global coordinates
                            # boxes are in [x1, y1, x2, y2] format
                            if len(crop_boxes) > 0:
                                # Clone to avoid inplace update
                                crop_boxes = crop_boxes.clone()
                                crop_boxes[:, [0, 2]] += x1  # Add x offset
                                crop_boxes[:, [1, 3]] += y1  # Add y offset

                                boxes_by_prompt[prompt_name]["boxes"].append(crop_boxes)
                                boxes_by_prompt[prompt_name]["scores"].append(crop_scores)

                    # Merge detailed results (only store first crop to avoid excessive memory)
                    if detailed and crop_per_class and not per_class_results:
                        per_class_results = crop_per_class

                    # Free crop tensors to reduce memory
                    del crop_seg_logits, crop_per_class, crop_semantic, crop_instance, crop_adaptive, crop

                    if self.memory_debugger:
                        self.memory_debugger.log_cuda_memory(f"Crop {crop_idx+1} after deletion")

                    # More aggressive cache cleanup: clean after every crop instead of just at end of row
                    # This is crucial when processing many crops with many prompts
                    torch.cuda.empty_cache()

                    if self.memory_debugger:
                        self.memory_debugger.log_cuda_memory(f"Crop {crop_idx+1} after empty_cache")

                    crop_idx += 1
                    pbar.update(1)

        # Average accumulated logits
        seg_logits = seg_logits_sum / count_map.unsqueeze(0).to(torch.bfloat16)
        semantic_logits = semantic_logits_sum / count_map.unsqueeze(0).to(torch.bfloat16) if semantic_logits_sum is not None else None
        instance_logits = instance_logits_sum / count_map.unsqueeze(0).to(torch.bfloat16) if instance_logits_sum is not None else None

        # Concatenate boxes and scores from all crops (only if boxes were collected)
        if boxes_by_prompt is not None:
            for prompt_name in boxes_by_prompt:
                if boxes_by_prompt[prompt_name]["boxes"]:
                    boxes_by_prompt[prompt_name]["boxes"] = torch.cat(
                        boxes_by_prompt[prompt_name]["boxes"], dim=0
                    )
                    boxes_by_prompt[prompt_name]["scores"] = torch.cat(
                        boxes_by_prompt[prompt_name]["scores"], dim=0
                    )

            # Store boxes in per_class_results (always available when detailed=True)
            # Used for both visualization/debugging and iterative refinement
            per_class_results["_boxes_by_prompt"] = boxes_by_prompt

        # Return adaptive thresholds from first crop (or None if not available)
        adaptive_prob_thresholds = adaptive_prob_thresholds_list[0] if adaptive_prob_thresholds_list else None

        return seg_logits, per_class_results, semantic_logits, instance_logits, adaptive_prob_thresholds

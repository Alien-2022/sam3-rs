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

        # Collect boxes and scores from all crops (only if detailed=True)
        # Format: {prompt_name: {"boxes": [[N1,4], [N2,4], ...], "scores": [[N1], [N2], ...]}}
        boxes_accumulator = {} if detailed else None

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

                    # Run inference on crop (always collect detailed results when detailed=True)
                    crop_seg_logits, crop_per_class, crop_semantic, crop_instance, crop_adaptive = self.inference_func(
                        crop, detailed=detailed, image_name=f"{image_name}_crop_{crop_y}_{crop_x}"
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

                    # Collect boxes from this crop and transform to global coordinates
                    if crop_per_class and boxes_accumulator is not None:
                        for prompt_name, class_data in crop_per_class.items():
                            if prompt_name not in boxes_accumulator:
                                boxes_accumulator[prompt_name] = {"boxes": [], "scores": []}

                            crop_boxes = class_data["boxes"]  # [N, 4] in crop coordinates
                            crop_scores = class_data["scores"]  # [N]

                            # Transform boxes from crop coordinates to global coordinates
                            # boxes are in [x1, y1, x2, y2] format
                            if crop_boxes is not None and len(crop_boxes) > 0:
                                # Clone to avoid inplace update
                                crop_boxes = crop_boxes.clone()
                                crop_boxes[:, [0, 2]] += x1  # Add x offset
                                crop_boxes[:, [1, 3]] += y1  # Add y offset

                                boxes_accumulator[prompt_name]["boxes"].append(crop_boxes)
                                boxes_accumulator[prompt_name]["scores"].append(crop_scores)



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

        # Merge boxes and scores from all crops and apply NMS to remove duplicates
        per_class_results = {}
        if boxes_accumulator is not None:
            for prompt_name in boxes_accumulator:
                if boxes_accumulator[prompt_name]["boxes"]:
                    # Concatenate all crop boxes and scores
                    all_boxes = torch.cat(boxes_accumulator[prompt_name]["boxes"], dim=0)
                    all_scores = torch.cat(boxes_accumulator[prompt_name]["scores"], dim=0)

                    # Apply NMS to remove duplicate detections across crops
                    keep = self._apply_nms(all_boxes, all_scores, iou_threshold=0.5)

                    per_class_results[prompt_name] = {
                        "boxes": all_boxes[keep],
                        "scores": all_scores[keep],
                    }
                else:
                    # No boxes found for this class across all crops
                    per_class_results[prompt_name] = {
                        "boxes": None,
                        "scores": torch.tensor([], device=self.device, dtype=torch.bfloat16),
                    }

        # Return adaptive thresholds from first crop (or None if not available)
        adaptive_prob_thresholds = adaptive_prob_thresholds_list[0] if adaptive_prob_thresholds_list else None

        return seg_logits, per_class_results, semantic_logits, instance_logits, adaptive_prob_thresholds

    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
        """Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            boxes: [N, 4] tensor of boxes in [x1, y1, x2, y2] format
            scores: [N] tensor of confidence scores
            iou_threshold: IOU threshold for suppression

        Returns:
            keep: indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        # Sort boxes by score (highest first)
        sorted_indices = torch.argsort(scores, descending=True)
        keep = []

        while sorted_indices.shape[0] > 0:
            # Keep the highest scoring box
            current = sorted_indices[0]
            keep.append(current.item())

            if sorted_indices.shape[0] == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]

            # Compute IoU
            ious = self._calculate_iou(current_box.unsqueeze(0), remaining_boxes)

            # Keep boxes with IoU below threshold
            mask = ious.squeeze(0) < iou_threshold  # [1, M] -> [M]
            sorted_indices = sorted_indices[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes.

        Args:
            boxes1: [N, 4] tensor
            boxes2: [M, 4] tensor

        Returns:
            iou: [N, M] tensor of IoU values
        """
        # Expand for broadcasting: [N, 1, 4] and [1, M, 4]
        boxes1 = boxes1.unsqueeze(1)
        boxes2 = boxes2.unsqueeze(0)

        # Calculate intersection
        x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Calculate areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        union = area1 + area2 - intersection

        # Avoid division by zero
        iou = intersection / (union + 1e-6)

        return iou

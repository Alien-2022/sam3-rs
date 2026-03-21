"""
TVP-Fusion: Text-Visual Dual-Prompt Fusion for SAM3-RS

This module implements the TVP-Fusion method that achieved best performance
on Potsdam dataset (+2.53% IoU improvement over text-only baseline).

Workflow:
1. First pass: Text-only sliding window inference to get global boxes
2. Extract and filter top-k boxes for target classes
3. Second pass: For crops containing boxes, run text+box dual-prompt inference
4. Fuse text-only and text+box results using selected strategy
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Non-Maximum Suppression to boxes.
    
    Args:
        boxes: [N, 4] in [x1, y1, x2, y2] format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered boxes and scores
    """
    if len(boxes) == 0:
        return boxes, scores
    
    # Sort by score descending
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    
    # Calculate areas once
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        # Intersection
        xx1 = torch.max(boxes[i, 0], boxes[i+1:, 0])
        yy1 = torch.max(boxes[i, 1], boxes[i+1:, 1])
        xx2 = torch.min(boxes[i, 2], boxes[i+1:, 2])
        yy2 = torch.min(boxes[i, 3], boxes[i+1:, 3])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        # IoU
        iou = inter / (areas[i] + areas[i+1:] - inter + 1e-8)
        
        # Suppress boxes with IoU >= threshold
        suppress_mask = iou >= iou_threshold
        suppressed[i+1:] = suppressed[i+1:] | suppress_mask
    
    # Return kept boxes and scores
    keep = torch.tensor(keep, device=boxes.device)
    return boxes[keep], scores[keep]


class TVPFusionRefiner:
    """
    TVP-Fusion: Text-Visual Dual-Prompt Fusion for segmentation refinement.
    
    This implements the best-performing method from experiments:
    - Extract global boxes from text-only inference
    - For each crop with boxes, run text+box dual-prompt inference
    - Fuse text-only and text+box results
    """
    
    def __init__(
        self,
        processor,
        device: str = "cuda",
        top_k: int = 100,
        confidence_threshold: float = 0.5,
        min_box_area: int = 10000,
        box_expansion: float = 0.1,
        fusion_strategy: str = "max",
        nms_iou_threshold: float = 0.5,
    ):
        """
        Args:
            processor: Sam3Processor instance
            device: Device for computation
            top_k: Number of top instances to use as visual prompts per class
            confidence_threshold: Minimum confidence for instance boxes
            min_box_area: Minimum box area (in pixels) to consider
            box_expansion: Expansion factor for boxes (0.1 = 10% expansion on each side)
            fusion_strategy: Fusion strategy - "fixed", "confidence", or "max"
            nms_iou_threshold: IoU threshold for NMS
        """
        self.processor = processor
        self.device = torch.device(device)
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.min_box_area = min_box_area
        self.box_expansion = box_expansion
        self.fusion_strategy = fusion_strategy
        self.nms_iou_threshold = nms_iou_threshold
        
        print(f"✓ TVP-Fusion Refiner initialized:")
        print(f"  - top_k: {top_k}")
        print(f"  - confidence_threshold: {confidence_threshold}")
        print(f"  - min_box_area: {min_box_area}")
        print(f"  - box_expansion: {box_expansion}")
        print(f"  - fusion_strategy: {fusion_strategy}")
        print(f"  - nms_iou_threshold: {nms_iou_threshold}")

    def extract_and_filter_boxes(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and filter boxes from inference results.
        
        Args:
            boxes: [N, 4] in [x1, y1, x2, y2] format
            scores: [N] confidence scores
            image_shape: (height, width) of the image
            
        Returns:
            Filtered boxes and scores
        """
        if len(boxes) == 0:
            return boxes, scores
        
        # Filter by confidence
        valid_mask = scores >= self.confidence_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        
        if len(boxes) == 0:
            return boxes, scores
        
        # Filter by minimum area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_mask = areas >= self.min_box_area
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        
        if len(boxes) == 0:
            return boxes, scores
        
        # Apply NMS
        boxes, scores = nms_boxes(boxes, scores, self.nms_iou_threshold)
        
        if len(boxes) == 0:
            return boxes, scores
        
        # Select top-k
        sorted_indices = torch.argsort(scores, descending=True)
        top_indices = sorted_indices[:min(self.top_k, len(sorted_indices))]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        
        return boxes, scores

    def run_tvp_fusion(
        self,
        image: Image.Image,
        seg_logits_text: torch.Tensor,
        boxes_by_prompt: Dict[str, Dict],
        refine_prompt_indices: List[int],
        prompt_names: List[str],
        crop_size: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Run TVP-Fusion refinement on the image.
        
        Args:
            image: PIL image
            seg_logits_text: Text-only segmentation logits [num_prompts, H, W]
            boxes_by_prompt: Dict of {prompt_name: {"boxes": tensor, "scores": tensor}}
            refine_prompt_indices: Indices of prompts to refine
            prompt_names: List of prompt names
            crop_size: Crop size for sliding window
            stride: Stride for sliding window
            
        Returns:
            Refined segmentation logits [num_prompts, H, W]
        """
        w, h = image.size
        
        # Initialize output with text-only results
        seg_logits_refined = seg_logits_text.clone()
        
        # Prepare global boxes for each prompt
        global_boxes_by_prompt = {}
        for prompt_idx in refine_prompt_indices:
            prompt_name = prompt_names[prompt_idx]
            if prompt_name in boxes_by_prompt:
                boxes = boxes_by_prompt[prompt_name]["boxes"]
                scores = boxes_by_prompt[prompt_name]["scores"]
                
                if len(boxes) > 0:
                    boxes, scores = self.extract_and_filter_boxes(boxes, scores, (h, w))
                    if len(boxes) > 0:
                        global_boxes_by_prompt[prompt_idx] = (boxes, scores)
                        print(f"  [{prompt_name}] Selected {len(boxes)} boxes for refinement")
        
        if not global_boxes_by_prompt:
            print("  No valid boxes found for any prompt, skipping refinement")
            return seg_logits_refined
        
        # Calculate number of crops
        num_crops_x = max(w - crop_size + stride - 1, 0) // stride + 1
        num_crops_y = max(h - crop_size + stride - 1, 0) // stride + 1
        total_crops = num_crops_y * num_crops_x
        
        print(f"  Running TVP-Fusion on {total_crops} crops...")
        
        windows_with_boxes = 0
        current_crop = 0
        
        # Sliding window with text+visual dual prompts
        for crop_y in range(num_crops_y):
            for crop_x in range(num_crops_x):
                current_crop += 1
                print(f"  Progress: {current_crop}/{total_crops} (refined: {windows_with_boxes})", end='\r')
                
                x1 = crop_x * stride
                y1 = crop_y * stride
                x2 = min(x1 + crop_size, w)
                y2 = min(y1 + crop_size, h)
                x1 = max(x2 - crop_size, 0)
                y1 = max(y2 - crop_size, 0)
                
                crop = image.crop((x1, y1, x2, y2))
                crop_w, crop_h = crop.size
                
                for prompt_idx in refine_prompt_indices:
                    prompt_name = prompt_names[prompt_idx]
                    
                    if prompt_idx not in global_boxes_by_prompt:
                        continue
                    
                    boxes, _ = global_boxes_by_prompt[prompt_idx]
                    
                    # Check if any box center falls within this crop
                    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
                    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
                    overlap_mask = (center_x >= x1) & (center_x < x2) & \
                                   (center_y >= y1) & (center_y < y2)
                    
                    if not overlap_mask.any():
                        continue
                    
                    windows_with_boxes += 1
                    window_boxes = boxes[overlap_mask]
                    
                    # Convert to crop coordinates and expand
                    visual_prompts_boxes = []
                    for box in window_boxes:
                        bx1, by1, bx2, by2 = box.tolist()
                        box_w = bx2 - bx1
                        box_h = by2 - by1
                        expand_w = box_w * self.box_expansion
                        expand_h = box_h * self.box_expansion
                        
                        # Convert to crop coordinates with expansion
                        cx1 = max(0, bx1 - x1 - expand_w / 2)
                        cy1 = max(0, by1 - y1 - expand_h / 2)
                        cx2 = min(crop_w, bx2 - x1 + expand_w / 2)
                        cy2 = min(crop_h, by2 - y1 + expand_h / 2)
                        
                        # Normalize to [0, 1]
                        norm_cx = (cx1 + cx2) / 2 / crop_w
                        norm_cy = (cy1 + cy2) / 2 / crop_h
                        norm_w = (cx2 - cx1) / crop_w
                        norm_h = (cy2 - cy1) / crop_h
                        
                        visual_prompts_boxes.append([norm_cx, norm_cy, norm_w, norm_h])
                    
                    if not visual_prompts_boxes:
                        continue
                    
                    # Run text+box inference
                    with torch.no_grad(), torch.autocast(device_type=str(self.device), dtype=torch.bfloat16):
                        inference_state = self.processor.set_image(crop)
                        
                        # Use actual text prompt (not dummy "visual")
                        text_output = self.processor.set_text_prompt(
                            state=inference_state, prompt=prompt_name
                        )
                        
                        # Initialize geometric_prompt
                        if "geometric_prompt" not in inference_state:
                            inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
                        
                        # Add boxes
                        boxes_tensor = torch.tensor(visual_prompts_boxes, device=self.device, dtype=torch.float32)
                        boxes_tensor = boxes_tensor.unsqueeze(1)  # [N, 1, 4]
                        labels_tensor = torch.ones(len(visual_prompts_boxes), device=self.device, dtype=torch.bool)
                        labels_tensor = labels_tensor.unsqueeze(1)  # [N, 1]
                        
                        inference_state["geometric_prompt"].append_boxes(boxes_tensor, labels_tensor)
                        output = self.processor._forward_grounding(inference_state)
                        tvp_logit = output["masks_logits"].max(dim=0)[0].squeeze(0)
                        
                        # Resize if needed
                        if tvp_logit.shape != (crop_h, crop_w):
                            tvp_logit = F.interpolate(
                                tvp_logit.unsqueeze(0).unsqueeze(0),
                                size=(crop_h, crop_w), mode="bilinear", align_corners=False
                            ).squeeze()
                        
                        # Get text-only logit for this crop
                        text_logit = seg_logits_text[prompt_idx, y1:y2, x1:x2]
                        
                        # Fuse based on strategy
                        if self.fusion_strategy == "fixed":
                            fused_logit = 0.5 * text_logit + 0.5 * tvp_logit
                        elif self.fusion_strategy == "confidence":
                            prob_text = torch.sigmoid(text_logit)
                            prob_tvp = torch.sigmoid(tvp_logit)
                            weight_text = prob_text / (prob_text + prob_tvp + 1e-8)
                            weight_tvp = prob_tvp / (prob_text + prob_tvp + 1e-8)
                            fused_logit = weight_text * text_logit + weight_tvp * tvp_logit
                        elif self.fusion_strategy == "max":
                            fused_logit = torch.max(text_logit, tvp_logit)
                        
                        seg_logits_refined[prompt_idx, y1:y2, x1:x2] = fused_logit
                        
                        # Clean up
                        del inference_state, output, tvp_logit, boxes_tensor, labels_tensor
                        torch.cuda.empty_cache()
                
                del crop
        
        print()
        print(f"  TVP-Fusion complete! Refined {windows_with_boxes} crop-prompt pairs")
        
        return seg_logits_refined

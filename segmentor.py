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
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for SAM3-RS inference"""
    # Model paths
    checkpoint_path: str
    bpe_path: str

    # Inference parameters
    device: str = "cuda"
    confidence_threshold: float = 0.5
    prob_threshold: float = 0.0
    bg_idx: int = 0

    # Head selection (SegEarthOV3 innovation)
    use_semantic_head: bool = True
    use_instance_head: bool = True
    use_presence_score: bool = True

    # Large image handling
    slide_crop_size: int = 0  # 0 means no sliding
    slide_stride: int = 512

    # Text prompts
    prompts_file: Optional[str] = None  # Path to prompts config


@dataclass
class SegmentationResult:
    """Result structure for each image"""
    image_path: str
    seg_logits: torch.Tensor  # [num_classes, H, W]
    seg_pred: torch.Tensor   # [H, W] with class IDs
    per_class_results: Dict[str, Dict]  # {class_name: {masks, boxes, scores}}

    # Optional detailed outputs for analysis
    instance_logits: Optional[torch.Tensor] = None
    semantic_logits: Optional[torch.Tensor] = None
    presence_scores: Optional[torch.Tensor] = None


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

        # Initialize SAM3 model (follow SegEarthOV3's approach)
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            bpe_path=config.bpe_path,
            checkpoint_path=config.checkpoint_path,
            device=config.device
        )
        self.processor = Sam3Processor(
            model,
            confidence_threshold=config.confidence_threshold,
            device=self.device
        )

        # Load prompts if provided
        self.prompts = self._load_prompts(config.prompts_file)
        self.num_classes = max(self.prompts['indices']) + 1 if self.prompts else 0
        self.num_prompts = len(self.prompts['names']) if self.prompts else 0

        # Convert class indices to tensor
        if self.prompts:
            self.query_indices = torch.tensor(
                self.prompts['indices'], dtype=torch.int64, device=self.device
            )

        print(f"✓ SAM3-RS initialized with {self.num_classes} classes, {self.num_prompts} prompts")

    def _load_prompts(self, prompts_file: Optional[str]) -> Optional[Dict]:
        """Load class names and their indices from config file.

        File format (same as SegEarthOV3):
            background
            bareland,barren
            grass
            road
            car,vehicle
            tree,forest
            water,river
            cropland,farmland
            building,roof,house

        Returns:
            Dict with 'names', 'indices', 'mapping' (synonym -> class_id)
        """
        if not prompts_file or not os.path.exists(prompts_file):
            return None

        names, indices, mapping = [], [], {}

        with open(prompts_file, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Split by comma for synonyms
                synonyms = [s.strip() for s in line.split(',') if s.strip()]
                class_id = line_idx

                names.extend(synonyms)
                indices.extend([class_id] * len(synonyms))

                # Create mapping: each synonym maps to class_id
                for synonym in synonyms:
                    mapping[synonym] = class_id

        return {
            'names': names,
            'indices': indices,
            'mapping': mapping
        }

    def _inference_single_view(self, image: Image.Image) -> Tuple[torch.Tensor, Dict]:
        """
        Inference on a single image (or crop patch).

        Returns:
            seg_logits: [num_prompts, H, W] fused segmentation logits
            per_class_results: {prompt_name: {masks, boxes, scores, ...}}
        """
        w, h = image.size
        seg_logits = torch.zeros((self.num_prompts, h, w), device=self.device)
        per_class_results = {}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

            # Process each prompt
            for prompt_idx, prompt_word in enumerate(self.prompts['names']):
                # Reset prompts for clean inference
                self.processor.reset_all_prompts(inference_state)
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt_word
                )

                # Store per-class detailed results
                per_class_results[prompt_word] = {
                    'masks': output['masks'],
                    'masks_logits': output['masks_logits'],
                    'boxes': output['boxes'],
                    'scores': output['scores'],
                    'semantic_logits': output['semantic_seg'],
                    'presence_score': output.get('presence_score', 1.0)
                }

                # ===== SegEarthOV3's Dual-Head Fusion =====
                current_logits = torch.zeros((h, w), device=self.device)

                # 1. Instance head (Transformer decoder)
                if self.config.use_instance_head:
                    num_instances = output['masks_logits'].shape[0]
                    if num_instances > 0:
                        for inst_id in range(num_instances):
                            inst_logits = output['masks_logits'][inst_id].squeeze()
                            inst_score = output['object_score'][inst_id]

                            # Resize if needed
                            if inst_logits.shape != (h, w):
                                inst_logits = F.interpolate(
                                    inst_logits.view(1, 1, *inst_logits.shape),
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze()

                            # Accumulate with score weighting
                            current_logits = torch.max(current_logits, inst_logits * inst_score)

                # 2. Semantic head
                if self.config.use_semantic_head:
                    semantic_logits = output['semantic_seg'].squeeze()
                    if semantic_logits.shape != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits.unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    current_logits = torch.max(current_logits, semantic_logits)

                # 3. Presence score filtering
                if self.config.use_presence_score:
                    presence_score = output.get('presence_score', 1.0)
                    current_logits = current_logits * presence_score

                seg_logits[prompt_idx] = current_logits

        return seg_logits, per_class_results

    def _sliding_window_inference(self, image: Image.Image) -> Tuple[torch.Tensor, Dict]:
        """
        Sliding window inference for large images.

        Args:
            image: PIL Image (can be very large, e.g., 10000x10000)

        Returns:
            seg_logits: Fused segmentation for the whole image
            per_class_results: Detailed results (aggregated from crops)
        """
        w_img, h_img = image.size
        crop_size = self.config.slide_crop_size
        stride = self.config.slide_stride

        # Initialize accumulators
        seg_logits = torch.zeros((self.num_prompts, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        # Calculate number of patches
        h_grids = max((h_img - crop_size + stride - 1) // stride + 1, 1)
        w_grids = max((w_img - crop_size + stride - 1) // stride + 1, 1)

        print(f"  Sliding window: {h_grids}x{w_grids} = {h_grids*w_grids} patches")

        per_class_results = {}

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # Calculate crop coordinates
                y1 = h_idx * stride
                x1 = w_idx * stride
                y2 = min(y1 + crop_size, h_img)
                x2 = min(x1 + crop_size, w_img)

                # Adjust for boundary (ensure full crop_size at edges)
                y1 = max(y2 - crop_size, 0)
                x1 = max(x2 - crop_size, 0)

                # Crop image
                crop_img = image.crop((x1, y1, x2, y2))

                # Inference on crop
                crop_logits, crop_results = self._inference_single_view(crop_img)

                # Accumulate results
                seg_logits[:, y1:y2, x1:x2] += crop_logits
                count_mat[:, y1:y2, x1:x2] += 1

                # Store detailed results for first patch only (or you can aggregate)
                if h_idx == 0 and w_idx == 0:
                    per_class_results = crop_results

        # Average overlapping regions
        seg_logits = seg_logits / count_mat

        return seg_logits, per_class_results

    def predict_single(
        self,
        image_path: str,
        detailed: bool = False
    ) -> SegmentationResult:
        """
        Predict segmentation for a single image.

        Args:
            image_path: Path to input image
            detailed: Whether to return per-instance results

        Returns:
            SegmentationResult with predictions
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_shape = (image.height, image.width)

        # Choose inference mode
        if (self.config.slide_crop_size > 0 and
            (self.config.slide_crop_size < image.width or
             self.config.slide_crop_size < image.height)):
            # Use sliding window for large images
            seg_logits, per_class_results = self._sliding_window_inference(image)
        else:
            # Single view inference
            seg_logits, per_class_results = self._inference_single_view(image)

        # Resize to original shape if needed
        if seg_logits.shape[-2:] != original_shape:
            seg_logits = F.interpolate(
                seg_logits.unsqueeze(0),
                size=original_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # ===== Post-processing =====

        # 1. Map prompts to actual class IDs (handle synonyms)
        if self.num_classes != self.num_prompts:
            seg_logits = seg_logits.unsqueeze(0)
            cls_index = F.one_hot(self.query_indices, num_classes=self.num_classes)
            cls_index = cls_index.T.view(self.num_classes, self.num_prompts, 1, 1)
            seg_logits = (seg_logits * cls_index).max(1)[0]

        # 2. Get final prediction (argmax)
        seg_pred = torch.argmax(seg_logits, dim=0)

        # 3. Apply probability threshold (filter low-confidence pixels to background)
        if self.config.prob_threshold > 0:
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.config.prob_threshold] = self.config.bg_idx

        # Prepare result
        result = SegmentationResult(
            image_path=image_path,
            seg_logits=seg_logits,
            seg_pred=seg_pred,
            per_class_results=per_class_results
        )

        if detailed:
            result.instance_logits = per_class_results.get('instance_logits')
            result.semantic_logits = per_class_results.get('semantic_logits')
            result.presence_scores = torch.tensor([
                r.get('presence_score', 1.0) for r in per_class_results.values()
            ])

        return result

    def predict_batch(
        self,
        image_paths: List[str],
        save_dir: Optional[str] = None,
        detailed: bool = False
    ) -> List[SegmentationResult]:
        """
        Predict segmentation for a batch of images.

        Args:
            image_paths: List of image paths
            save_dir: Optional directory to save results
            detailed: Whether to return detailed per-instance results

        Returns:
            List of SegmentationResult
        """
        results = []

        print(f"\n{'='*60}")
        print(f"Processing {len(image_paths)} images with SAM3-RS")
        print(f"{'='*60}\n")

        for idx, img_path in enumerate(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] {os.path.basename(img_path)}")

            result = self.predict_single(img_path, detailed=detailed)
            results.append(result)

            # Save if directory provided
            if save_dir:
                self._save_result(result, save_dir)

        print(f"\n✓ Completed {len(results)} images")
        return results

    def _save_result(self, result: SegmentationResult, save_dir: str):
        """Save segmentation result to disk."""
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(result.image_path))[0]

        # Save semantic segmentation (main output)
        pred_path = os.path.join(save_dir, f"{base_name}_pred.png")
        pred_img = Image.fromarray(result.seg_pred.cpu().numpy().astype(np.uint8))
        pred_img.save(pred_path)

        # Save logits (for visualization or post-processing)
        logits_path = os.path.join(save_dir, f"{base_name}_logits.npy")
        np.save(logits_path, result.seg_logits.cpu().numpy())

        # Optional: save per-class binary masks
        class_masks_dir = os.path.join(save_dir, f"{base_name}_masks")
        os.makedirs(class_masks_dir, exist_ok=True)

        for class_name, class_data in result.per_class_results.items():
            if class_data['masks'].shape[0] > 0:
                # Combine all instance masks for this class
                class_mask = class_data['masks'].cpu().numpy()
                combined_mask = class_mask.max(axis=0)  # OR operation
                binary_mask = (combined_mask > 0.5).astype(np.uint8) * 255

                mask_img = Image.fromarray(binary_mask, mode='L')
                # Use mapped class ID instead of raw name for filename
                class_id = self.prompts['mapping'].get(class_name, 0)
                mask_path = os.path.join(class_masks_dir, f"class_{class_id}.png")
                mask_img.save(mask_path)

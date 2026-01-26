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
    bg_idx: Optional[int] = 0
    # Set to True only when prompts.txt 已包含背景类；默认 False 表示需要注入背景通道
    prompt_includes_bg: bool = False

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
            model, confidence_threshold=config.confidence_threshold, device=self.device
        )

        # Load prompts if provided
        self.prompts = self._load_prompts(config.prompts_file)
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

        with open(prompts_file, "r") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Split by comma for synonyms
                synonyms = [s.strip() for s in line.split(",") if s.strip()]
                class_id = line_idx

                names.extend(synonyms)
                indices.extend([class_id] * len(synonyms))

                # Create mapping: each synonym maps to class_id
                for synonym in synonyms:
                    mapping[synonym] = class_id

        return {"names": names, "indices": indices, "mapping": mapping}

    def _inference_single_view(
        self, image: Image.Image, detailed: bool = False
    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Inference on a single image (or crop patch).

        Returns:
            seg_logits: [num_prompts, H, W] fused segmentation logits
            per_class_results: {prompt_name: {masks, boxes, scores, ...}} (empty if not detailed)
            semantic_logits_only: [num_prompts, H, W] from semantic head only (None if not enabled)
            instance_logits_only: [num_prompts, H, W] from instance head only (None if not enabled)
        """
        w, h = image.size
        seg_logits = torch.zeros((self.num_prompts, h, w), device=self.device)
        per_class_results = {}

        # Initialize separate logits for individual heads
        if self.config.use_semantic_head:
            semantic_logits_only = torch.zeros(
                (self.num_prompts, h, w), device=self.device
            )
        else:
            semantic_logits_only = None

        if self.config.use_instance_head:
            instance_logits_only = torch.zeros(
                (self.num_prompts, h, w), device=self.device
            )
        else:
            instance_logits_only = None

        inference_state = None
        # Use bfloat16 autocast for RTX 50 series (significant speedup over float32)
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            inference_state = self.processor.set_image(image)

            # Process each prompt
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # Reset prompts for clean inference
                self.processor.reset_all_prompts(inference_state)
                output = self.processor.set_text_prompt(
                    state=inference_state, prompt=prompt_word
                )

                # Store per-class detailed results if requested (move to CPU to save GPU memory)
                if detailed:
                    per_class_results[prompt_word] = {
                        "masks": output["masks"].cpu(),
                        "masks_logits": output["masks_logits"].cpu(),
                        "boxes": output["boxes"].cpu(),
                        "scores": output["scores"].cpu(),
                        "semantic_logits": output["semantic_seg"].cpu(),
                        "presence_score": output.get("presence_score", 1.0),
                    }

                # ===== SegEarthOV3's Dual-Head Fusion =====
                current_logits = torch.zeros((h, w), device=self.device)

                # 1. Instance head (Transformer decoder)
                if self.config.use_instance_head:
                    inst_current = torch.zeros((h, w), device=self.device)
                    num_instances = output["masks_logits"].shape[0]
                    if num_instances > 0:
                        for inst_id in range(num_instances):
                            # masks_logits: [inst_num, 1, H_orig, W_orig]
                            inst_logits = output["masks_logits"][
                                inst_id
                            ]  # [1, H_orig, W_orig]
                            # SAM3 uses 'scores' which already includes presence_score
                            inst_score = output["scores"][inst_id]

                            # inst_logits.shape: [1, 1024, 1024] -> interpolate to [H, W]
                            # Add channel dim for interpolate: [1, 1, H_orig, W_orig]
                            if inst_logits.dim() == 3:
                                inst_logits = inst_logits.unsqueeze(
                                    0
                                )  # [1, 1, H_orig, W_orig]
                            inst_logits = F.interpolate(
                                inst_logits,
                                size=(h, w),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()  # [H, W]

                            # Accumulate with max pooling over instances (without score weighting)
                            # NOTE: Score weighting is applied via presence_score at the end
                            # inst_current = torch.max(inst_current, inst_logits)
                            inst_current = torch.max(
                                inst_current, inst_logits
                            )

                    current_logits = torch.max(current_logits, inst_current)
                    # print("current_logits shape after instance head: ", current_logits.shape)
                    # print("current_logits min after instance head: ", current_logits.min())
                    # print("current_logits max after instance head: ", current_logits.max())
                    instance_logits_only[prompt_idx] = inst_current

                # 2. Semantic head
                if self.config.use_semantic_head:
                    # semantic_logits: [1, 1, H_orig, W_orig] (4D tensor)
                    semantic_logits = output["semantic_seg"]
                    semantic_logits = F.interpolate(
                        semantic_logits,  # Already 4D [1, 1, H_orig, W_orig]
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()  # [H, W]

                    # Fusion: take max of instance and semantic predictions
                    current_logits = torch.max(current_logits, semantic_logits)
                    semantic_logits_only[prompt_idx] = semantic_logits

                # print(f"current prompt: {prompt_word}")
                # print(
                #     f"Instance logits range: [{inst_current.min():.3f}, {inst_current.max():.3f}]"
                # )
                # print(
                #     f"Semantic logits range: [{semantic_logits.min():.3f}, {semantic_logits.max():.3f}]"
                # )
                # print(f"Num instances: {num_instances}")

                # 3. Presence score filtering
                # NOTE: Following SegEarthOV3's approach: apply to fused result (both heads)
                # Multiply presence_score after max fusion
                if self.config.use_presence_score:
                    presence_score = output.get("presence_score", 1.0)
                    current_logits = current_logits * presence_score

                seg_logits[prompt_idx] = current_logits

        # Clean up inference_state to free GPU memory
        # if inference_state is not None:
        #     for key in list(inference_state.keys()):
        #         if isinstance(inference_state[key], torch.Tensor):
        #             del inference_state[key]

        return seg_logits, per_class_results, semantic_logits_only, instance_logits_only

    def _sliding_window_inference(
        self, image: Image.Image, detailed: bool = False
    ) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sliding window inference for large images.

        Returns:
            seg_logits: Fused segmentation for the whole image
            per_class_results: Detailed results (aggregated from first crop when detailed=True)
            semantic_logits_only: None (skip to save memory for large images)
            instance_logits_only: None (skip to save memory for large images)
        """
        w_img, h_img = image.size
        crop_size = self.config.slide_crop_size
        stride = self.config.slide_stride

        # Initialize accumulators
        seg_logits = torch.zeros((self.num_prompts, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        # Skip storing semantic/instance logits in sliding mode to reduce memory footprint
        semantic_logits_only = None
        instance_logits_only = None

        # Calculate number of patches
        h_grids = max((h_img - crop_size + stride - 1) // stride + 1, 1)
        w_grids = max((w_img - crop_size + stride - 1) // stride + 1, 1)

        print(f"  Sliding window: {h_grids}x{w_grids} = {h_grids*w_grids} patches")

        per_class_results = {} if detailed else None

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
                crop_logits, crop_results, _, _ = (
                    self._inference_single_view(crop_img, detailed=detailed)
                )

                # Accumulate results
                seg_logits[:, y1:y2, x1:x2] += crop_logits
                count_mat[:, y1:y2, x1:x2] += 1

                # Store detailed results for first patch only (or you can aggregate)
                if detailed and h_idx == 0 and w_idx == 0:
                    per_class_results = crop_results

        # Average overlapping regions
        seg_logits = seg_logits / count_mat

        return seg_logits, per_class_results, semantic_logits_only, instance_logits_only

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

        # Choose inference mode
        if self.config.slide_crop_size > 0 and (
            self.config.slide_crop_size < image.width
            or self.config.slide_crop_size < image.height
        ):
            # Use sliding window for large images
            (
                seg_logits,
                per_class_results,
                semantic_logits_only,
                instance_logits_only,
            ) = self._sliding_window_inference(image, detailed=detailed)
        else:
            # Single view inference
            (
                seg_logits,
                per_class_results,
                semantic_logits_only,
                instance_logits_only,
            ) = self._inference_single_view(image, detailed=detailed)

        # Resize fused head logits if needed
        if seg_logits.shape[-2:] != original_shape:
            seg_logits = F.interpolate(
                seg_logits.unsqueeze(0),
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Resize individual head logits if needed
        if (
            semantic_logits_only is not None
            and semantic_logits_only.shape[-2:] != original_shape
        ):
            semantic_logits_only = F.interpolate(
                semantic_logits_only.unsqueeze(0),
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if (
            instance_logits_only is not None
            and instance_logits_only.shape[-2:] != original_shape
        ):
            instance_logits_only = F.interpolate(
                instance_logits_only.unsqueeze(0),
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # ===== Post-processing =====

        # print("num_classes: ", self.num_classes)
        # print("num_prompts: ",self.num_prompts)

        # 1. Map prompts to actual class IDs (handle synonyms)
        if self.num_classes != self.num_prompts:
            # seg_logits.shape: [num_queries, H, W] -> [1, num_queries, H, W]
            seg_logits = seg_logits.unsqueeze(0)

            # 把query_idx转换为one-hot向量，比如原来的query_idx=[0, 1, 1, 2]，得到的one_hot向量为：
            # [[1, 0, 0],
            #  [0, 1, 0],
            #  [0, 1, 0],
            #  [0, 0, 1]]
            cls_index = F.one_hot(self.query_indices, num_classes=self.num_classes)

            """
            cls_index.shape: [num_queries, num_cls]
            cls_index.T 计算转置，让每个类占一行 [num_queries, num_cls]->[num_cls, num_queries],转置后的cls_index为：
            [[1, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 1]]
            view 把cls_index转换为[num_cls, num_queries, 1, 1], 扩展维度以便与seg_logits进行广播运算
            现在的cls_index=[
                [ [[1]], [[0]], [[0]], [[0]] ],
                [ [[0]], [[1]], [[1]], [[0]] ],
                [ [[0]], [[0]], [[0]], [[1]] ]
            ]

            广播运算的特点是让不同的数组（或张量）在进行算术运算时，自动“扩展”成兼容的形状，而无需复制数据。
            """
            cls_index = cls_index.T.view(self.num_classes, self.num_prompts, 1, 1)
            # print("cls_index shape: ", cls_index.shape)

            """
            相乘之前: seg_logits.shape: [1, num_queries, h, w], cls_index.shape: [num_cls, num_queries, 1, 1]
            相乘时：
                触发广播机制将seg_logits扩展到num_cls个类别: seg_logits.shape: [num_cls, num_queries, h, w]
                假设图像h=1, w=2 共有2个像素:
                    seg_logits 存了这个像素在4个查询词上的置信度
                    seg_logits_broadcast = [
                        [ [[0.6, 0.9]], [[0.4, 0.7]], [[0.3, 0.1]], [[0.2, 0.8]] ],  # 给类别0用
                        [ [[0.6, 0.9]], [[0.4, 0.7]], [[0.3, 0.9]], [[0.2, 0.8]] ],  # 给类别1用(复制)
                        [ [[0.6, 0.9]], [[0.4, 0.7]], [[0.3, 0.1]], [[0.2, 0.8]] ],  # 给类别2用(复制)
                    ]
                    上面[[0.6, 0.9]]代表一个查询词对应的图像概率图，包含了每个像素位置存在该实例的概率
                    所以上面共有四个查询词的概率图: [[0.6, 0.9]], [[0.4, 0.7]], [[0.3, 0.1]], [[0.2, 0.8]]
                    第2,3行是为了广播复制的第一行, 内容一样
            接下来逐元素相乘 = [
                    [ [[0.6x1, 0.9x1]], [[0.4x0, 0.7x0]], [[0.3x0, 0.1x0]], [[0.2x0, 0.8x0]] ],
                    [ [[0.6x0, 0.9x0]], [[0.4x1, 0.7x1]], [[0.3x1, 0.9x1]], [[0.2x0, 0.8x0]] ], 
                    [ [[0.6x0, 0.9x0]], [[0.4x0, 0.7x0]], [[0.3x0, 0.1x0]], [[0.2x1, 0.8x1]] ], 
                ]
                逐元素相乘时才可以看到刚才的广播操作的作用, 我们把cls_index中的每一行与seg_logits_broadcast的每一行相乘
                因为seg_logits被广播扩展到了num_cls个类别, 与cls_index一致, 所以可以执行相乘
            相乘结果 = [
                    [ [[0.6, 0.9]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]] ],
                    [ [[0.0, 0.0]], [[0.4, 0.7]], [[0.3, 0.9]], [[0.0, 0.0]] ], 
                    [ [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.2, 0.8]] ], 
                ]
                得到的结果中, 每个类比(每一行), 只保留了属于该类别的查询词的概率图
                相乘之后: seg_logits.shape: [num_cls,num_queries, h, w]
            取max(1): 
                遍历每个类别:
                    [ [[0.6, 0.9]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]] ]
                    ...
                遍历每个像素:
                    [0.6, 0.0, 0.0, 0.0]
                    ...
                在num_queries(列)维度上取最大值
                    max([0.6, 0, 0, 0, 0]) = 0.6
                    ...
                遍历结束得到最终结果 (values, indices)
                    values, indices的shape都是[num_cls, h, w]
                    values=[
                        [ [0.6, 0.9] ],
                        [ [0.4, 0.9] ], 
                        [ [0.2, 0.8] ]
                    ]
                    indices=[
                        [ [0, 0] ],
                        [ [1, 2] ], 
                        [ [3, 3] ]
                    ]
            取max(1)[0]: 相当于取(values, indices)中的values
            最终结果seg_logits=values
            """
            seg_logits = (seg_logits * cls_index).max(1)[0]

        # 2. Get final prediction (argmax)
        # 通过上面的处理同一个类别中的多个查询词已经融合到一起了
        # 对于每个像素从num_cls个类别中取出最大概率值为该像素类别
        # argmax(dim=0) 的意思是固定第1,2维度，在第0维度(类别维度)中找最大值索引
        # 以上面得到seg_logits为例, 遍历每个像素取每个中的3个类别值:
        # 第一个像素 [0.6, 0.4, 0.2] -> 最大值索引0
        # 第二个像素 [0.9, 0.9, 0.8] -> 最大值索引为0(有两个相同最大值时，argmax 取第一个出现的索引)
        # 最终返回 [[0,0]], 对应shape:[h,w]

        # print("before argmax seg_logits shape: ", seg_logits.shape)
        # print("before argmax seg_logits min: ", seg_logits.min())
        # print("before argmax seg_logits max: ", seg_logits.max())

        # Unified prediction logic (single/multi-class)
        bg_idx = 0 if self.config.bg_idx is None else self.config.bg_idx

        def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
            # If prompts do NOT include background, explicitly add a zero-logit background channel
            if not self.config.prompt_includes_bg:
                bg_pad = torch.zeros((1, *logits.shape[1:]), device=logits.device, dtype=logits.dtype)
                logits_for_argmax = torch.cat([bg_pad, logits], dim=0)
                pred = torch.argmax(logits_for_argmax, dim=0)

                # Map background to bg_idx; keep foreground IDs contiguous starting from 1
                if bg_idx != 0:
                    pred = torch.where(pred == 0, torch.tensor(bg_idx, device=pred.device), pred + bg_idx - 1)
            else:
                pred = torch.argmax(logits, dim=0)

            if self.config.prob_threshold > 0:
                max_vals = logits.max(0)[0]
                pred[max_vals < self.config.prob_threshold] = bg_idx
            elif not self.config.prompt_includes_bg:
                # Fallback: if no explicit background prompt and no prob_threshold, treat non-positive logits as background
                max_vals = logits.max(0)[0]
                pred[max_vals <= 0] = bg_idx
            return pred

        seg_pred = logits_to_pred(seg_logits)

        # print("after argmax seg_pred shape: ", seg_pred.shape)
        # print("after argmax seg_pred min: ", seg_pred.min())
        # print("after argmax seg_pred max: ", seg_pred.max())

        # Prepare individual head logits (no argmax/threshold here)
        semantic_logits = None
        instance_logits = None

        if semantic_logits_only is not None:
            if self.num_classes != self.num_prompts:
                semantic_logits_only = semantic_logits_only.unsqueeze(0)
                cls_index = F.one_hot(self.query_indices, num_classes=self.num_classes)
                cls_index = cls_index.T.view(self.num_classes, self.num_prompts, 1, 1)
                semantic_logits_only = (semantic_logits_only * cls_index).max(1)[0]
            semantic_logits = semantic_logits_only

        if instance_logits_only is not None:
            if self.num_classes != self.num_prompts:
                instance_logits_only = instance_logits_only.unsqueeze(0)
                cls_index = F.one_hot(self.query_indices, num_classes=self.num_classes)
                cls_index = cls_index.T.view(self.num_classes, self.num_prompts, 1, 1)
                instance_logits_only = (instance_logits_only * cls_index).max(1)[0]
            instance_logits = instance_logits_only

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
        save_dir: Optional[str] = None,
        detailed: bool = False,
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
            if class_data["masks"].shape[0] > 0:
                # Combine all instance masks for this class
                class_mask = class_data["masks"].cpu().numpy()
                combined_mask = class_mask.max(axis=0)  # OR operation
                binary_mask = (combined_mask > 0.5).astype(np.uint8) * 255

                mask_img = Image.fromarray(binary_mask, mode="L")
                # Use mapped class ID instead of raw name for filename
                class_id = self.prompts["mapping"].get(class_name, 0)
                mask_path = os.path.join(class_masks_dir, f"class_{class_id}.png")
                mask_img.save(mask_path)

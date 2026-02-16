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

    # Large image handling
    slide_crop_size: int = 0  # 0 means no sliding
    slide_stride: int = 512

    # Text prompts
    prompts_file: Optional[str] = None  # Path to prompts config

    # Debug options
    debug_memory: bool = False  # Print memory usage information
    debug_log_file: Optional[str] = None  # Path to log file for debug output


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

        # Enable memory debugging and setup logging file FIRST
        self.debug_memory = config.debug_memory if hasattr(config, 'debug_memory') else False
        self.debug_log_file = config.debug_log_file
        self.debug_log_handle = None
        if self.debug_log_file:
            import sys
            self.debug_log_handle = open(self.debug_log_file, 'w', encoding='utf-8')

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

        # Pre-compute text features for all prompts to avoid repeated computation
        self.text_features_cache = None
        self._precompute_text_features()

    def __del__(self):
        """Close log file on cleanup."""
        if self.debug_log_handle:
            self.debug_log_handle.close()

    def _debug_print(self, message: str):
        """Print debug message to log file only (not to console)."""
        # Only write to log file to avoid console spam
        if self.debug_log_handle:
            self.debug_log_handle.write(message + '\n')
            self.debug_log_handle.flush()

    def _log_tensor_memory(self, name: str, tensor: torch.Tensor, detail: bool = False):
        """Log tensor memory usage."""
        if not self.debug_memory or tensor is None:
            return

        numel = tensor.numel()
        element_size = tensor.element_size()
        memory_mb = numel * element_size / (1024 * 1024)

        msg = f"[MEMORY] {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, " \
              f"memory={memory_mb:.2f} MB ({numel:,} elements)"
        self._debug_print(msg)

        if detail and tensor.dim() > 2:
            # Log statistics for 3D+ tensors
            detail_msg = f"        min={tensor.min():.4f}, max={tensor.max():.4f}, " \
                         f"mean={tensor.mean():.4f}"
            self._debug_print(detail_msg)

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

    def _precompute_text_features(self):
        """Pre-compute text features for all prompts to avoid repeated computation during inference."""
        if not self.prompts or self.num_prompts == 0:
            return

        print(f"Pre-computing text features for {self.num_prompts} prompts...")

        # Initialize cache dictionary
        self.text_features_cache = {}

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # Compute text features for this prompt
                text_outputs = self.processor.model.backbone.forward_text(
                    [prompt_word], device=self.device
                )

                # Store the features in cache
                self.text_features_cache[prompt_idx] = {
                    "language_features": text_outputs.get("language_features"),
                    "language_mask": text_outputs.get("language_mask"),
                    "language_embeds": text_outputs.get("language_embeds"),
                }

                # Progress indicator
                if (prompt_idx + 1) % 10 == 0 or prompt_idx == self.num_prompts - 1:
                    print(f"  Progress: {prompt_idx + 1}/{self.num_prompts}")

        print(f"✓ Text features pre-computed successfully!")

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
        # Use float32 autocast to avoid bf16/float32 weight mismatch on some backbones
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            inference_state = self.processor.set_image(image)

            if self.debug_memory:
                self._debug_print(f"\n[MEMORY] ===== Image: {image_name}, Size: {w}x{h} =====")
                self._debug_print(f"[MEMORY] Image feature extraction completed: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB")

            # Process each prompt
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                if self.debug_memory:
                    self._debug_print(f"[MEMORY] ===== Prompt [{prompt_idx+1}/{self.num_prompts}]: {prompt_word} =====")

                # Reset prompts for clean inference
                self.processor.reset_all_prompts(inference_state)
                output = self.processor.set_text_prompt(
                    state=inference_state, prompt=prompt_word
                )

                # Only log tensor info for the first prompt (avoid redundancy)
                if self.debug_memory and prompt_idx == 0:
                    self._log_tensor_memory("masks_logits", output["masks_logits"])
                    self._log_tensor_memory("semantic_seg", output["semantic_seg"])

                if self.debug_memory:
                    self._debug_print(f"[MEMORY] After set_text_prompt: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB")

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

                            # ========================================================================
                            # 方式1（当前使用）：纯 max 融合，不乘 instance_score
                            # ========================================================================
                            # inst_current = torch.max(inst_current, inst_logits)

                            # ========================================================================
                            # 方式2（SegEarthOV3 方式）：乘以 instance_score 后再取 max
                            # 优点：高置信度的实例对最终结果影响更大
                            # 缺点：可能会过度依赖分数，低置信度的有效实例被忽略
                            # ========================================================================
                            # inst_current = torch.max(inst_current, inst_logits * inst_score)

                            # ========================================================================
                            # 方式3（加权平均融合）：所有实例按分数加权求和
                            # 优点：考虑了所有实例的贡献，更平滑
                            # 缺点：计算量更大，可能模糊边界
                            # ========================================================================
                            # if inst_score > self.config.confidence_threshold:
                            #     inst_current = inst_current + inst_logits * inst_score
                            # count += 1
                            # if count > 0:
                            #     inst_current = inst_current / count

                            # ========================================================================
                            # 方式4（阈值过滤 + max）：只保留置信度超过阈值的实例再取 max
                            # 优点：过滤低质量实例，保持边界清晰
                            # 缺点：需要调整阈值参数
                            # ========================================================================
                            # if inst_score > self.config.confidence_threshold:
                            #     inst_current = torch.max(inst_current, inst_logits)

                            inst_current = torch.max(inst_current, inst_logits * inst_score)

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
                # Multiply presence_score after max fusio
                # presence_score = output.get("presence_score", -1)
                # print(f"presence_score of {prompt_word}: {presence_score}")
                if self.config.use_presence_score:
                    presence_score = output.get("presence_score", 1.0)
                    current_logits = current_logits * presence_score

                seg_logits[prompt_idx] = current_logits

                if self.debug_memory:
                    self._log_tensor_memory("seg_logits", seg_logits)
                    self._debug_print(f"[MEMORY] After fusion: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB")

                # ===== 清理当前 prompt 的中间变量，释放 GPU 内存 =====
                allocated_before = torch.cuda.memory_allocated() if self.debug_memory else 0

                # 删除 output 字典中的大 tensor
                del output

                # 删除实例头计算过程中的中间变量
                if self.config.use_instance_head:
                    del inst_current

                # 删除语义头计算过程中的中间变量
                if self.config.use_semantic_head:
                    del semantic_logits

                # 删除当前 logits
                del current_logits

                # 定期清理 GPU 内存缓存 - 每 5 个 prompt 清理一次，避免频繁清理影响性能
                if (prompt_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()

                if self.debug_memory:
                    allocated_after = torch.cuda.memory_allocated()
                    released_gb = (allocated_before - allocated_after) / 1024**3
                    self._debug_print(f"[MEMORY] After cleanup: allocated={allocated_after / 1024**3:.2f} GB (released: {released_gb:.2f} GB)")

        # Clean up inference_state to free GPU memory
        if inference_state is not None:
            for key in list(inference_state.keys()):
                if isinstance(inference_state[key], torch.Tensor):
                    del inference_state[key]

        return seg_logits, per_class_results, semantic_logits_only, instance_logits_only

    def _inference_batch_view(self, images: List[Image.Image], detailed: bool = False, image_names: Optional[List[str]] = None):
        """
        高性能批量推理实现 (True Batch Inference)。
        直接调用底层 model.backbone 和 model.forward_grounding，绕过处理器内部的单图限制。
        """
        import torchvision.transforms.v2 as v2
        from sam3.model.data_misc import FindStage

        batch_size = len(images)
        w_orig, h_orig = images[0].size
        if image_names is None:
            image_names = [f"img_{i}" for i in range(batch_size)]

        # 1. 图像预处理：将 PIL 转换为 Batch Tensor
        # 图像预处理和特征提取相当于 processor.set_image 的前半部分
        input_tensors = []
        for img in images:
            t = v2.functional.to_image(img).to(self.device)
            t = self.processor.transform(t)
            input_tensors.append(t)
        batch_input = torch.stack(input_tensors, dim=0)  # [B, 3, 1008, 1008]

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            # 2. 图像特征提取 (Encoder) - 批量并行
            backbone_out = self.processor.model.backbone.forward_image(batch_input)

            if self.debug_memory:
                img_info = ", ".join(image_names[:3])
                if len(image_names) > 3:
                    img_info += f" ... (+{len(image_names)-3} more)"
                self._debug_print(f"\n[MEMORY] ===== Batch of {batch_size} images: {img_info} =====")
                self._debug_print(f"[MEMORY] Image size: {w_orig}x{h_orig}")
                self._debug_print(f"[MEMORY] Feature extraction completed: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB")

                # Log backbone output tensors (only once)
                if "vision_features" in backbone_out:
                    self._log_tensor_memory("vision_features", backbone_out["vision_features"])
                if "backbone_fpn" in backbone_out:
                    for i, fpn_feat in enumerate(backbone_out["backbone_fpn"]):
                        self._log_tensor_memory(f"backbone_fpn[{i}]", fpn_feat)

            # 初始化批量结果 [B, num_prompts, H, W]
            batch_seg_logits = torch.zeros(
                (batch_size, self.num_prompts, h_orig, w_orig), device=self.device
            )

            # 关键：构造 Batch 模式的 FindStage
            # 每张图片会自动产生 200 个 query，所以 img_ids 只需要包含 batch 中的图片索引即可
            find_stage = FindStage(
                img_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
                text_ids=torch.zeros(batch_size, device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            # 为每一张图创建一个 dummy prompt
            dummy_geometric = self.processor.model._get_dummy_prompt(
                num_prompts=batch_size
            )

            # 3. 循环遍历每个 Prompt (Decoder / Grounding)
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                if self.debug_memory:
                    self._debug_print(f"[MEMORY] ===== Prompt [{prompt_idx+1}/{self.num_prompts}]: {prompt_word} =====")

                # ===== 使用预计算的文本特征（避免重复计算）=====
                # 清除历史文本特征
                for k in ["language_features", "language_mask", "language_embeds"]:
                    if k in backbone_out:
                        del backbone_out[k]

                # 从缓存中获取预计算的文本特征
                if self.text_features_cache is not None and prompt_idx in self.text_features_cache:
                    cached_features = self.text_features_cache[prompt_idx]
                    backbone_out.update({
                        "language_features": cached_features["language_features"],
                        "language_mask": cached_features["language_mask"],
                        "language_embeds": cached_features["language_embeds"],
                    })
                else:
                    # 回退到实时计算（兼容性）
                    text_outputs = self.processor.model.backbone.forward_text(
                        [prompt_word], device=self.device
                    )
                    backbone_out.update(text_outputs)

                # 解码阶段 (Grounding) - 处理 Batch
                # 模拟 sam3_image_processor.py中的_forward_grounding
                outputs = self.processor.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_stage,
                    geometric_prompt=dummy_geometric,
                    find_target=None,
                )

                # Only log tensor info for first prompt (avoid redundancy)
                if self.debug_memory and prompt_idx == 0:
                    self._log_tensor_memory("pred_masks", outputs["pred_masks"])
                    self._log_tensor_memory("pred_logits", outputs["pred_logits"])
                    self._log_tensor_memory("semantic_seg", outputs["semantic_seg"])

                if self.debug_memory:
                    self._debug_print(f"[MEMORY] After forward_grounding: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB")


                # pred_masks: [B, 200, 288, 288]
                # semantic_seg: [B, 1, 288, 288]
                # pred_logits: [B, 200, 1]
                # presence_score: [B, 1, 1]
                # presence_logit_dec: [B, 1]
                out_logits = outputs["pred_logits"]
                out_masks = outputs["pred_masks"]
                out_probs = out_logits.sigmoid()
                presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)

                # [B, 200]
                out_probs = (out_probs * presence_score).squeeze(-1)  # [B, 200]

                # Mask filtering: keep only queries above confidence threshold
                # But we preserve batch dimension by zeroing out low-confidence masks
                mask_keep = out_probs > self.config.confidence_threshold  # [B, 200]
                # [B, 200, 1, 1]

                mask_keep_expanded = mask_keep.unsqueeze(-1).unsqueeze(-1)

                # Zero out low-confidence queries (preserves shape)
                # [B, 200, 288, 288]
                out_masks = out_masks * mask_keep_expanded.float()
                # [B, 200]
                scores = out_probs * mask_keep.float()

                # Interpolate masks to original image size
                # [B, 200, h_orig, w_orig]
                inst_mask_logits = F.interpolate(
                    out_masks,
                    (h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()

                # Semantic segmentation
                # [B, 1, h_orig, w_orig]
                sem_mask_logits = F.interpolate(
                    outputs["semantic_seg"],
                    (h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()

                # ===== Dual-Head Fusion (Batch Version) =====
                # Initialize current_logits for this prompt
                current_batch_logits = torch.zeros(
                    (batch_size, h_orig, w_orig), device=self.device
                )

                # 1. Instance head (200 queries fusion)
                if self.config.use_instance_head:
                    # inst_mask_logits: [B, 200, h_orig, w_orig]
                    # ========================================================================
                    # 方式1（当前使用）：纯 max 融合，不乘 score
                    # ========================================================================
                    # inst_current = inst_mask_logits.max(dim=1)[0]  # [B, h_orig, w_orig]

                    # ========================================================================
                    # 方式2（SegEarthOV3 方式）：乘以 score 后再取 max
                    # 需要先计算 weighted_logits，然后取 max
                    # ========================================================================
                    # scores: [B, 200] -> expand to [B, 200, 1, 1]
                    weighted_logits = inst_mask_logits * scores.unsqueeze(-1).unsqueeze(-1)
                    inst_current = weighted_logits.max(dim=1)[0]  # [B, h_orig, w_orig]

                    # ========================================================================
                    # 方式3（阈值过滤 + max）：只保留高置信度 query
                    # ========================================================================
                    # mask_keep = scores > self.config.confidence_threshold  # [B, 200]
                    # inst_mask_filtered = inst_mask_logits * mask_keep.unsqueeze(-1).unsqueeze(-1).float()
                    # inst_current = inst_mask_filtered.max(dim=1)[0]

                    current_batch_logits = torch.max(current_batch_logits, inst_current)

                # 2. Semantic head
                if self.config.use_semantic_head:
                    # sem_mask_logits: [B, 1, h_orig, w_orig] -> [B, h_orig, w_orig]
                    sem_current = sem_mask_logits.squeeze(1)
                    current_batch_logits = torch.max(current_batch_logits, sem_current)

                # 3. Presence score filtering (if enabled)
                # Note: same as single mode - apply to fused result after dual-head fusion
                if self.config.use_presence_score:
                    # Use max presence score across queries for each image
                    img_presence = (
                        outputs["presence_logit_dec"].sigmoid().max(dim=1)[0]
                    )  # [B]
                    current_batch_logits = (
                        current_batch_logits * img_presence.unsqueeze(-1).unsqueeze(-1)
                    )

                # Store in batch_seg_logits
                batch_seg_logits[:, prompt_idx] = current_batch_logits

                if self.debug_memory:
                    self._debug_print(f"[MEMORY] After prompt {prompt_idx+1}: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB")

                # ===== 清理当前 prompt 的中间变量，释放 GPU 内存 =====
                allocated_before = torch.cuda.memory_allocated() if self.debug_memory else 0

                # 删除 outputs 字典中的大 tensor
                del outputs, out_logits, out_masks, out_probs, presence_score

                # 删除计算过程中的中间变量
                del mask_keep, mask_keep_expanded, scores
                del inst_mask_logits, sem_mask_logits
                del weighted_logits, inst_current, sem_current

                # 清理当前 prompt 的计算结果
                del current_batch_logits

                # 定期清理 GPU 内存缓存 - 每 5 个 prompt 清理一次
                if (prompt_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()

                if self.debug_memory:
                    allocated_after = torch.cuda.memory_allocated()
                    released_gb = (allocated_before - allocated_after) / 1024**3
                    self._debug_print(f"[MEMORY] After cleanup: allocated={allocated_after / 1024**3:.2f} GB (released: {released_gb:.2f} GB)")

        # Clean up backbone_out to free GPU memory
        if 'backbone_out' in dir():
            for key in list(backbone_out.keys()):
                if isinstance(backbone_out[key], torch.Tensor):
                    del backbone_out[key]
            del backbone_out
        torch.cuda.empty_cache()

        return batch_seg_logits, [{} for _ in range(batch_size)], None, None

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
            ) = self._sliding_window_inference(image, detailed=detailed, image_name=image_name)
        else:
            # Single view inference
            (
                seg_logits,
                per_class_results,
                semantic_logits_only,
                instance_logits_only,
            ) = self._inference_single_view(image, detailed=detailed, image_name=image_name)

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
                得到的结果中, 每个类别(每一行), 只保留了属于该类别的查询词的概率图
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
            # Case 1: Background is NOT in prompts - inject a zero-logit background channel
            if not self.config.use_prompted_background:
                # Create explicit background channel with zero logits
                bg_pad = torch.zeros(
                    (1, *logits.shape[1:]), device=logits.device, dtype=logits.dtype
                )
                logits_for_argmax = torch.cat([bg_pad, logits], dim=0)
                pred = torch.argmax(logits_for_argmax, dim=0)

                # Apply prob_threshold: assign pixels with max logit < threshold to background
                # This ensures ambiguous/low-confidence regions are marked as background
                max_vals = logits.max(0)[0]
                pred[max_vals < self.config.prob_threshold] = bg_idx
            else:
                # Case 2: Background IS in prompts - no need to inject background channel
                pred = torch.argmax(logits, dim=0)
                # Still apply prob_threshold filtering to suppress low-confidence predictions
                max_vals = logits.max(0)[0]
                pred[max_vals < self.config.prob_threshold] = bg_idx

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
        # seg_pred.shape like [H, W], seg_logits.shape like [num_classes, H, W]
        # when use_prompted_background is True, pixel values in seg_pred are class IDs, consistent with the order read from prompts file
        # when use_prompted_background is False, add background=0, class IDs in seg_pred are shifted by 1

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
            (
                seg_logits,
                per_class_results,
                semantic_logits_only,
                instance_logits_only,
            ) = self._inference_batch_view(images, detailed=detailed, image_names=image_names)
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
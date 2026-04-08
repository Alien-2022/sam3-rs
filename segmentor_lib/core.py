"""Core inference engine for SAM3-RS without debug/analyzer code."""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Optional, Tuple
import numpy as np
from .debug import MemoryDebugger


class InferenceEngine:
    """Core inference logic for SAM3-RS.

    This class contains only the essential inference logic without
    debug code, statistics collection, or analysis features.
    """

    def __init__(self, processor, prompts: Dict, config, device):
        """
        Args:
            processor: SAM3Processor instance
            prompts: Dict with 'names', 'indices', 'mapping'
            config: InferenceConfig instance
            device: torch.device
        """
        self.processor = processor
        self.prompts = prompts
        self.config = config
        self.device = device
        self.num_classes = max(prompts["indices"]) + 1 if prompts else 0
        self.num_prompts = len(prompts["names"]) if prompts else 0

        # Initialize memory debugger
        self.memory_debugger = MemoryDebugger(
            enabled=getattr(config, 'debug_memory', False),
            log_file=getattr(config, 'debug_log_file', None)
        )

        # Convert class indices to tensor
        if prompts:
            self.query_indices = torch.tensor(
                prompts["indices"], dtype=torch.int64, device=device
            )
        else:
            self.query_indices = None

        # Initialize analyzers
        self.analyzers = []
        self._init_analyzers(config)

    def _init_analyzers(self, config):
        """Initialize and attach analyzers based on config."""
        from .analyzers import PresenceScoreAnalyzer

        self.analyzers = []

        if getattr(config, 'analyze_presence_score', False):
            self.analyzers.append(PresenceScoreAnalyzer(enabled=True))

    def register_analyzer(self, analyzer):
        """Register an external analyzer."""
        self.analyzers.append(analyzer)

    def get_analyzer(self, analyzer_class):
        """Get analyzer by class type.

        Args:
            analyzer_class: The analyzer class to search for

        Returns:
            The first matching analyzer instance, or None if not found
        """
        for analyzer in self.analyzers:
            if isinstance(analyzer, analyzer_class):
                return analyzer
        return None

    def _call_hooks(self, hook_name: str, context: Dict):
        """Call all analyzers for a specific hook point."""
        for analyzer in self.analyzers:
            hook_method = getattr(analyzer, hook_name, None)
            if hook_method:
                hook_method(context)

    def inference_single_view(
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
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            inference_state = self.processor.set_image(image)

            # Call before_inference hooks
            self._call_hooks('on_before_inference', {
                "image": image,
                "image_name": image_name,
                "mode": "single_view"
            })

            # Pre-compute per-class geometric prompts from visual prototypes (geo_box mode)
            inject_mode = getattr(self, 'prototype_inject_mode', 'lang')
            proto_geo_prompts = None
            if inject_mode == "geo_box" and self.visual_prototype_bank is not None:
                backbone_out_preview = inference_state["backbone_out"]
                proto_geo_prompts = self._compute_proto_geo_prompts(backbone_out_preview, batch_size=1)

            # Process each prompt
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # 显存追踪：每个prompt开始
                self.memory_debugger.debug_print(f"\n[PROMPT {prompt_idx}/{self.num_prompts}] {prompt_word}")
                self.memory_debugger.log_cuda_memory(f"Prompt {prompt_idx} start: {prompt_word}")

                # Reset prompts for clean inference
                self.processor.reset_all_prompts(inference_state)

                # Get backbone_out for this image (cache original for vision injection)
                backbone_out = inference_state["backbone_out"]
                original_backbone_fpn = backbone_out.get("backbone_fpn")

                # Get class_id for this prompt
                class_id = self.prompts["indices"][prompt_idx]

                # Set geometric_prompt: use proto-based box prompt or dummy
                if proto_geo_prompts is not None and prompt_idx in proto_geo_prompts:
                    inference_state["geometric_prompt"] = proto_geo_prompts[prompt_idx]
                elif "geometric_prompt" not in inference_state:
                    inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

                # Inject visual prototype into vision features if in vision mode
                if inject_mode == "vision":
                    self._inject_proto_to_vision(backbone_out, class_id)

                output = None
                # Use pre-computed text features if available, otherwise real-time encoding
                if self.text_features_cache is not None and prompt_idx in self.text_features_cache:
                    cached_features = self.text_features_cache[prompt_idx]
                    # Inject visual prototype
                    if inject_mode == "concat":
                        # Concat mode: append prototype as extra token
                        injected = self._inject_proto_as_token(cached_features, class_id)
                        backbone_out.update(injected)
                    elif inject_mode == "replace_last":
                        injected = self._inject_proto_replace_last(cached_features, class_id)
                        backbone_out.update(injected)
                    elif (inject_mode == "lang"
                            and self.visual_prototype_bank is not None
                            and class_id in self.visual_prototype_bank):
                        proto = self.visual_prototype_bank[class_id]  # [1, 1, 256], L2-norm=1
                        lang_feats = cached_features["language_features"].to(self.device)
                        proto = proto.to(self.device).to(lang_feats.dtype)
                        # Scale-aligned injection: scale proto to text_features norm * alpha
                        text_norm = lang_feats.norm()
                        alpha = getattr(self, 'prototype_alpha', 1.0)
                        lang_feats = lang_feats + alpha * text_norm * proto
                        backbone_out.update({
                            "language_features": lang_feats,
                            "language_mask": cached_features["language_mask"],
                            "language_embeds": cached_features.get("language_embeds"),
                        })
                    else:
                        backbone_out.update({
                            "language_features": cached_features["language_features"],
                            "language_mask": cached_features["language_mask"],
                            "language_embeds": cached_features["language_embeds"],
                        })
                    # Run grounding inference
                    self.memory_debugger.log_cuda_memory(f"  [Before _forward_grounding] using cached text features")
                    output = self.processor._forward_grounding(inference_state)
                    self.memory_debugger.log_cuda_memory(f"  [After _forward_grounding] using cached text features")
                    if output and isinstance(output, dict):
                        for k, v in output.items():
                            if isinstance(v, torch.Tensor):
                                self.memory_debugger.log_tensor_memory(f"  output.{k}", v)
                else:
                    # Fallback to real-time text encoding
                    self.memory_debugger.log_cuda_memory(f"  [Before set_text_prompt] real-time encoding")
                    output = self.processor.set_text_prompt(
                        state=inference_state, prompt=prompt_word
                    )
                    self.memory_debugger.log_cuda_memory(f"  [After set_text_prompt] real-time encoding")

                # Clean up output intermediate tensors to save memory
                # Only keep what we need, but preserve semantic_seg if using semantic head
                if output is not None:
                    # Extract essential fields and discard the rest
                    cleaned_output = {
                        "masks_logits": output.get("masks_logits"),
                        "pred_logits": output.get("pred_logits"),
                        "pred_masks": output.get("pred_masks"),
                        "presence_score": output.get("presence_score"),
                        "boxes": output.get("boxes"),  # boxes from processor (filtered by confidence_threshold)
                        "scores": output.get("scores"),  # Needed for instance head
                    }
                    # Keep semantic_seg if semantic head is enabled
                    if self.config.use_semantic_head and "semantic_seg" in output:
                        cleaned_output["semantic_seg"] = output["semantic_seg"]
                    output = cleaned_output

                self.memory_debugger.log_cuda_memory(f"  [After cleaning output]")

                # Call after_prompt hooks
                self._call_hooks('on_after_prompt', {
                    "prompt_idx": prompt_idx,
                    "prompt_word": prompt_word,
                    "image_name": image_name,
                    "output": output
                })

                # Store per-class detailed results if requested
                # Only collect boxes and scores for iterative visual prompt refinement
                # Skipping masks, masks_logits, and semantic_logits to save time/memory
                if detailed:
                    per_class_results[prompt_word] = {
                        # "masks": output["masks"].cpu(),  # Commented for performance
                        # "masks_logits": output["masks_logits"].cpu(),  # Not needed for logit-level fusion
                        "boxes": output.get("boxes"),  # Keep on GPU for iterative refinement
                        "scores": output.get("scores"),  # Keep on GPU for iterative refinement
                        # "semantic_logits": output["semantic_seg"].cpu(),  # Commented for performance
                        # "presence_score": output.get("presence_score", 1.0),  # Commented for performance
                    }

                # ===== Dual-Head Fusion =====
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
                            inst_score = output["scores"][inst_id]

                            # inst_logits.shape: [1, 1024, 1024] -> interpolate to [H, W]
                            if inst_logits.dim() == 3:
                                inst_logits = inst_logits.unsqueeze(0)

                            inst_logits = F.interpolate(
                                inst_logits,
                                size=(h, w),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()

                            # Apply instance head threshold BEFORE fusion (if configured)
                            if self.config.instance_prob_thresholds and class_id in self.config.instance_prob_thresholds:
                                inst_thresh = self.config.instance_prob_thresholds[class_id]
                                inst_logits = inst_logits * (inst_logits >= inst_thresh).float()

                            inst_current = torch.max(inst_current, inst_logits * inst_score)

                    current_logits = torch.max(current_logits, inst_current)
                    instance_logits_only[prompt_idx] = inst_current

                # 2. Semantic head
                if self.config.use_semantic_head:
                    # semantic_logits: [1, 1, H_orig, W_orig] (4D tensor)
                    semantic_logits = output["semantic_seg"]
                    semantic_logits = F.interpolate(
                        semantic_logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()  # [H, W]

                    # Apply semantic head threshold BEFORE fusion (if configured)
                    if self.config.semantic_prob_thresholds and class_id in self.config.semantic_prob_thresholds:
                        sem_thresh = self.config.semantic_prob_thresholds[class_id]
                        semantic_logits = semantic_logits * (semantic_logits >= sem_thresh).float()

                    # Apply presence score to semantic head if enabled
                    if self.config.use_presence_score and self.config.presence_score_mode == "before_fusion":
                        presence_score = output.get("presence_score", 0)
                        semantic_logits = semantic_logits * presence_score

                    # Fusion: take max of instance and semantic predictions
                    current_logits = torch.max(current_logits, semantic_logits)
                    semantic_logits_only[prompt_idx] = semantic_logits

                # 3. Presence score filtering (after fusion mode)
                if self.config.use_presence_score and self.config.presence_score_mode == "after_fusion":
                    presence_score = output.get("presence_score", 0)
                    current_logits = current_logits * presence_score

                seg_logits[prompt_idx] = current_logits

                # Restore original backbone_fpn if vision injection was applied
                if original_backbone_fpn is not None:
                    backbone_out["backbone_fpn"] = original_backbone_fpn

                # ===== Clean up current prompt's intermediate variables =====
                del output
                if self.config.use_instance_head:
                    del inst_current
                if self.config.use_semantic_head:
                    del semantic_logits
                del current_logits

                self.memory_debugger.log_cuda_memory(f"  [After deleting intermediate vars]")

                # 显存使用日志
                mem_usage_gb = torch.cuda.memory_allocated() / 1024**3
                self.memory_debugger.debug_print(f"  [GC] current usage: {mem_usage_gb:.2f}GB")

        # Call after_inference hooks
        self._call_hooks('on_after_inference', {
            "image_name": image_name,
            "mode": "single_view"
        })

        # Clean up inference_state to free GPU memory
        if inference_state is not None:
            for key in list(inference_state.keys()):
                if isinstance(inference_state[key], torch.Tensor):
                    del inference_state[key]

        return seg_logits, per_class_results, semantic_logits_only, instance_logits_only, None

    def inference_batch_view(
        self, images: List[Image.Image], detailed: bool = False, image_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
        """High-performance batch inference implementation.

        Directly calls model.backbone and model.forward_grounding,
        bypassing processor's single-image limitations.
        """
        import torchvision.transforms.v2 as v2
        from sam3.model.data_misc import FindStage

        batch_size = len(images)
        w_orig, h_orig = images[0].size
        if image_names is None:
            image_names = [f"img_{i}" for i in range(batch_size)]

        # 1. Image preprocessing: convert PIL to batch tensor
        input_tensors = []
        for img in images:
            t = v2.functional.to_image(img).to(self.device)
            t = self.processor.transform(t)
            input_tensors.append(t)
        batch_input = torch.stack(input_tensors, dim=0)  # [B, 3, 1008, 1008]

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            # 2. Image feature extraction (Encoder) - batch parallel
            backbone_out = self.processor.model.backbone.forward_image(batch_input)

            # Cache original backbone_fpn for vision injection restoration
            original_backbone_fpn = backbone_out.get("backbone_fpn")

            # Call before_inference hooks
            self._call_hooks('on_before_inference', {
                "images": images,
                "image_names": image_names,
                "mode": "batch_view"
            })

            # Initialize batch results [B, num_prompts, H, W]
            batch_seg_logits = torch.zeros(
                (batch_size, self.num_prompts, h_orig, w_orig), device=self.device
            )

            # Initialize per-class results for detailed mode
            batch_per_class_results = [{} for _ in range(batch_size)] if detailed else None

            # Construct FindStage for batch mode
            find_stage = FindStage(
                img_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
                text_ids=torch.zeros(batch_size, device=self.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            # Create dummy geometric prompt for each image
            dummy_geometric = self.processor.model._get_dummy_prompt(
                num_prompts=batch_size
            )

            # Pre-compute per-class geometric prompts from visual prototypes (geo_box mode)
            inject_mode = getattr(self, 'prototype_inject_mode', 'lang')
            proto_geo_prompts = None
            if inject_mode == "geo_box" and self.visual_prototype_bank is not None:
                proto_geo_prompts = self._compute_proto_geo_prompts(backbone_out, batch_size)

            # 3. Loop through each Prompt (Decoder / Grounding)
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # ===== Use pre-computed text features =====
                # Clear historical text features
                for k in ["language_features", "language_mask", "language_embeds"]:
                    if k in backbone_out:
                        del backbone_out[k]

                # Get pre-computed text features from cache
                class_id = self.prompts["indices"][prompt_idx]

                # Inject visual prototype into vision features if in vision mode
                if inject_mode == "vision":
                    self._inject_proto_to_vision(backbone_out, class_id)

                if self.text_features_cache is not None and prompt_idx in self.text_features_cache:
                    cached_features = self.text_features_cache[prompt_idx]
                    # Inject visual prototype
                    if inject_mode == "concat":
                        # Concat mode: append prototype as extra token
                        injected = self._inject_proto_as_token(cached_features, class_id)
                        backbone_out.update(injected)
                    elif inject_mode == "replace_last":
                        injected = self._inject_proto_replace_last(cached_features, class_id)
                        backbone_out.update(injected)
                    elif (inject_mode == "lang"
                            and self.visual_prototype_bank is not None
                            and class_id in self.visual_prototype_bank):
                        proto = self.visual_prototype_bank[class_id]  # [1, 1, 256], L2-norm=1
                        lang_feats = cached_features["language_features"].to(self.device)
                        proto = proto.to(self.device).to(lang_feats.dtype)
                        # Scale-aligned injection: scale proto to text_features norm * alpha
                        text_norm = lang_feats.norm()
                        alpha = getattr(self, 'prototype_alpha', 1.0)
                        lang_feats = lang_feats + alpha * text_norm * proto
                        backbone_out.update({
                            "language_features": lang_feats,
                            "language_mask": cached_features["language_mask"],
                            "language_embeds": cached_features.get("language_embeds"),
                        })
                    else:
                        backbone_out.update({
                            "language_features": cached_features["language_features"],
                            "language_mask": cached_features["language_mask"],
                            "language_embeds": cached_features["language_embeds"],
                        })
                else:
                    # Fallback to real-time computation
                    text_outputs = self.processor.model.backbone.forward_text(
                        [prompt_word], device=self.device
                    )
                    backbone_out.update(text_outputs)

                # Select geometric prompt: proto-based box or dummy
                if proto_geo_prompts is not None and prompt_idx in proto_geo_prompts:
                    geo_prompt = proto_geo_prompts[prompt_idx]
                else:
                    geo_prompt = dummy_geometric

                # Decoding stage (Grounding) - process Batch
                outputs = self.processor.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_stage,
                    geometric_prompt=geo_prompt,
                    find_target=None,
                )

                # Call after_prompt hooks
                self._call_hooks('on_after_prompt', {
                    "prompt_idx": prompt_idx,
                    "prompt_word": prompt_word,
                    "image_names": image_names,
                    "output": outputs
                })

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
                out_probs = (out_probs * presence_score).squeeze(-1)

                # Mask filtering: keep only queries above confidence threshold
                mask_keep = out_probs > self.config.confidence_threshold  # [B, 200]
                mask_keep_expanded = mask_keep.unsqueeze(-1).unsqueeze(-1)

                # Zero out low-confidence queries
                out_masks = out_masks * mask_keep_expanded.float()
                scores = out_probs * mask_keep.float()

                # Interpolate masks to original image size
                inst_mask_logits = F.interpolate(
                    out_masks,
                    (h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()

                # Semantic segmentation
                sem_mask_logits = F.interpolate(
                    outputs["semantic_seg"],
                    (h_orig, w_orig),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()

                # ===== Dual-Head Fusion (Batch Version) =====
                current_batch_logits = torch.zeros(
                    (batch_size, h_orig, w_orig), device=self.device
                )

                # 1. Instance head (200 queries fusion)
                if self.config.use_instance_head:
                    # scores: [B, 200] -> expand to [B, 200, 1, 1]
                    weighted_logits = inst_mask_logits * scores.unsqueeze(-1).unsqueeze(-1)
                    inst_current = weighted_logits.max(dim=1)[0]

                    current_batch_logits = torch.max(current_batch_logits, inst_current)

                # 2. Semantic head
                if self.config.use_semantic_head:
                    sem_current = sem_mask_logits.squeeze(1)

                    # Apply semantic head threshold BEFORE fusion (if configured)
                    if self.config.semantic_prob_thresholds and class_id in self.config.semantic_prob_thresholds:
                        sem_thresh = self.config.semantic_prob_thresholds[class_id]
                        sem_current = sem_current * (sem_current >= sem_thresh).float()

                    # Apply presence score to semantic head if enabled
                    if self.config.use_presence_score and self.config.presence_score_mode == "before_fusion":
                        img_presence = (
                            outputs["presence_logit_dec"].sigmoid().max(dim=1)[0]
                        )
                        sem_current = sem_current * img_presence.unsqueeze(-1).unsqueeze(-1)

                    current_batch_logits = torch.max(current_batch_logits, sem_current)

                # 3. Presence score filtering (after fusion mode)
                if self.config.use_presence_score and self.config.presence_score_mode == "after_fusion":
                    img_presence = (
                        outputs["presence_logit_dec"].sigmoid().max(dim=1)[0]
                    )
                    current_batch_logits = (
                        current_batch_logits * img_presence.unsqueeze(-1).unsqueeze(-1)
                    )

                # Store in batch_seg_logits
                batch_seg_logits[:, prompt_idx] = current_batch_logits

                # Restore original backbone_fpn if vision injection was applied
                if original_backbone_fpn is not None:
                    backbone_out["backbone_fpn"] = original_backbone_fpn

                # Collect per-class detailed results if requested (boxes and scores)
                if detailed and batch_per_class_results is not None:
                    # inst_mask_logits: [B, 200, H, W], scores: [B, 200], mask_keep: [B, 200]
                    for b in range(batch_size):
                        # Get valid instances for this image
                        img_mask_keep = mask_keep[b]  # [200]
                        valid_count = img_mask_keep.sum().item()

                        if valid_count > 0:
                            # Get valid masks and scores
                            valid_indices = torch.where(img_mask_keep)[0]
                            valid_masks = inst_mask_logits[b, valid_indices]  # [N, H, W]
                            valid_scores = scores[b, valid_indices]  # [N]

                            # Convert masks to boxes
                            valid_boxes = self._masks_to_boxes(valid_masks)  # [N, 4]

                            # Store in per-class results
                            if prompt_word not in batch_per_class_results[b]:
                                batch_per_class_results[b][prompt_word] = {
                                    "boxes": [],
                                    "scores": []
                                }
                            batch_per_class_results[b][prompt_word]["boxes"].append(valid_boxes)
                            batch_per_class_results[b][prompt_word]["scores"].append(valid_scores)

                # ===== Clean up current prompt's intermediate variables =====
                del outputs, out_logits, out_masks, out_probs, presence_score
                del mask_keep, mask_keep_expanded, scores
                del inst_mask_logits, sem_mask_logits
                if self.config.use_instance_head:
                    del weighted_logits, inst_current
                if self.config.use_semantic_head:
                    del sem_current
                del current_batch_logits

                # Periodic GPU cache cleanup
                if (prompt_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()

        # Call after_inference hooks
        self._call_hooks('on_after_inference', {
            "image_names": image_names,
            "mode": "batch_view"
        })

        # Clean up backbone_out to free GPU memory
        if 'backbone_out' in dir():
            for key in list(backbone_out.keys()):
                if isinstance(backbone_out[key], torch.Tensor):
                    del backbone_out[key]
        torch.cuda.empty_cache()

        # Concatenate boxes and scores from all prompts for each image
        if detailed and batch_per_class_results is not None:
            for b in range(batch_size):
                for prompt_word in batch_per_class_results[b]:
                    if batch_per_class_results[b][prompt_word]["boxes"]:
                        batch_per_class_results[b][prompt_word]["boxes"] = torch.cat(
                            batch_per_class_results[b][prompt_word]["boxes"], dim=0
                        )
                        batch_per_class_results[b][prompt_word]["scores"] = torch.cat(
                            batch_per_class_results[b][prompt_word]["scores"], dim=0
                        )

        return batch_seg_logits, batch_per_class_results if detailed else [{} for _ in range(batch_size)], None, None, None

    def set_text_features_cache(self, cache: Optional[Dict]):
        """Set the pre-computed text features cache."""
        self.text_features_cache = cache

    def set_visual_prototype_bank(self, prototypes: Optional[Dict[int, torch.Tensor]],
                                   alpha: float = 1.0, mode: str = "lang",
                                   geo_threshold: float = 0.3, geo_topk: int = 10,
                                   geo_correlation: str = "cosine",
                                   geo_presence_threshold: float = 0.0,
                                   geo_fpn_level: int = -1):
        """Set the visual prototype bank for class-conditional visual guidance.

        Args:
            prototypes: Dict mapping class_id -> prototype tensor [1, 1, 256] (L2-normalized).
            alpha: Injection strength. Default 1.0.
            mode: Injection mode.
                - "lang": add to language_features (text encoder space, cross-space addition)
                - "concat": append prototype as an extra token to language_features
                - "replace_last": replace last token with scaled prototype
                - "vision": add to backbone_fpn[-1] (vision backbone space)
                - "geo_box": bounding boxes from response maps → geometric prompts
                - "geo_point": top-K points from response maps → geometric prompts
            geo_threshold: threshold for response map binarization (box mode only).
            geo_topk: number of top-K points to select as point prompts (point mode only).
            geo_correlation: correlation method for response map computation.
                - "cosine": cosine similarity (default)
                - "dot": raw dot product (preserves magnitude)
                - "euclidean_inv": inverse euclidean distance 1/(1+||f-p||)
                - "channel_attn": channel-wise attention weighting
            geo_presence_threshold: minimum max-response value to add geo prompt.
                0.0 = always add (default). Set higher to filter absent classes.
            geo_fpn_level: which FPN level to compute response map on.
                -1 = 72x72 (default, matches encoder), -2 = 144x144, -3 = 288x288.
                Points/boxes are always normalized to [0,1] so any level works.
        """
        self.visual_prototype_bank = prototypes
        self.prototype_alpha = alpha
        self.prototype_inject_mode = mode
        self.geo_box_threshold = geo_threshold
        self.geo_topk = geo_topk
        self.geo_correlation = geo_correlation
        self.geo_presence_threshold = geo_presence_threshold
        self.geo_fpn_level = geo_fpn_level

    def _inject_proto_as_token(self, cached_features: dict, class_id: int) -> dict:
        """Append visual prototype as an extra token to language_features.

        Extends language_features from [32, 1, 256] to [33, 1, 256] and
        language_mask from [1, 32] to [1, 33]. The prototype is scaled to
        match the norm of the original language_features (mean per-token norm),
        so the cross-attention can naturally attend to it.

        Returns:
            dict with updated language_features, language_mask, language_embeds.
        """
        if self.visual_prototype_bank is None or class_id not in self.visual_prototype_bank:
            return cached_features

        proto = self.visual_prototype_bank[class_id]  # [1, 1, 256], L2-norm=1
        lang_feats = cached_features["language_features"].to(self.device)  # [T, 1, 256]
        lang_mask = cached_features["language_mask"].to(self.device)       # [1, T]
        alpha = getattr(self, 'prototype_alpha', 1.0)

        T, B, D = lang_feats.shape  # T=32, B=1, D=256

        # Scale prototype to match per-token norm of language_features
        # lang_feats norm ~ 93 globally, per-token norm ~ 93/sqrt(32) ~ 16.4
        per_token_norm = lang_feats.flatten(1).norm(dim=1).mean()  # mean L2 norm per token
        proto_token = alpha * per_token_norm * proto  # [1, 1, 256]
        proto_token = proto_token.to(lang_feats.dtype)  # match dtype (bfloat16)

        # Concatenate: [T, 1, 256] + [1, 1, 256] -> [T+1, 1, 256]
        new_feats = torch.cat([lang_feats, proto_token], dim=0)  # [33, 1, 256]

        # Extend mask: [1, 32] -> [1, 33], new token is valid (False = not masked)
        new_mask = torch.cat([
            lang_mask,
            torch.zeros(1, 1, dtype=lang_mask.dtype, device=self.device),
        ], dim=1)  # [1, 33]

        # language_embeds is NOT used by the grounding decoder, pass through unchanged
        return {
            "language_features": new_feats,
            "language_mask": new_mask,
            "language_embeds": cached_features.get("language_embeds"),
        }

    def _inject_proto_replace_last(self, cached_features: dict, class_id: int) -> dict:
        """Replace the last token of language_features with a scaled visual prototype.

        Keeps shape [32, 1, 256] and mask [1, 32] unchanged. The prototype is scaled
        to match the per-token norm of language_features so it looks like a natural text
        token to the cross-attention layers.

        Returns:
            dict with updated language_features, language_mask, language_embeds.
        """
        if self.visual_prototype_bank is None or class_id not in self.visual_prototype_bank:
            return cached_features

        proto = self.visual_prototype_bank[class_id]  # [1, 1, 256], L2-norm=1
        lang_feats = cached_features["language_features"].to(self.device)  # [T, 1, 256]
        alpha = getattr(self, 'prototype_alpha', 1.0)

        T, B, D = lang_feats.shape  # T=32, B=1, D=256

        # Scale prototype to match per-token norm
        per_token_norm = lang_feats.flatten(1).norm(dim=1).mean()
        proto_scaled = alpha * per_token_norm * proto  # [1, 1, 256]
        proto_scaled = proto_scaled.to(lang_feats.dtype)

        # Replace last token (clone to avoid modifying cached features)
        new_feats = lang_feats.clone()
        new_feats[-1:] = proto_scaled  # replace position T-1

        return {
            "language_features": new_feats,
            "language_mask": cached_features["language_mask"],
            "language_embeds": cached_features.get("language_embeds"),
        }

    def _inject_proto_to_vision(self, backbone_out: dict, class_id: int):
        """Inject visual prototype into backbone_fpn[-1] (vision feature space).

        Prototype is broadcast-added to all spatial positions of the last FPN level.
        This works because the prototype was extracted from the same FPN level and
        thus lives in the same feature space.
        """
        if self.visual_prototype_bank is None or class_id not in self.visual_prototype_bank:
            return
        if "backbone_fpn" not in backbone_out:
            return

        fpn = backbone_out["backbone_fpn"]
        if not isinstance(fpn, list) or len(fpn) == 0:
            return

        proto = self.visual_prototype_bank[class_id]  # [1, 1, 256]
        # Target the last FPN level (same level used for prototype extraction)
        feat = fpn[-1]  # [B, C, H, W], e.g. [1, 256, 72, 72]
        proto = proto.to(feat.device, dtype=feat.dtype)  # [1, 1, 256]

        # Scale: alpha * feat_channel_norm * proto
        # feat_channel_norm = mean L2 norm across spatial positions (per-channel)
        alpha = getattr(self, 'prototype_alpha', 1.0)
        # Use the mean spatial feature norm as reference
        spatial_feats = feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        feat_norm = spatial_feats.norm(dim=-1).mean()  # scalar: mean L2 norm of spatial vectors
        scaled_proto = alpha * feat_norm * proto  # [1, 1, 256]

        # Broadcast add: [1, 1, 256] -> [B, C, H, W]
        # proto shape [1, 1, 256] = [B_spatial=1, C=1, D=256], need to match [B, 256, H, W]
        # Reshape proto to [1, 256, 1, 1] and broadcast
        scaled_proto = scaled_proto.squeeze(0).permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # [256, 1, 1]

        # Deep copy the FPN list to avoid modifying cached features for other prompts
        fpn_new = list(fpn)
        fpn_new[-1] = feat + scaled_proto  # [B, 256, H, W]
        backbone_out["backbone_fpn"] = fpn_new

    def _masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """Convert binary masks to bounding boxes.

        Args:
            masks: [N, H, W] tensor of binary masks (values > 0 are considered foreground)

        Returns:
            boxes: [N, 4] tensor of boxes in [x1, y1, x2, y2] format
        """
        N, H, W = masks.shape
        if N == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=masks.dtype)

        # Create coordinate grids
        y_coords = torch.arange(H, device=masks.device, dtype=masks.dtype).view(1, H, 1)
        x_coords = torch.arange(W, device=masks.device, dtype=masks.dtype).view(1, 1, W)

        # Expand to [N, H, W]
        y_coords = y_coords.expand(N, H, W)
        x_coords = x_coords.expand(N, H, W)

        # Mask out background pixels (set to large/small values that won't affect min/max)
        masked_y = torch.where(masks > 0, y_coords, torch.full_like(y_coords, H))
        masked_x = torch.where(masks > 0, x_coords, torch.full_like(x_coords, W))
        masked_y_neg = torch.where(masks > 0, y_coords, torch.full_like(y_coords, -1))
        masked_x_neg = torch.where(masks > 0, x_coords, torch.full_like(x_coords, -1))

        # Find bounding box coordinates
        y1 = masked_y_neg.view(N, -1).max(dim=1)[0]
        x1 = masked_x_neg.view(N, -1).max(dim=1)[0]
        y2 = masked_y.view(N, -1).min(dim=1)[0]
        x2 = masked_x.view(N, -1).min(dim=1)[0]

        # Clamp to valid range
        y1 = torch.clamp(y1, 0, H - 1)
        x1 = torch.clamp(x1, 0, W - 1)
        y2 = torch.clamp(y2, 0, H - 1)
        x2 = torch.clamp(x2, 0, W - 1)

        # Stack to [N, 4]
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        return boxes

    def _compute_proto_geo_prompts(self, backbone_out: dict, batch_size: int):
        """Compute per-class geometric (box) prompts from visual prototype response maps.

        For each class with a visual prototype:
        1. Select FPN level and extract features
        2. Compute response map between spatial features and prototype
        3. Presence filtering: skip classes with max-response below threshold
        4. Extract geometric prompts:
           - "geo_box": threshold → binary mask → bounding box
           - "geo_point": top-K highest response points
        5. Return Prompt objects per prompt_idx

        Args:
            backbone_out: Dict containing backbone features (must have "backbone_fpn")
            batch_size: Number of images in the batch

        Returns:
            Dict mapping prompt_idx -> Prompt object with geometric prompt,
            or empty Prompt if the class has no prototype or response too weak.
        """
        from sam3.model.geometry_encoders import Prompt

        if self.visual_prototype_bank is None:
            return {}

        if "backbone_fpn" not in backbone_out:
            return {}

        fpn_list = backbone_out["backbone_fpn"]
        fpn_level = getattr(self, 'geo_fpn_level', -1)
        fpn_feats = fpn_list[fpn_level]  # [B, 256, H, W]
        B, C, H, W = fpn_feats.shape
        device = fpn_feats.device
        dtype = fpn_feats.dtype

        # Compute spatial features: [B, 256, H*W]
        feats_flat = fpn_feats.flatten(2)  # [B, 256, H*W]

        geo_mode = getattr(self, 'prototype_inject_mode', 'lang')
        correlation = getattr(self, 'geo_correlation', 'cosine')
        threshold = getattr(self, 'geo_box_threshold', 0.3)
        topk = getattr(self, 'geo_topk', 10)
        presence_thresh = getattr(self, 'geo_presence_threshold', 0.0)

        prompt_geo = {}

        for prompt_idx in range(self.num_prompts):
            class_id = self.prompts["indices"][prompt_idx]

            if class_id not in self.visual_prototype_bank:
                prompt_geo[prompt_idx] = Prompt()  # empty prompt (CLS token only)
                continue

            proto = self.visual_prototype_bank[class_id].to(device, dtype=dtype)  # [1, 1, 256]
            proto_flat = proto.flatten(1)  # [1, 256]

            # Compute response map based on correlation method
            response = self._compute_response_map(feats_flat, proto_flat, B, H, W,
                                                   method=correlation)  # [B, H, W]

            # Presence filtering: skip if max response too low
            max_response = response.view(B, -1).max(dim=1)[0]  # [B]
            if (max_response < presence_thresh).all():
                prompt_geo[prompt_idx] = Prompt()
                continue

            # Build geometric prompt based on mode
            if geo_mode == "geo_box":
                prompt_geo[prompt_idx] = self._build_box_prompt(
                    response, B, H, W, device, dtype, threshold)
            elif geo_mode == "geo_point":
                prompt_geo[prompt_idx] = self._build_point_prompt(
                    response, B, H, W, device, dtype, topk, max_response, presence_thresh)
            else:
                prompt_geo[prompt_idx] = Prompt()

        return prompt_geo

    def _compute_response_map(self, feats_flat, proto_flat, B, H, W, method="cosine"):
        """Compute response map between spatial features and prototype.

        Args:
            feats_flat: [B, 256, H*W] spatial features
            proto_flat: [1, 256] prototype
            B, H, W: batch, height, width
            method: correlation method

        Returns:
            response: [B, H, W] response map
        """
        device, dtype = feats_flat.device, feats_flat.dtype

        if method == "cosine":
            feats_norm = F.normalize(feats_flat, p=2, dim=1)   # [B, 256, H*W]
            proto_norm = F.normalize(proto_flat, p=2, dim=1)   # [1, 256]
            response = torch.bmm(proto_norm.unsqueeze(0).expand(B, -1, -1), feats_norm)
            response = response.view(B, H, W)

        elif method == "dot":
            # Raw dot product (no normalization, preserves magnitude)
            response = torch.bmm(proto_flat.unsqueeze(0).expand(B, -1, -1), feats_flat)
            response = response.view(B, H, W)

        elif method == "euclidean_inv":
            # Inverse euclidean distance: 1 / (1 + ||f - p||)
            proto_expanded = proto_flat.unsqueeze(0).expand(B, -1, -1)  # [B, 256, H*W]
            diff = feats_flat - proto_expanded  # [B, 256, H*W]
            dist = diff.norm(dim=1)  # [B, H*W]
            response = (1.0 / (1.0 + dist)).view(B, H, W)

        elif method == "channel_attn":
            # Channel-wise attention: prototype as channel weights
            # proto [1, 256] -> weights, apply softmax
            weights = F.softmax(proto_flat, dim=1)  # [1, 256]
            # feats [B, 256, H*W] -> weighted sum across channels
            response = (feats_flat * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
            response = response.view(B, H, W)

        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return response

    def _build_box_prompt(self, response, B, H, W, device, dtype, threshold):
        """Build a box geometric prompt from response map.

        Args:
            response: [B, H, W] response map
            B, H, W: dimensions
            device, dtype: torch device and dtype
            threshold: binarization threshold

        Returns:
            Prompt object with box embeddings
        """
        from sam3.model.geometry_encoders import Prompt

        binary_mask = (response > threshold).float()  # [B, H, W]
        boxes_xyxy = self._masks_to_boxes(binary_mask)  # [B, 4]

        valid_mask = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])

        if valid_mask.any():
            x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
            boxes_cxcywh = torch.stack([
                ((x1 + x2) / 2) / W,
                ((y1 + y2) / 2) / H,
                (x2 - x1) / W,
                (y2 - y1) / H,
            ], dim=1).clamp(0, 1)
            for b in range(B):
                if not valid_mask[b]:
                    boxes_cxcywh[b] = torch.tensor([0.5, 0.5, 1.0, 1.0], device=device, dtype=dtype)
        else:
            boxes_cxcywh = torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=device, dtype=dtype)
            boxes_cxcywh = boxes_cxcywh.expand(B, -1)

        return Prompt(
            box_embeddings=boxes_cxcywh.unsqueeze(0),  # [1, B, 4]
            box_mask=torch.zeros(B, 1, dtype=torch.bool, device=device),
            box_labels=torch.ones(1, B, dtype=torch.long, device=device),
        )

    def _build_point_prompt(self, response, B, H, W, device, dtype, topk,
                             max_response, presence_thresh):
        """Build a point geometric prompt from response map (top-K points).

        Args:
            response: [B, H, W] response map
            B, H, W: dimensions
            device, dtype: torch device and dtype
            topk: number of top points to select
            max_response: [B] max response per image
            presence_thresh: minimum response to consider present

        Returns:
            Prompt object with point embeddings, or empty Prompt if absent.
        """
        from sam3.model.geometry_encoders import Prompt

        # Collect valid points per image
        all_points = []  # list of [N_valid, 2] per image
        all_labels = []  # list of [N_valid] per image
        N_points = topk  # fixed number of point slots

        for b in range(B):
            if max_response[b] < presence_thresh:
                # Class not present: use dummy points at image center
                pts = torch.full((N_points, 2), 0.5, device=device, dtype=dtype)
                all_points.append(pts)
                all_labels.append(torch.ones(N_points, dtype=torch.long, device=device))
            else:
                resp = response[b]  # [H, W]
                flat = resp.flatten()  # [H*W]
                topk_vals, topk_idx = flat.topk(min(topk, H * W))

                # Convert flat indices to (y, x) -> normalized (x, y) in [0, 1]
                y_idx = (topk_idx // W).float() / H
                x_idx = (topk_idx % W).float() / W
                pts = torch.stack([x_idx, y_idx], dim=1)  # [topk, 2]
                pts = pts.clamp(0, 1)

                all_points.append(pts)
                all_labels.append(torch.ones(pts.shape[0], dtype=torch.long, device=device))

        # Pad to uniform length across batch
        max_pts = max(p.shape[0] for p in all_points)
        if max_pts == 0:
            return Prompt()

        points = torch.zeros(max_pts, B, 2, device=device, dtype=dtype)
        point_labels = torch.zeros(max_pts, B, dtype=torch.long, device=device)
        point_mask = torch.ones(B, max_pts, dtype=torch.bool, device=device)

        for b in range(B):
            n = all_points[b].shape[0]
            points[:n, b, :] = all_points[b]
            point_labels[:n, b] = all_labels[b]
            point_mask[b, :n] = False  # valid = not masked

        return Prompt(
            point_embeddings=points,
            point_mask=point_mask,
            point_labels=point_labels,
        )

"""Core inference engine for SAM3-RS without debug/analyzer code."""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import ndimage
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

        # Visual prototype bank (set via set_visual_prototype_bank, None by default)
        self.visual_prototype_bank = None
        self._inter_class_sim = None
        self._geo_only_classes = None
        self._geo_only_mode = False
        self._current_geo_only_rerun = False  # True when current pass is a geo-only rerun

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

            # Pre-compute per-class geometric prompts from visual prototypes
            proto_geo_prompts = None
            dinov3_max_responses = {}
            dinov3_geo_prompts_full = None  # full geo prompts for two-pass rerun
            use_two_pass = (self.visual_prototype_bank is not None
                            and self.geo_presence_sam3_thresh > 0)
            if self.visual_prototype_bank is not None:
                dinov3_geo_prompts_full, dinov3_max_responses = self._compute_dinov3_geo_prompts(image, batch_size=1)
                if not use_two_pass:
                    proto_geo_prompts = dinov3_geo_prompts_full
                # else: two-pass mode, proto_geo_prompts stays None, will inject conditionally

            # Process each prompt
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # 显存追踪：每个prompt开始
                self.memory_debugger.debug_print(f"\n[PROMPT {prompt_idx}/{self.num_prompts}] {prompt_word}")
                self.memory_debugger.log_cuda_memory(f"Prompt {prompt_idx} start: {prompt_word}")

                # Reset prompts for clean inference
                self.processor.reset_all_prompts(inference_state)

                # Get backbone_out for this image
                backbone_out = inference_state["backbone_out"]

                # Get class_id for this prompt
                class_id = self.prompts["indices"][prompt_idx]

                # Two-pass mode for dinov3_geo_point:
                # Pass 1: always run text-only first (never skip preemptively)
                # Pass 2 (geo_only_mode=False): presence < sam3_thresh + dinov3 confident → text + geo point
                # Pass 2 (geo_only_mode=True):  presence < sam3_thresh + dinov3 confident → geo point ONLY (no text)
                # No-skip fallback:              presence < sam3_thresh + dinov3 not confident → keep pass 1 result
                need_geo_rerun = False
                if use_two_pass and prompt_idx in dinov3_max_responses:
                    dinov3_resp = dinov3_max_responses[prompt_idx]
                    has_geo_prompt = (dinov3_geo_prompts_full is not None
                                      and prompt_idx in dinov3_geo_prompts_full)
                    if dinov3_resp >= self.geo_presence_threshold:
                        # This class has high DINOv3 confidence → candidate for enhancement
                        # but only if SAM3 is uncertain (will check after pass 1)
                        need_geo_rerun = True
                    elif getattr(self, 'geo_neg_independent', False) and has_geo_prompt:
                        # geo_neg_independent: even without confident positive points,
                        # if negative-only prompt was generated, still candidate for rerun
                        need_geo_rerun = True

                # Set geometric_prompt: dummy for pass 1, geo prompt for pass 2 / normal mode
                if not need_geo_rerun and proto_geo_prompts is not None and prompt_idx in proto_geo_prompts:
                    inference_state["geometric_prompt"] = proto_geo_prompts[prompt_idx]
                else:
                    inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

                output = None
                # Use pre-computed text features if available, otherwise real-time encoding
                if self.text_features_cache is not None and prompt_idx in self.text_features_cache:
                    cached_features = self.text_features_cache[prompt_idx]
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

                # Two-pass re-run: check presence score and conditionally re-run with geo points
                if need_geo_rerun and output is not None:
                    sam3_presence = output.get("presence_score", 1.0)
                    if sam3_presence < self.geo_presence_sam3_thresh:
                        # SAM3 uncertain + DINOv3 confident → re-run
                        if dinov3_geo_prompts_full is not None and prompt_idx in dinov3_geo_prompts_full:
                            self.memory_debugger.debug_print(
                                f"  [Two-pass] Rerunning: "
                                f"sam3_presence={sam3_presence:.3f} < {self.geo_presence_sam3_thresh}, "
                                f"dinov3_resp={dinov3_max_responses[prompt_idx]:.3f}, "
                                f"geo_only={self._geo_only_mode}")
                            self.processor.reset_all_prompts(inference_state)
                            inference_state["geometric_prompt"] = dinov3_geo_prompts_full[prompt_idx]

                            if self._geo_only_mode:
                                # Geo-only mode: skip text features, use only point prompt
                                self._current_geo_only_rerun = True
                                output = self.processor._forward_grounding(
                                    inference_state, encode_text=False)
                            else:
                                # Normal two-pass: text + geo point
                                if self.text_features_cache is not None and prompt_idx in self.text_features_cache:
                                    cached_features = self.text_features_cache[prompt_idx]
                                    backbone_out.update({
                                        "language_features": cached_features["language_features"],
                                        "language_mask": cached_features["language_mask"],
                                        "language_embeds": cached_features.get("language_embeds"),
                                    })
                                    output = self.processor._forward_grounding(inference_state)
                                else:
                                    output = self.processor.set_text_prompt(
                                        state=inference_state, prompt=prompt_word)
                    else:
                        need_geo_rerun = False  # SAM3 confident, no need to rerun
                elif self._geo_only_mode and use_two_pass:
                    # DINOv3 not confident → fall back to pass 1 text-only result (never skip)
                    dinov3_resp = dinov3_max_responses.get(prompt_idx, 0.0)
                    self.memory_debugger.debug_print(
                        f"  [Fallback] DINOv3 not confident (resp={dinov3_resp:.3f}), "
                        f"keeping pass 1 text-only result")

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
                    # semantic_logits: [1, 1, H_orig, W_orig] (4D tensor, H_orig=288 for 1008 input)
                    semantic_logits = output["semantic_seg"]

                    # Apply presence score FIRST, before resp injection.
                    # presence_score reflects SAM3's own confidence — should NOT scale the
                    # DINOv3 response map which is from an independent model.
                    # Order: (sem × presence_score) + resp, NOT (sem + resp) × presence_score
                    if self.config.use_presence_score and self.config.presence_score_mode == "before_fusion":
                        if self._current_geo_only_rerun:
                            dinov3_weight = dinov3_max_responses.get(prompt_idx, 0.0)
                            semantic_logits = semantic_logits * dinov3_weight
                        else:
                            presence_score = output.get("presence_score", 0)
                            semantic_logits = semantic_logits * presence_score

                    # Now interpolate to (h, w) — SAM3's own interpolation handles final upscale
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

                    # Fusion: take max of instance and semantic predictions
                    current_logits = torch.max(current_logits, semantic_logits)
                    semantic_logits_only[prompt_idx] = semantic_logits

                # 3. Presence score filtering (after fusion mode)
                if self.config.use_presence_score and self.config.presence_score_mode == "after_fusion":
                    if self._current_geo_only_rerun:
                        # Geo-only rerun: use DINOv3 response as weight
                        dinov3_weight = dinov3_max_responses.get(prompt_idx, 0.0)
                        current_logits = current_logits * dinov3_weight
                    else:
                        presence_score = output.get("presence_score", 0)
                        current_logits = current_logits * presence_score

                seg_logits[prompt_idx] = current_logits

                # ===== Clean up current prompt's intermediate variables =====
                del output
                if self.config.use_instance_head:
                    del inst_current
                if self.config.use_semantic_head:
                    del semantic_logits
                del current_logits
                self._current_geo_only_rerun = False

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

            # Pre-compute per-class geometric prompts from visual prototypes
            proto_geo_prompts = None
            if self.visual_prototype_bank is not None:
                # DINOv3: batch extract features, then per-image geo prompts, merge into batch
                precomputed = self._extract_dinov3_features_batch(images)
                # Collect per-image prompts first
                per_image_prompts = []  # list of dict or None
                for b in range(batch_size):
                    per_img, _ = self._compute_dinov3_geo_prompts(
                        images[b], batch_size=1, spatial_feats=precomputed[b])
                    per_image_prompts.append(per_img)
                # Merge into batch prompts
                merged = {}
                for b in range(batch_size):
                    if per_image_prompts[b] is None:
                        continue
                    for pidx, prompt in per_image_prompts[b].items():
                        if pidx not in merged:
                            merged[pidx] = [None] * batch_size
                        merged[pidx][b] = prompt
                # Build batch Prompts from collected single-image prompts
                batch_merged = {}
                for pidx, prompt_list in merged.items():
                    batch_merged[pidx] = self._build_batch_prompt(prompt_list, batch_size)
                proto_geo_prompts = batch_merged if batch_merged else None

            # 3. Loop through each Prompt (Decoder / Grounding)
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # ===== Use pre-computed text features =====
                # Clear historical text features
                for k in ["language_features", "language_mask", "language_embeds"]:
                    if k in backbone_out:
                        del backbone_out[k]

                # Get pre-computed text features from cache
                class_id = self.prompts["indices"][prompt_idx]

                if self.text_features_cache is not None and prompt_idx in self.text_features_cache:
                    cached_features = self.text_features_cache[prompt_idx]
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

    def set_visual_prototype_bank(self, prototypes: Optional[Dict],
                                   geo_topk: int = 10,
                                   geo_presence_threshold: float = 0.0,
                                   geo_skip_bg_idx: Optional[int] = None,
                                   geo_point_mode: str = "centroid",
                                   geo_centroid_thresh_ratio: float = 0.7,
                                   geo_centroid_min_area: int = 4,
                                   geo_topk_suppress_r: int = 0,
                                   geo_centroid_area_beta: float = 0.0,
                                   geo_centroid_method: str = "peak",
                                   dinov3_weights: Optional[str] = None,
                                   dinov3_input_size: int = 1008,
                                   geo_presence_sam3_thresh: float = 0.0,
                                   dinov3_layers: Optional[list] = None,
                                   geo_competition: str = "none",
                                   geo_neg_topk: int = 0,
                                   inter_class_sim: Optional[dict] = None,
                                   geo_only_classes: Optional[list] = None,
                                   geo_only_mode: bool = False,
                                   geo_neg_independent: bool = False):
        """Set the visual prototype bank for class-conditional visual guidance.

        Uses DINOv3 features to compute per-class response maps, then generates
        geometric point prompts (positive/negative) injected into SAM3's
        SequenceGeometryEncoder via cross-attention.

        Args:
            prototypes: Dict mapping class_id -> prototype tensor [1, 1, C].
            geo_topk: number of points to select as point prompts.
            geo_presence_threshold: minimum max-response to add geo prompt.
                0.0 = always add (default).
            geo_skip_bg_idx: skip geo prompt injection for this class_id.
            geo_point_mode: "centroid" (connected components + weighted center)
                or "topk" (global top-K highest response locations).
            geo_centroid_thresh_ratio: threshold ratio for centroid mode.
            geo_competition: class competition mode.
                - "none": no competition (default)
                - "mean"/"max"/"weighted": subtract other classes' responses
                - "exclusive": winner-take-all per pixel
            geo_neg_topk: number of negative points to add from other classes'
                response maps (default: 0, disabled). Each class's positive points
                are supplemented with top-K points from other classes' competition-modified
                response maps, labeled as negative (label=0) in the geometric prompt.
            inter_class_sim: pre-computed inter-class cosine similarity matrix.
            geo_only_classes: only inject geo prompts for these class IDs.
            geo_only_mode: if True, two-pass rerun uses geo point ONLY (no text).
            geo_neg_independent: if True, inject negative points even when positive
                points are suppressed by geo_presence_threshold. This allows leveraging
                the high-accuracy negative points (>94%) independently of positive point
                confidence, useful for classes with low positive-point accuracy but
                where negative points can still suppress cross-class confusion.
            dinov3_weights: path to DINOv3 SAT model directory.
            dinov3_input_size: input resolution for DINOv3.
        """
        self.visual_prototype_bank = prototypes
        self.geo_topk = geo_topk
        self.geo_presence_threshold = geo_presence_threshold
        self.geo_skip_bg_idx = geo_skip_bg_idx
        self.geo_point_mode = geo_point_mode
        self.geo_centroid_thresh_ratio = geo_centroid_thresh_ratio
        self.geo_centroid_min_area = geo_centroid_min_area
        self.geo_topk_suppress_r = geo_topk_suppress_r
        self.geo_centroid_area_beta = geo_centroid_area_beta
        self.geo_centroid_method = geo_centroid_method
        self.dinov3_input_size = dinov3_input_size
        self.geo_presence_sam3_thresh = geo_presence_sam3_thresh
        self.geo_competition = geo_competition
        self.geo_neg_topk = geo_neg_topk
        self._inter_class_sim = inter_class_sim
        self._geo_only_classes = set(geo_only_classes) if geo_only_classes else None
        self._geo_only_mode = geo_only_mode
        self.geo_neg_independent = geo_neg_independent

        # Load DINOv3 model if needed
        self._dinov3_model = None
        self._dinov3_n_reg = 0
        self._dinov3_transform = None
        self._dinov3_layers = dinov3_layers
        if dinov3_weights is not None:
            self._load_dinov3_model(dinov3_weights)
            layers_info = f", layers={dinov3_layers}" if dinov3_layers else ""
            print(f"  [core] DINOv3 model loaded, input_size={dinov3_input_size}{layers_info}")

    def _load_dinov3_model(self, weights_path):
        """Load DINOv3 ViT-L/16 SAT model for feature extraction."""
        from transformers import AutoModel
        import torchvision.transforms.v2 as v2
        print(f"  [core] Loading DINOv3 from {weights_path} ...")
        self._dinov3_model = AutoModel.from_pretrained(weights_path, local_files_only=True)
        self._dinov3_model = self._dinov3_model.to(self.device).eval()
        cfg = self._dinov3_model.config
        self._dinov3_n_reg = getattr(cfg, "num_register_tokens", 0)
        self._dinov3_patch_size = cfg.patch_size
        print(f"  [core] DINOv3: hidden_size={cfg.hidden_size}, patch_size={cfg.patch_size}, "
              f"register_tokens={self._dinov3_n_reg}")
        # DINOv3 SAT normalization
        self._dinov3_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(self.dinov3_input_size, self.dinov3_input_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143]),
        ])






    def _extract_dinov3_features_batch(self, images):
        """Extract DINOv3 patch tokens from a batch of PIL images.

        Returns:
            list of (spatial_feats [1, 1024*len(layers), H, W], patch_res) tuples
        """
        layers = self._dinov3_layers
        if layers is None:
            layers = [self._dinov3_model.config.num_hidden_layers]

        img_tensors = torch.stack([self._dinov3_transform(img) for img in images]).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = self._dinov3_model(img_tensors, output_hidden_states=True)

        n_skip = 1 + self._dinov3_n_reg
        patch_res = self.dinov3_input_size // self._dinov3_patch_size
        last_layer_idx = self._dinov3_model.config.num_hidden_layers

        results = []
        for b in range(len(images)):
            patch_feat_list = []
            for layer_idx in layers:
                if layer_idx == last_layer_idx:
                    layer_tokens = outputs.last_hidden_state[b:b+1]  # [1, N, C]
                else:
                    layer_tokens = outputs.hidden_states[layer_idx][b:b+1]
                p_tokens = layer_tokens[:, n_skip:, :]
                N, C = p_tokens.shape[1], p_tokens.shape[2]
                assert N == patch_res * patch_res
                spatial = p_tokens.reshape(1, patch_res, patch_res, C).permute(0, 3, 1, 2).float()
                patch_feat_list.append(spatial)
            spatial_feats = torch.cat(patch_feat_list, dim=1)
            results.append((spatial_feats, patch_res))
        return results

    def _extract_dinov3_features(self, image):
        """Extract DINOv3 patch tokens from a PIL image.

        Supports multi-layer fusion: concatenates patch tokens from specified layers.

        Returns:
            spatial_feats: [1, 1024*len(layers), patch_res, patch_res] float32
            patch_res: int
        """
        layers = self._dinov3_layers
        if layers is None:
            layers = [self._dinov3_model.config.num_hidden_layers]  # hidden_states has [0..24] for 24-layer model

        img_tensor = self._dinov3_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = self._dinov3_model(img_tensor, output_hidden_states=True)

        n_skip = 1 + self._dinov3_n_reg
        patch_res = self.dinov3_input_size // self._dinov3_patch_size

        last_layer_idx = self._dinov3_model.config.num_hidden_layers
        patch_feat_list = []
        for layer_idx in layers:
            if layer_idx == last_layer_idx:
                layer_tokens = outputs.last_hidden_state  # includes final LayerNorm
            else:
                layer_tokens = outputs.hidden_states[layer_idx]  # raw, no final LN
            p_tokens = layer_tokens[:, n_skip:, :]  # [1, H*W, 1024]
            B, N, C = p_tokens.shape
            assert N == patch_res * patch_res
            spatial = p_tokens.reshape(B, patch_res, patch_res, C).permute(0, 3, 1, 2).float()
            patch_feat_list.append(spatial)

        # Concatenate along channel dim: [1, 1024*len(layers), H, W]
        spatial_feats = torch.cat(patch_feat_list, dim=1)
        return spatial_feats, patch_res

    def _compute_dinov3_response_map(self, features, prototype):
        """Compute cosine similarity response map for DINOv3 features.

        Args:
            features: [1, 1024, H, W]
            prototype: [1, 1, 1024]

        Returns:
            response: [H, W] numpy array
        """
        B, C, H, W = features.shape
        feats_flat = features.flatten(2)  # [1, 1024, H*W]
        proto_flat = prototype.flatten(1).to(features.device, features.dtype)  # [1, 1024]
        feats_norm = F.normalize(feats_flat, p=2, dim=1)
        proto_norm = F.normalize(proto_flat, p=2, dim=1)
        resp = torch.bmm(proto_norm.unsqueeze(1), feats_norm)  # [1, 1, H*W]
        return resp.view(H, W).float().cpu().numpy()

    def _compute_dinov3_geo_prompts(self, image, batch_size=1, spatial_feats=None):
        """Compute per-class geo point prompts using DINOv3 features.

        Independent of SAM3 backbone — uses DINOv3 to extract features and
        compare with DINOv3 prototypes to generate response maps → geo points.

        Args:
            image: PIL Image (single image), ignored if spatial_feats is provided
            batch_size: always 1 for single-view
            spatial_feats: pre-computed (spatial_feats, patch_res) tuple to skip DINOv3 forward

        Returns:
            (prompt_geo, max_responses) tuple:
                prompt_geo: Dict[prompt_idx -> Prompt] or None
                max_responses: Dict[prompt_idx -> float] (max cosine similarity per class)
        """
        from sam3.model.geometry_encoders import Prompt

        if self._dinov3_model is None:
            return None, {}

        skip_bg = self.geo_skip_bg_idx
        topk = self.geo_topk
        presence_thresh = self.geo_presence_threshold
        point_mode = self.geo_point_mode
        competition = self.geo_competition

        # Extract DINOv3 features from the inference image
        if spatial_feats is not None:
            spatial_feats, patch_res = spatial_feats
        else:
            spatial_feats, patch_res = self._extract_dinov3_features(image)
        # spatial_feats: [1, 1024, patch_res, patch_res]
        H, W = patch_res, patch_res

        # ---- Phase 1: compute ALL raw response maps ----
        class_responses = {}  # class_id -> numpy [H, W]
        class_prompt_ids = {}  # class_id -> list[prompt_idx]
        neg_independent = getattr(self, 'geo_neg_independent', False)
        for prompt_idx in range(self.num_prompts):
            class_id = self.prompts["indices"][prompt_idx]
            if skip_bg is not None and class_id == skip_bg:
                continue
            # When geo_neg_independent: compute response for ALL classes (needed for
            # generating negative points for non-only-classes). Otherwise, only compute
            # for geo_only_classes.
            if (not neg_independent
                    and self._geo_only_classes is not None
                    and class_id not in self._geo_only_classes):
                continue
            if class_id not in self.visual_prototype_bank:
                continue
            if class_id not in class_responses:  # compute response once per class
                proto = self.visual_prototype_bank[class_id].to(self.device)
                if proto.shape[0] > 1:
                    proto = proto[0:1]
                response = self._compute_dinov3_response_map(spatial_feats, proto)
                class_responses[class_id] = response
            if class_id not in class_prompt_ids:
                class_prompt_ids[class_id] = []
            class_prompt_ids[class_id].append(prompt_idx)

        # ---- Phase 2: apply class competition ----
        if competition in ("mean", "max", "weighted", "exclusive") and len(class_responses) > 1:
            class_ids = list(class_responses.keys())

            # Pre-compute inter-class prototype cosine similarities for weighted mode
            if competition == "weighted":
                sim_matrix = {}
                # Prefer pre-computed similarity from bank file
                if self._inter_class_sim is not None:
                    for cid in class_ids:
                        sim_matrix[cid] = {
                            oid: max(v, 0.01)
                            for oid, v in self._inter_class_sim.get(cid, {}).items()
                            if oid in class_ids and oid != cid
                        }
                else:
                    proto_vecs = {}
                    for cid in class_ids:
                        p = self.visual_prototype_bank[cid].to(self.device)
                        if p.shape[0] > 1:
                            p = p[0:1]
                        proto_vecs[cid] = F.normalize(
                            p.flatten(1).float(), p=2, dim=1
                        )  # [1, C]
                    for cid in class_ids:
                        sim_matrix[cid] = {}
                        for oid in class_ids:
                            if cid == oid:
                                continue
                            sim = torch.mm(proto_vecs[cid], proto_vecs[oid].T).item()
                            sim_matrix[cid][oid] = max(sim, 0.01)

            # Save original responses before competition (avoid in-place contamination)
            orig_responses = {cid: class_responses[cid].copy() for cid in class_ids}

            for cid in class_ids:
                if competition == "exclusive":
                    # exclusive handled below after this loop
                    continue
                resp = class_responses[cid]
                others = np.stack(
                    [orig_responses[oid] for oid in class_ids if oid != cid], axis=0
                )  # [N-1, H, W]
                if competition == "mean":
                    resp = resp - others.mean(axis=0)
                elif competition == "max":
                    resp = resp - others.max(axis=0)
                else:  # weighted
                    other_ids = [oid for oid in class_ids if oid != cid]
                    weights = np.array([sim_matrix[cid][oid] for oid in other_ids])
                    weights = weights / weights.sum()
                    resp = resp - np.tensordot(weights, others, axes=([0], [0]))
                class_responses[cid] = resp

            # Exclusive (winner-take-all): each pixel only keeps the max-class response
            if competition == "exclusive":
                cids = list(class_responses.keys())
                stacked = np.stack([class_responses[cid] for cid in cids], axis=0)  # [N, H, W]
                winner = np.argmax(stacked, axis=0)  # [H, W]
                for i, cid in enumerate(cids):
                    mask = (winner == i)
                    class_responses[cid] = np.where(mask, class_responses[cid], 0.0)

        # ---- Phase 3: select points from (possibly modified) response maps ----
        prompt_geo = {}
        max_responses = {}
        neg_topk = self.geo_neg_topk
        for class_id, response in class_responses.items():
            prompt_idx_list = class_prompt_ids[class_id]
            max_response = response.max()
            # Store max_response for ALL prompt_idx of this class
            for pidx in prompt_idx_list:
                max_responses[pidx] = max_response

            # Determine if this class is eligible for positive point injection
            is_pos_class = (self._geo_only_classes is None
                            or class_id in self._geo_only_classes)
            positive_confident = max_response >= presence_thresh

            # Skip if: not eligible for positive, not confident, and neg_independent is off
            if not is_pos_class and not positive_confident and not neg_independent:
                continue

            # --- Positive points: only for pos-eligible classes with sufficient confidence ---
            resp_np = response  # numpy [H, W]
            pts = np.empty((0, 2), dtype=np.float64)
            if is_pos_class and positive_confident:
                if point_mode == "topk":
                    pts = self._select_points_topk(resp_np, H, W, topk)
                else:
                    pts = self._select_points_centroid(resp_np, H, W, topk)

            # --- Negative points from other classes ---
            neg_pts = np.empty((0, 2), dtype=np.float64)
            if neg_topk > 0 and len(class_responses) > 1:
                # Collect other classes' responses, find combined high-response points
                other_responses = []
                for oid, oresp in class_responses.items():
                    if oid != class_id:
                        other_responses.append(oresp)
                if other_responses:
                    # Merge other classes by taking element-wise max
                    others_merged = np.stack(other_responses, axis=0).max(axis=0)
                    neg_pts = self._select_points_topk(others_merged, H, W, neg_topk)

            n_pos = pts.shape[0]
            n_neg = neg_pts.shape[0]
            if n_pos == 0 and n_neg == 0:
                continue

            # Combine positive and negative points
            all_pts = np.concatenate([pts, neg_pts], axis=0) if n_neg > 0 else pts
            all_labels = np.concatenate([
                np.ones(n_pos, dtype=np.int64),
                np.zeros(n_neg, dtype=np.int64),
            ])

            pts_tensor = torch.as_tensor(all_pts, device=self.device, dtype=torch.float32)
            pts_tensor = pts_tensor.clamp(0, 1)
            n_pts = pts_tensor.shape[0]

            points = pts_tensor.unsqueeze(1)  # [n_pts, 1]
            point_labels = torch.as_tensor(all_labels, device=self.device, dtype=torch.long).unsqueeze(1)
            point_mask = torch.zeros(1, n_pts, dtype=torch.bool, device=self.device)

            geo_prompt = Prompt(
                point_embeddings=points,
                point_mask=point_mask,
                point_labels=point_labels,
            )
            # Assign the SAME geo prompt to ALL prompt_idx of this class
            for pidx in prompt_idx_list:
                prompt_geo[pidx] = geo_prompt

        return (prompt_geo if prompt_geo else None), max_responses

    @staticmethod
    def _build_batch_prompt(prompt_list: list, batch_size: int):
        """Build a batch Prompt from a list of single-image Prompts (may contain None).

        Args:
            prompt_list: list of length batch_size, each element is a Prompt or None
            batch_size: total batch size

        Returns:
            A Prompt with batch dimension, where missing images have masked-out points.
        """
        from sam3.model.geometry_encoders import Prompt

        # Find max number of points across images
        n_max = 0
        device = None
        dtype = None
        for p in prompt_list:
            if p is not None and p.point_embeddings is not None:
                n_max = max(n_max, p.point_embeddings.shape[0])
                device = p.point_embeddings.device
                dtype = p.point_embeddings.dtype

        if n_max == 0 or device is None:
            return None

        # All-masked batch prompt (no valid points for any image)
        pts = torch.zeros(n_max, batch_size, 2, device=device, dtype=dtype)
        labels = torch.zeros(n_max, batch_size, dtype=torch.long, device=device)
        mask = torch.ones(batch_size, n_max, dtype=torch.bool, device=device)  # True = masked (ignored)

        for b in range(batch_size):
            p = prompt_list[b]
            if p is not None and p.point_embeddings is not None and p.point_embeddings.shape[0] > 0:
                n = p.point_embeddings.shape[0]
                pts[:n, b, :] = p.point_embeddings[:, 0, :]
                labels[:n, b] = p.point_labels[:, 0]
                mask[b, :n] = p.point_mask[0, :]

        return Prompt(point_embeddings=pts, point_mask=mask, point_labels=labels)







    def _select_points_topk(self, resp_np, H, W, topk):
        """Select global top-K highest response locations with spatial suppression."""
        if topk <= 0:
            return np.empty((0, 2), dtype=np.float64)
        suppress_r = getattr(self, 'geo_topk_suppress_r', 0)
        if suppress_r <= 0 or topk <= 1:
            flat = resp_np.flatten()
            k = min(topk, len(flat))
            topk_idx = np.argpartition(flat, -k)[-k:]
            topk_idx = topk_idx[np.argsort(flat[topk_idx])[::-1]][:topk]
            y_idx = (topk_idx // W).astype(np.float64) / H
            x_idx = (topk_idx % W).astype(np.float64) / W
            return np.stack([x_idx, y_idx], axis=1)

        # Greedy topk with spatial suppression
        points = []
        working = resp_np.copy()
        for _ in range(topk):
            flat = working.flatten()
            if flat.max() <= 0:
                break
            idx = np.argmax(flat)
            py, px = idx // W, idx % W
            points.append([px / W, py / H])
            # Suppress neighborhood
            y_lo = max(0, py - suppress_r)
            y_hi = min(H, py + suppress_r + 1)
            x_lo = max(0, px - suppress_r)
            x_hi = min(W, px + suppress_r + 1)
            working[y_lo:y_hi, x_lo:x_hi] = -np.inf

        return np.array(points) if points else np.empty((0, 2))

    def _select_points_centroid(self, resp_np, H, W, topk):
        """Select points via connected-component analysis.

        Supports 'weighted' (response-weighted centroid), 'peak' (max response
        pixel within CC), and 'interior_peak' (response × distance-to-boundary) methods.
        """
        if topk <= 0:
            return np.empty((0, 2), dtype=np.float64)
        method = getattr(self, 'geo_centroid_method', 'peak')
        min_area = getattr(self, 'geo_centroid_min_area', 4)
        thresh_ratio = getattr(self, 'geo_centroid_thresh_ratio', 0.7)
        rmin, rmax = resp_np.min(), resp_np.max()
        threshold = rmin + thresh_ratio * (rmax - rmin)
        binary = (resp_np >= threshold).astype(np.int32)
        if binary.sum() == 0:
            return self._select_points_topk(resp_np, H, W, topk)
        labeled, num_components = ndimage.label(binary, structure=np.ones((3, 3)))
        if num_components == 0:
            return self._select_points_topk(resp_np, H, W, topk)

        def _pick_point(mask_2d, qscore):
            """Pick a point from a binary mask based on method."""
            if method == "interior_peak":
                dist = ndimage.distance_transform_edt(mask_2d)
                d_max = dist.max()
                dist_norm = dist / d_max if d_max > 0 else dist
                alpha = getattr(self, 'geo_centroid_interior_alpha', 0.5)
                score = resp_np * (1.0 + alpha * dist_norm)
                sy, sx = np.where(mask_2d)
                scores = score[mask_2d]
                idx = np.argmax(scores)
                return (sx[idx] / W, sy[idx] / H, qscore)
            elif method == "peak":
                vals = resp_np[mask_2d]
                idx = np.argmax(vals)
                sy, sx = np.where(mask_2d)
                return (sx[idx] / W, sy[idx] / H, qscore)
            else:
                w = resp_np * mask_2d
                tw = w.sum()
                if tw < 1e-8:
                    sy, sx = np.where(mask_2d)
                    return (sx.mean() / W, sy.mean() / H, qscore)
                else:
                    cx = (w * gxs).sum() / tw / W
                    cy = (w * gys).sum() / tw / H
                    return (cx, cy, qscore)

        all_points = []  # (cx_norm, cy_norm, qscore)
        gys, gxs = np.mgrid[0:H, 0:W].astype(np.float64)

        for cid in range(1, num_components + 1):
            comp_mask = (labeled == cid)
            area = comp_mask.sum()
            if area < min_area:
                continue

            ys_pos, xs_pos = np.where(comp_mask)
            y0, y1 = ys_pos.min(), ys_pos.max()
            x0, x1 = xs_pos.min(), xs_pos.max()
            bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1
            aspect = max(bbox_h, bbox_w) / (min(bbox_h, bbox_w) + 1e-8)
            fill_ratio = area / (bbox_h * bbox_w + 1e-8)

            weights = resp_np * comp_mask
            total_w = weights.sum()
            area_beta = getattr(self, 'geo_centroid_area_beta', 0.0)
            mean_resp = total_w / area if area > 0 else 0.0
            if area_beta > 0:
                # Quality-aware ranking: mean_response * area^beta
                comp_qscore = (mean_resp * (area ** area_beta)) if total_w > 1e-8 else (area ** area_beta)
            else:
                comp_qscore = total_w if total_w > 1e-8 else float(area)

            # Elongated & sparse → linear feature: subdivide along long axis
            if aspect > 3.0 and fill_ratio < 0.35 and max(bbox_h, bbox_w) > 8:
                n_sub = min(max(int(max(bbox_h, bbox_w) / 5), 2), 5)
                sub_qscore = comp_qscore / n_sub
                if bbox_h >= bbox_w:
                    edges = np.linspace(y0, y1 + 1, n_sub + 1).astype(int)
                    for s in range(n_sub):
                        lm = (ys_pos >= edges[s]) & (ys_pos < edges[s + 1])
                        if lm.sum() < 1:
                            continue
                        sm2d = np.zeros((H, W), dtype=bool)
                        sm2d[ys_pos[lm], xs_pos[lm]] = True
                        all_points.append(_pick_point(sm2d, sub_qscore))
                else:
                    edges = np.linspace(x0, x1 + 1, n_sub + 1).astype(int)
                    for s in range(n_sub):
                        lm = (xs_pos >= edges[s]) & (xs_pos < edges[s + 1])
                        if lm.sum() < 1:
                            continue
                        sm2d = np.zeros((H, W), dtype=bool)
                        sm2d[ys_pos[lm], xs_pos[lm]] = True
                        all_points.append(_pick_point(sm2d, sub_qscore))
            else:
                # Compact component
                all_points.append(_pick_point(comp_mask, comp_qscore))

        if not all_points:
            return self._select_points_topk(resp_np, H, W, topk)

        all_points.sort(key=lambda p: p[2], reverse=True)
        pts = np.array([[p[0], p[1]] for p in all_points[:topk]])

        # Pad with topk on suppressed map
        if len(all_points) < topk:
            remaining = topk - len(all_points)
            used_mask = np.zeros((H, W), dtype=bool)
            for i in range(len(pts)):
                px = np.clip(int(pts[i, 0] * W), 0, W - 1)
                py = np.clip(int(pts[i, 1] * H), 0, H - 1)
                r = max(1, min(H, W) // 20)
                y_lo, y_hi = max(0, py - r), min(H, py + r + 1)
                x_lo, x_hi = max(0, px - r), min(W, px + r + 1)
                used_mask[y_lo:y_hi, x_lo:x_hi] = True
            suppressed = resp_np.copy()
            suppressed[used_mask] = -np.inf
            flat = suppressed.flatten()
            n_need = min(remaining, int((flat > -np.inf).sum()))
            if n_need > 0:
                topk_idx = np.argpartition(flat, -n_need)[-n_need:]
                topk_idx = topk_idx[np.argsort(flat[topk_idx])[::-1]]
                y_idx = (topk_idx // W).astype(np.float64) / H
                x_idx = (topk_idx % W).astype(np.float64) / W
                extra = np.stack([x_idx, y_idx], axis=1)
                pts = np.concatenate([pts, extra], axis=0)

        return pts

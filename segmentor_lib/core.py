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
            # Store avg_embeddings if available (from semantic enhancement with avg)
            self.avg_embeddings = prompts.get("avg_embeddings", None)
            if self.avg_embeddings:
                print(f"✓ Using pre-computed average embeddings for {len(self.avg_embeddings)} classes")
        else:
            self.avg_embeddings = None

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

                # Ensure geometric_prompt exists (needed for _forward_grounding)
                if "geometric_prompt" not in inference_state:
                    inference_state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

                output = None
                # Priority 1: avg_embeddings (semantic enhancement with average)
                if self.avg_embeddings and class_id in self.avg_embeddings:
                    avg_emb = self.avg_embeddings[class_id]
                    backbone_out.update({
                        "language_features": avg_emb["language_features"],
                        "language_mask": avg_emb["language_mask"],
                        "language_embeds": avg_emb["language_embeds"],
                    })
                    # Run grounding inference
                    self.memory_debugger.log_cuda_memory(f"  [Before _forward_grounding] using avg_embeddings")
                    output = self.processor._forward_grounding(inference_state)
                    self.memory_debugger.log_cuda_memory(f"  [After _forward_grounding] using avg_embeddings")
                    if output and isinstance(output, dict):
                        for k, v in output.items():
                            if isinstance(v, torch.Tensor):
                                self.memory_debugger.log_tensor_memory(f"  output.{k}", v)
                # Priority 2: text_features_cache (standard pre-computation)
                elif self.text_features_cache is not None and prompt_idx in self.text_features_cache:
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

            # 3. Loop through each Prompt (Decoder / Grounding)
            for prompt_idx, prompt_word in enumerate(self.prompts["names"]):
                # ===== Use pre-computed text features =====
                # Clear historical text features
                for k in ["language_features", "language_mask", "language_embeds"]:
                    if k in backbone_out:
                        del backbone_out[k]

                # Get pre-computed text features from cache or avg_embeddings
                class_id = self.prompts["indices"][prompt_idx]

                # Priority 1: avg_embeddings (semantic enhancement with average)
                if self.avg_embeddings and class_id in self.avg_embeddings:
                    avg_emb = self.avg_embeddings[class_id]
                    backbone_out.update({
                        "language_features": avg_emb["language_features"],
                        "language_mask": avg_emb["language_mask"],
                        "language_embeds": avg_emb["language_embeds"],
                    })
                # Priority 2: text_features_cache (standard pre-computation)
                elif self.text_features_cache is not None and prompt_idx in self.text_features_cache:
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

                # Decoding stage (Grounding) - process Batch
                outputs = self.processor.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_stage,
                    geometric_prompt=dummy_geometric,
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
                del weighted_logits, inst_current, sem_current
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

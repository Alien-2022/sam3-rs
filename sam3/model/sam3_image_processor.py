# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import PIL
import torch

from sam3.model import box_ops

from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            # if image is a PIL image, get the height and width directly
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            # if image is a tensor or numpy array, get the height and width from the shape
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        # convert all kinds of input formats into standardized PyTorch tensor
        image = v2.functional.to_image(image).to(self.device)
        # apply transforms to the image and add batch dimension [3, 1008, 1008]->[1, 3, 1008, 1008]
        image = self.transform(image).unsqueeze(0)
        state["original_height"] = height
        state["original_width"] = width

        state["backbone_out"] = self.model.backbone.forward_image(image)
        # forward through the SAM3VLBackbone(vl_combiner.py), the shape of output:
        # backbone_out = {
        #     "vision_features": sam3_src, # [1, 256, 72, 72]
        #     "vision_pos_enc": sam3_pos, # [[1, 256, 288, 288],[1, 256, 144, 144],[1, 256, 72, 72]]
        #     "backbone_fpn": sam3_features, # [[1, 256, 288, 288],[1, 256, 144, 144],[1, 256, 72, 72]]
        #     "sam2_backbone_out": sam2_output, # None
        # }
        # The shape of vision_pos_enc and backbone_fpn is determined by the scalp parameter in _create_vl_backbone (model_builder.py)
        # - scalp=1: discards the lowest resolution features, keeping 3 levels (shapes: [288x288], [144x144], [72x72])
        # - scalp=0: keeps all 4 feature levels (shapes: [288x288], [144x144], [72x72], [36x36])

        # inst_interactive_predictor enables Interactive Instance Segmentation (like SAM1/SAM2)
        # by default inst_interactive_predictor is None, so the model only supports zero-shot segmentation based on text prompts
        inst_interactivity_en = self.model.inst_interactive_predictor is not None

        # sam2_backbone_out is None by default because:
        # 1. When sam2_features is None or sam2_pos is None, sam2_output remains None (see necks.py:110)
        # 2. This is typical for SAM3 models that don't require SAM2's backbone features
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        # the shape of state:
        # state = {
        #     "original_heights": height,
        #     "original_widths": width,
        #     "backbone_out": {
        #         "vision_features": [1, 256, 72, 72],
        #         "vision_pos_enc": [
        #             [1, 256, 288, 288],
        #             [1, 256, 144, 144],
        #             [1, 256, 72, 72],
        #         ],
        #         "backbone_fpn": [
        #             [1, 256, 288, 288],
        #             [1, 256, 144, 144],
        #             [1, 256, 72, 72],
        #         ],
        #         "sam2_backbone_out": None,
        #     },
        # }
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        # Ensure that vision features have been extracted before setting the text prompt
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # the shape of text_outputs:
        # text_outputs = {
        #     "language_features": [32, 1, 256],
        #     "language_mask": [1, 32],
        #     "language_embeds": [32, 1, 1024],
        # }

        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        # the shape of state after update:
        # state = {
        #     "original_heights": height,
        #     "original_widths": width,
        #     "backbone_out": {
        #         "vision_features": [1, 256, 72, 72],
        #         "vision_pos_enc": [
        #             [1, 256, 288, 288],
        #             [1, 256, 144, 144],
        #             [1, 256, 72, 72],
        #         ],
        #         "backbone_fpn": [
        #             [1, 256, 288, 288],
        #             [1, 256, 144, 144],
        #             [1, 256, 72, 72],
        #         ],
        #         "sam2_backbone_out": None,
        #         "language_features": [32, 1, 256],
        #         "language_mask": [1, 32],
        #         "language_embeds": [32, 1, 1024],
        #     },
        # }

        # geometric_prompt represents user-provided geometric prompts (besides text prompt)
        # including: boxes (bounding boxes), points (click points), masks (mask prompts)
        # If not provided, create a dummy prompt with empty geometric inputs
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
            # _get_dummy_prompt() returns a Prompt object with:
            #   - box_embeddings: torch.zeros(0, 1, 4)  # 0 boxes
            #   - box_mask: torch.zeros(1, 0, dtype=bool)  # empty attention mask
            #   - point_embeddings: None  # 0 points
            #   - mask_embeddings: None  # 0 masks
            # This indicates the user is using only text prompts, no geometric prompts

        # add new keys(predict masks, boxes, scores) to state and return, new keys:
        #   - "masks_logits": [inst_num, 1, original_height, original_width]
        #   - "masks": [inst_num, 1, original_height, original_width]
        #   - "boxes": [inst_num, 4]
        #   - "scores": [inst_num]
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):

        # Perform grounding forward pass to generate predictions
        # Inputs:
        #   - backbone_out: Combined vision and text features
        #   - find_input: Current processing stage
        #   - geometric_prompt: User-provided geometric prompts (boxes/points/masks)
        #   - find_target: Used for supervised training, not used in inference
        # Outputs will contain:
        #   - pred_boxes: Predicted bounding boxes in cxcywh format
        #   - pred_logits: Classification logits
        #   - pred_masks: Predicted segmentation masks
        #   - presence_logit_dec: Presence confidence score
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        # out_masks: Each queries outputs the pixel probability of its detected target.
        out_masks = outputs["pred_masks"]
        # out_probs: Existence probability of each instance detected by 200 queries
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        # update out_probs with presence_score
        out_probs = (out_probs * presence_score).squeeze(-1) # [B, 200, 1] -> [B, 200]

        keep = out_probs > self.confidence_threshold # [B, 200]

        # Filter probabilities, masks and boxes of 200 queries based on confidence threshold
        out_probs = out_probs[keep] # [B, 200] -> [B, n]  
        out_masks = out_masks[keep] # [B, 200, 288, 288] -> [n, 288, 288]
        out_bbox = out_bbox[keep] # [1, n, 4]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        # [n, 288, 288] -> [n, 1, img_h, img_w]
        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        # masks_logits is the  prediction(0~1) for each pixel, masks is the binary masks with threshold 0.5
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

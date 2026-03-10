"""Post-processing module for SAM3-RS inference results."""

import torch
import torch.nn.functional as F
from typing import Dict


def fuse_prompts_to_classes(seg_logits: torch.Tensor, query_indices: torch.Tensor,
                         num_classes: int, num_prompts: int) -> torch.Tensor:
    """
    Map prompt-level logits to class-level logits by fusing synonyms.

    Args:
        seg_logits: [num_prompts, H, W] or [B, num_prompts, H, W]
        query_indices: [num_prompts] tensor mapping prompts to class IDs
        num_classes: Number of classes
        num_prompts: Number of prompts

    Returns:
        [num_classes, H, W] or [B, num_classes, H, W] fused logits
    """
    # Handle both single image and batch
    batch_mode = seg_logits.dim() == 4
    if batch_mode:
        # seg_logits: [B, num_prompts, H, W]
        seg_logits = seg_logits.unsqueeze(1)  # [B, 1, num_prompts, H, W]
    else:
        # seg_logits: [num_prompts, H, W] -> [1, num_prompts, H, W]
        seg_logits = seg_logits.unsqueeze(0)

    # Convert query indices to one-hot vectors
    # query_idx=[0, 1, 1, 2] -> one_hot -> [[1,0,0], [0,1,0], [0,1,0], [0,0,1]]
    cls_index = F.one_hot(query_indices, num_classes=num_classes)

    # Transpose to [num_classes, num_prompts], reshape to [num_classes, num_prompts, 1, 1]
    cls_index = cls_index.T.view(num_classes, num_prompts, 1, 1)

    # Broadcast multiply: [B, 1, num_prompts, H, W] * [1, num_classes, num_prompts, 1, 1]
    # Then take max over prompts dimension to get [B, num_classes, H, W]
    if batch_mode:
        seg_logits = (seg_logits * cls_index.unsqueeze(0)).max(2)[0]
    else:
        seg_logits = (seg_logits * cls_index).max(1)[0]

    return seg_logits


def logits_to_pred(logits: torch.Tensor, use_prompted_background: bool,
                prob_threshold: float, bg_idx: int) -> torch.Tensor:
    """
    Convert logits to class predictions with background handling.

    Args:
        logits: [num_classes, H, W] or [B, num_classes, H, W]
        use_prompted_background: Whether background is in prompts
        prob_threshold: Threshold for low-confidence suppression
        bg_idx: Background class index

    Returns:
        [H, W] or [B, H, W] class predictions
    """
    batch_mode = logits.dim() == 4

    def _logits_to_pred_single(logit_single: torch.Tensor) -> torch.Tensor:
        # Case 1: Background is NOT in prompts - inject a zero-logit background channel
        if not use_prompted_background:
            bg_pad = torch.zeros(
                (1, *logit_single.shape[1:]), device=logit_single.device, dtype=logit_single.dtype
            )
            logits_for_argmax = torch.cat([bg_pad, logit_single], dim=0)
            pred = torch.argmax(logits_for_argmax, dim=0)

            # Apply prob_threshold: assign pixels with max logit < threshold to background
            max_vals = logit_single.max(0)[0]
            pred[max_vals < prob_threshold] = bg_idx
        else:
            # Case 2: Background IS in prompts - no need to inject background channel
            pred = torch.argmax(logit_single, dim=0)
            # Still apply prob_threshold filtering
            max_vals = logit_single.max(0)[0]
            pred[max_vals < prob_threshold] = bg_idx

        return pred

    if batch_mode:
        # Process each image in batch
        preds = [_logits_to_pred_single(logits[b]) for b in range(logits.shape[0])]
        return torch.stack(preds, dim=0)
    else:
        return _logits_to_pred_single(logits)


def resize_logits(logits: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Resize logits to target shape using bilinear interpolation.

    Args:
        logits: Tensor to resize [C, H, W] or [B, C, H, W]
        target_shape: (H, W) target size

    Returns:
        Resized logits with same batch dimension
    """
    if logits.shape[-2:] == target_shape:
        return logits

    squeeze = logits.dim() == 3
    if squeeze:
        logits = logits.unsqueeze(0)  # [1, C, H, W]

    logits = F.interpolate(
        logits,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )

    if squeeze:
        logits = logits.squeeze(0)

    return logits

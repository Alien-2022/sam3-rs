"""Concept Bank for SAM3-RS open vocabulary semantic segmentation.

Adapted from ConceptBank (Pei et al., 2026) with remote-sensing-specific modifications:
- No DDP dependency (single GPU)
- Direct integration with SAM3-RS text_features_cache
- Support for both GT masks and pseudo-labels

Pipeline:
  Stage I: Prototype Collection - extract visual prototypes from GT/pseudo-label crops
  Stage II: Representative Mining - select top-K representative crops per class
  Stage III: Prompt Scoring & Fusion - score candidate prompts, fuse with temperature-scaled softmax
"""

import os
import gc
import time
import heapq
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F


# =========================================================
# Crop View Generation
# =========================================================

def square_box_from_mask(
    train_ids: np.ndarray,
    class_id: int,
    pad_ratio: float = 0.05,
    min_size: int = 128,
    max_size: int = 1024,
    rng: np.random.RandomState = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute a square bounding box around class_id pixels in the mask.

    If the class region is larger than max_size, randomly samples a max_size
    sub-region centered on class pixels to avoid huge crops (especially for
    background/large classes in remote sensing images).
    """
    if rng is None:
        rng = np.random.RandomState()

    ys, xs = np.where(train_ids == class_id)
    if len(xs) == 0:
        return None

    # For large regions, sample a random sub-region instead of using the full bbox
    region_size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    if region_size > max_size:
        idx = rng.randint(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        half = max_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(train_ids.shape[1] - 1, x1 + max_size - 1)
        y2 = min(train_ids.shape[0] - 1, y1 + max_size - 1)
        # Re-center if clipped
        x1 = max(0, x2 - max_size + 1)
        y1 = max(0, y2 - max_size + 1)
        return x1, y1, x2, y2

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    pad = int(max(w, h) * pad_ratio)

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(train_ids.shape[1] - 1, x2 + pad)
    y2p = min(train_ids.shape[0] - 1, y2 + pad)

    cx, cy = (x1p + x2p) // 2, (y1p + y2p) // 2
    half = max((x2p - x1p + 1), (y2p - y1p + 1), min_size) // 2

    x1p, x2p = max(0, cx - half), min(train_ids.shape[1] - 1, cx + half)
    y1p, y2p = max(0, cy - half), min(train_ids.shape[0] - 1, cy + half)
    return x1p, y1p, x2p, y2p


def make_crop_views_from_np(
    img_np: np.ndarray,
    train_ids: np.ndarray,
    class_id: int,
    pad_ratio: float = 0.05,
    min_size: int = 128,
    max_size: int = 1024,
    use_context: bool = True,
    use_masked: bool = False,
    rng: np.random.RandomState = None,
) -> Tuple[Optional[List[Image.Image]], Optional[np.ndarray]]:
    """Create crop views from image for a given class.

    Returns:
        views: list of PIL images (context and/or masked), resized to max_size if needed
        crop_mask: binary mask of the target class within the crop (at original crop resolution)
    """
    box = square_box_from_mask(train_ids, class_id, pad_ratio, min_size, max_size, rng)
    if box is None:
        return None, None

    x1, y1, x2, y2 = box
    crop_np = img_np[y1:y2 + 1, x1:x2 + 1].copy()
    m = (train_ids[y1:y2 + 1, x1:x2 + 1] == class_id).astype(np.uint8)

    views: List[Image.Image] = []
    ctx_view = Image.fromarray(crop_np)
    if use_context:
        views.append(ctx_view)
    if use_masked:
        bg = np.full_like(crop_np, 127, dtype=np.uint8)
        masked = np.where(m[..., None].astype(bool), crop_np, bg)
        views.append(Image.fromarray(masked))

    if not views:
        views.append(ctx_view)
    return views, m


# =========================================================
# Visual Embedding Extraction
# =========================================================

@torch.inference_mode()
def extract_sam3_image_embedding(backbone_out: Dict[str, Any]) -> torch.Tensor:
    """Extract L2-normalized visual embedding from SAM3 backbone output.

    Handles various tensor shapes:
    - [1, C, h, w] -> mean over h, w -> [C] -> L2 norm
    - [B, C, h, w] -> mean over all dims -> [C] -> L2 norm
    - [N, D] -> mean -> [D] -> L2 norm
    """
    preferred_keys = [
        "image_embedding", "image_embed", "img_embedding", "img_embed",
        "vision_embedding", "vision_embed", "image_features", "vision_features",
        "backbone_features", "features", "feat", "x",
    ]

    def _pick_tensor():
        for k in preferred_keys:
            if k in backbone_out and torch.is_tensor(backbone_out[k]):
                return backbone_out[k]
        # Fallback: find any large float tensor
        best_score, best_t = -1, None
        for k, v in backbone_out.items():
            if torch.is_tensor(v) and v.dtype in (torch.float16, torch.float32, torch.bfloat16):
                score = v.numel()
                if score > best_score:
                    best_score = score
                    best_t = v
        return best_t

    t = _pick_tensor()
    if t is None:
        return torch.zeros(1, dtype=torch.float32)

    if t.ndim == 4:
        if t.size(0) == 1:
            t = t[0]
        if t.ndim == 3:
            t = t.mean(dim=(1, 2))
        else:
            t = t.mean(dim=(0, 2, 3))
    elif t.ndim >= 2:
        t = t.reshape(-1, t.shape[-1]).mean(dim=0)
    else:
        t = t.flatten()

    emb = t.float()
    return emb / emb.norm(p=2).clamp_min(1e-6)


# =========================================================
# Tensor Layout Utilities (from official Concept Bank)
# =========================================================
# SAM3's text encoder output layout is [T, B, D], not [B, T, D].
# These helpers convert between the two for consistent fusion.

def _find_dim_of_size(shape: torch.Size, size: int) -> Optional[int]:
    for i, s in enumerate(shape):
        if s == size:
            return i
    return None


def _align_3d_to_BTD(x: torch.Tensor, B: int) -> Tuple[torch.Tensor, dict]:
    """Reorder a 3D tensor to [B, T, D] layout, recording the original layout."""
    assert x.ndim == 3
    bd = _find_dim_of_size(x.shape, B)
    if bd is None:
        raise RuntimeError(f"Cannot find batch dim B={B} in {tuple(x.shape)}")
    rem = [0, 1, 2]
    rem.remove(bd)
    d_dim = rem[0] if x.shape[rem[0]] >= x.shape[rem[1]] else rem[1]
    t_dim = rem[1] if d_dim == rem[0] else rem[0]
    x_btd = x.permute(bd, t_dim, d_dim).contiguous()
    layout = {"raw_shape": tuple(x.shape), "batch_dim": bd, "token_dim": t_dim, "d_dim": d_dim}
    return x_btd, layout


def _align_mask_to_BT(x: torch.Tensor, B: int) -> Tuple[torch.Tensor, dict]:
    """Reorder a 2D mask to [B, T] layout, recording the original layout."""
    x = torch.as_tensor(x).squeeze()
    if x.ndim == 1:
        if B != 1:
            raise RuntimeError(f"language_mask is 1D {tuple(x.shape)} but B={B} != 1")
        return x.view(1, -1).contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": 0, "token_dim": 1}

    if x.ndim != 2:
        raise RuntimeError(f"language_mask must be 1D or 2D after squeeze, got {tuple(x.shape)}")

    bd = _find_dim_of_size(x.shape, B)
    if bd is None:
        if B == 1:
            if x.shape[0] == 1:
                return x.contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": 0, "token_dim": 1}
            if x.shape[1] == 1:
                return x.t().contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": 0, "token_dim": 1}
        raise RuntimeError(f"Cannot find batch dim B={B} in mask {tuple(x.shape)}")

    t_dim = 1 - bd
    return x.permute(bd, t_dim).contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": bd, "token_dim": t_dim}


def _align_language_outputs(out: Dict[str, Any], B: int):
    """Align language encoder outputs to [B, T, D] / [B, T] layout."""
    lf_btd, lf_layout = _align_3d_to_BTD(out["language_features"], B)
    pm_bt, pm_layout = _align_mask_to_BT(out["language_mask"], B)
    le_btd, le_layout = _align_3d_to_BTD(out["language_embeds"], B)
    layout = {"language_features": lf_layout, "language_mask": pm_layout, "language_embeds": le_layout}
    return lf_btd, pm_bt, le_btd, layout


def to_raw_from_BTD(x_btd: torch.Tensor, lay: dict) -> torch.Tensor:
    """Convert [B, T, D] back to the original tensor layout."""
    bd, td, dd = lay["batch_dim"], lay["token_dim"], lay["d_dim"]
    src = [0, 0, 0]
    src[bd], src[td], src[dd] = 0, 1, 2
    return x_btd.permute(*src).contiguous()


def mask_to_raw_from_BT(x_bt: torch.Tensor, lay: dict) -> torch.Tensor:
    """Convert [B, T] mask back to the original layout."""
    bd = lay["batch_dim"]
    src = [0, 0]
    src[bd], src[1 - bd] = 0, 1
    return x_bt.permute(*src).contiguous()


# =========================================================
# Token Fusion (from official Concept Bank)
# =========================================================

def fuse_tokens(
    selected: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse text token sequences with weighted average.

    All inputs must have the same token length T (achieved by batch-encoding).

    Args:
        selected: list of (language_features[T, Df], language_mask[T], language_embeds[T, De], weight)

    Returns:
        fused_lf: [T, Df]
        fused_pm: [T] bool (True = padding position)
        fused_le: [T, De]
    """
    assert len(selected) >= 1
    T = selected[0][0].shape[0]
    w = torch.tensor([s[3] for s in selected], dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-6)

    lf = torch.stack([s[0].float() for s in selected], dim=0)
    pm = torch.stack([s[1].bool() for s in selected], dim=0)
    le = torch.stack([s[2].float() for s in selected], dim=0)

    valid = (~pm).float()
    denom = (w.view(-1, 1) * valid).sum(dim=0)
    valid_f = denom > 1e-6
    pm_f = ~valid_f
    denom_safe = denom.clamp_min(1e-6).view(1, T, 1)

    lf_num = (w.view(-1, 1, 1) * lf * valid.unsqueeze(-1)).sum(dim=0, keepdim=True)
    le_num = (w.view(-1, 1, 1) * le * valid.unsqueeze(-1)).sum(dim=0, keepdim=True)

    return (lf_num / denom_safe).squeeze(0), pm_f, (le_num / denom_safe).squeeze(0)


# =========================================================
# Probability Map Construction
# =========================================================

@torch.inference_mode()
def build_prob_map(
    outputs: dict,
    H: int,
    W: int,
    confidence_threshold: float = 0.0,
    topk_inst: int = 0,
    use_sem: bool = True,
    use_presence: bool = True,
) -> torch.Tensor:
    """Build per-class probability map from SAM3 grounding output.

    Returns:
        prob_map: [B, H, W]
    """
    from sam3.model.data_misc import interpolate

    pres = outputs["presence_logit_dec"].sigmoid().view(-1)  # [B]
    pl = outputs["pred_logits"]
    if pl.ndim == 3 and pl.size(-1) == 1:
        pl = pl.squeeze(-1)
    scores = pl.sigmoid()  # [B, N]

    pm = outputs["pred_masks"]
    if pm.ndim == 5 and pm.size(2) == 1:
        pm = pm.squeeze(2)  # [B, N, hf, wf]

    sem_up = None
    if use_sem and ("semantic_seg" in outputs):
        sem = outputs["semantic_seg"]  # [B, 1, hf, wf]
        sem_up = interpolate(sem, (H, W), mode="bilinear", align_corners=False).sigmoid().squeeze(1)

    B = scores.shape[0]
    out_maps = []
    for b in range(B):
        s = scores[b]
        masks_b = pm[b]

        if topk_inst > 0 and s.numel() > topk_inst:
            val, idx = torch.topk(s, k=topk_inst, largest=True)
            s = val
            masks_b = masks_b[idx]

        if use_presence:
            s = s * pres[b]

        keep = s > confidence_threshold
        inst_map = torch.zeros((H, W), device=masks_b.device, dtype=torch.float16)

        if keep.any():
            masks_k = masks_b[keep]
            scores_k = s[keep]
            masks_up = interpolate(
                masks_k.unsqueeze(1), (H, W), mode="bilinear", align_corners=False
            ).sigmoid().squeeze(1)
            inst_map = (masks_up * scores_k.view(-1, 1, 1)).amax(dim=0)

        if sem_up is not None:
            sem_val = sem_up[b] * pres[b]
            inst_map = torch.max(inst_map, sem_val)

        out_maps.append(inst_map)
    return torch.stack(out_maps, dim=0)


# =========================================================
# Dice Score Computation
# =========================================================

def soft_dice_score(
    prob_map: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """Compute soft Dice score between probability map and GT mask.

    Args:
        prob_map: [H, W] or [B, H, W] float
        gt_mask: [H, W] binary

    Returns:
        dice: float
    """
    if prob_map.ndim == 3:
        prob_map = prob_map.float().mean(dim=0)  # average over batch
    prob_map = prob_map.float()
    gt_mask = gt_mask.float()

    gt_sum = gt_mask.sum().clamp_min(1e-6)
    inter = (prob_map * gt_mask).sum()
    denom = prob_map.sum() + gt_sum + 1e-6
    return float((2 * inter + 1e-6) / denom)


# =========================================================
# LRU Embedding Cache
# =========================================================

class LRUEmbCache:
    """Simple LRU cache for crop embeddings."""

    def __init__(self, max_items: int = 8192):
        self.max_items = int(max_items)
        self.od: OrderedDict = OrderedDict()

    def get(self, key):
        v = self.od.get(key, None)
        if v is not None:
            self.od.move_to_end(key)
        return v

    def put(self, key, value):
        if self.max_items <= 0:
            return
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.max_items:
            self.od.popitem(last=False)


# =========================================================
# Concept Bank Builder
# =========================================================

class ConceptBankBuilder:
    """Build a concept bank from calibration images with GT masks or pseudo-labels.

    The concept bank replaces per-prompt text embeddings with fused embeddings
    that are visually grounded in the target domain.

    Usage:
        builder = ConceptBankBuilder(processor, device, prompts)
        text_features_cache = builder.build(dataset, num_calib=24)
        segmentor.engine.text_features_cache = text_features_cache
    """

    def __init__(
        self,
        processor,
        device: torch.device,
        prompts: Dict,
        engine=None,
        segmentor=None,
        # Stage I hyperparameters
        pad_ratio: float = 0.05,
        min_crop_size: int = 128,
        max_crop_size: int = 1024,
        max_per_class_pass1: int = 10,
        # Stage II hyperparameters
        top_k_per_class: int = 10,
        pass2_max_epochs: int = 3,
        pass2_emb_cache_size: int = 8192,
        # Stage III hyperparameters
        tau_w: float = 0.15,
        cand_topk: int = 999,
        fuse_topk: int = 999,
        confidence_threshold: float = 0.1,
        topk_inst: int = 100,
        # View settings
        use_context_view: bool = True,
        use_masked_view: bool = False,
        view_weight_context: float = 0.9,
        view_weight_masked: float = 0.1,
        # Misc
        ignore_index: int = 255,
        amp_enabled: bool = True,
    ):
        self.processor = processor
        self.engine = engine
        self.segmentor = segmentor
        self.device = device
        self.prompts = prompts
        self.num_classes = max(prompts["indices"]) + 1
        self.num_prompts = len(prompts["names"])

        # Hyperparameters
        self.pad_ratio = pad_ratio
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.max_per_class_pass1 = max_per_class_pass1
        self.top_k_per_class = top_k_per_class
        self.pass2_max_epochs = pass2_max_epochs
        self.pass2_emb_cache_size = pass2_emb_cache_size
        self.tau_w = tau_w
        self.cand_topk = cand_topk
        self.fuse_topk = fuse_topk
        self.confidence_threshold = confidence_threshold
        self.topk_inst = topk_inst
        self.use_context_view = use_context_view
        self.use_masked_view = use_masked_view
        self.view_weight_context = view_weight_context
        self.view_weight_masked = view_weight_masked
        self.ignore_index = ignore_index
        self.amp_enabled = amp_enabled
        self.rng = np.random.RandomState(42)

        # Build class -> prompt indices mapping
        self.class_to_prompt_indices: Dict[int, List[int]] = {}
        for prompt_idx, class_id in enumerate(prompts["indices"]):
            if class_id not in self.class_to_prompt_indices:
                self.class_to_prompt_indices[class_id] = []
            self.class_to_prompt_indices[class_id].append(prompt_idx)

    @property
    def amp_ctx(self):
        if self.amp_enabled and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        from contextlib import nullcontext
        return nullcontext()

    def _view_weights(self, views):
        if self.use_context_view and self.use_masked_view and len(views) == 2:
            w = [self.view_weight_context, self.view_weight_masked]
        else:
            w = [1.0] * len(views)
        s = sum(w)
        return [x / max(1e-6, s) for x in w]

    @torch.inference_mode()
    def _compute_crop_embedding(self, views) -> Optional[torch.Tensor]:
        """Compute weighted visual embedding from crop views."""
        ws = self._view_weights(views)
        emb_acc = None
        for v, wv in zip(views, ws):
            with self.amp_ctx:
                st = self.processor.set_image(v)
                e = extract_sam3_image_embedding(st["backbone_out"])
            e = e.float() * float(wv)
            emb_acc = e if emb_acc is None else (emb_acc + e)
        if emb_acc is None:
            return None
        return emb_acc / emb_acc.norm(p=2).clamp_min(1e-6)

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def build(
        self,
        dataset,
        num_calib: int = 24,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Build concept bank and return text_features_cache.

        Args:
            dataset: Dataset with __getitem__ returning {"image_path": str, "mask": tensor}
            num_calib: Number of calibration images to use

        Returns:
            text_features_cache: Same format as segmentor's text_features_cache,
                but with fused embeddings replacing individual prompt embeddings.
        """
        C = self.num_classes
        N = min(num_calib, len(dataset))

        print(f"\n{'='*60}")
        print(f"Concept Bank Builder")
        print(f"  Classes: {C}, Prompts: {self.num_prompts}")
        print(f"  Calibration images: {N}/{len(dataset)}")
        print(f"  Top-K per class: {self.top_k_per_class}, tau_w: {self.tau_w}")
        print(f"{'='*60}\n")

        # =====================================================
        # Stage I: Prototype Collection
        # =====================================================
        print("[Stage I] Collecting visual prototypes...")
        t_start = time.time()

        D = 0
        sum_embeddings = None
        count_per_class = np.zeros((C,), dtype=np.int32)

        # Determine cap: how many classes to process per image.
        # Original formula targets large-vocabulary datasets (e.g. ADE20K with 150 classes).
        # For few-class datasets (C <= 20), process all classes per image since each
        # crop is cheap and we want sufficient samples per class.
        if C <= 20:
            cap_pass1 = C
        else:
            cap_pass1 = max(3, min(9, int(np.sqrt(C) / 5.0 + 2.0)))

        it1 = tqdm(range(N), desc="Stage I", dynamic_ncols=True)
        for idx in it1:
            if (count_per_class >= self.max_per_class_pass1).all():
                break

            try:
                sample = dataset[idx]
                image_path = sample["image_path"]
                gt_mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else np.asarray(sample["mask"])
            except Exception as e:
                continue

            img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

            # Find valid classes in this mask
            valid_classes = []
            for class_id in range(C):
                if class_id == self.ignore_index:
                    continue
                if count_per_class[class_id] >= self.max_per_class_pass1:
                    continue
                if np.any(gt_mask == class_id):
                    valid_classes.append(class_id)

            # Prioritize classes with fewer samples
            valid_classes.sort(key=lambda c: (count_per_class[c], c))

            for class_id in valid_classes[:cap_pass1]:
                views, _ = make_crop_views_from_np(
                    img, gt_mask, class_id,
                    pad_ratio=self.pad_ratio,
                    min_size=self.min_crop_size,
                    use_context=self.use_context_view,
                    use_masked=self.use_masked_view,
                )
                if views is None:
                    continue

                try:
                    e = self._compute_crop_embedding(views)
                except Exception:
                    continue

                if e is None:
                    continue

                if D == 0:
                    D = int(e.numel())
                    sum_embeddings = torch.zeros((C, D), device=self.device, dtype=torch.float32)

                if int(e.numel()) == D:
                    sum_embeddings[class_id] += e.float()
                    count_per_class[class_id] += 1

        if D == 0:
            print("[WARNING] No valid crops found in Stage I. Falling back to original text features.")
            return self._build_original_cache()

        count_t = torch.tensor(count_per_class, device=self.device, dtype=torch.float32)
        proto = sum_embeddings / count_t.clip(min=1).unsqueeze(1)
        proto = proto / proto.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        proto_valid = count_per_class > 0

        print(f"  Stage I done in {time.time()-t_start:.1f}s, "
              f"valid protos: {proto_valid.sum()}/{C}, "
              f"total crops: {count_per_class.sum()}")

        del sum_embeddings
        self._cleanup_cuda()

        # =====================================================
        # Stage II: Representative Mining (Top-K Selection)
        # =====================================================
        print("[Stage II] Mining representative crops...")
        t_start = time.time()

        heaps = [[] for _ in range(C)]
        tie_counter = 0
        target = [self.top_k_per_class] * C
        emb_cache = LRUEmbCache(max_items=self.pass2_emb_cache_size)

        def all_full():
            for c in range(C):
                if target[c] > 0 and len(heaps[c]) < target[c] and proto_valid[c]:
                    return False
            return True

        for epoch in range(self.pass2_max_epochs):
            if all_full():
                break
            cap_pass2 = min(12, cap_pass1 * (2 ** epoch))

            it2 = tqdm(range(N), desc=f"Stage II (e{epoch})", dynamic_ncols=True)
            for idx in it2:
                if all_full():
                    break

                try:
                    sample = dataset[idx]
                    image_path = sample["image_path"]
                    gt_mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else np.asarray(sample["mask"])
                except Exception:
                    continue

                img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

                # Find candidate classes (need more samples)
                cand = []
                for class_id in range(C):
                    if target[class_id] <= 0 or not proto_valid[class_id]:
                        continue
                    if len(heaps[class_id]) < target[class_id]:
                        if np.any(gt_mask == class_id):
                            cand.append((target[class_id] - len(heaps[class_id]), class_id))

                if not cand:
                    continue
                cand.sort(key=lambda x: (-x[0], x[1]))

                qids, dsids, embs = [], [], []
                for _, class_id in cand[:cap_pass2]:
                    if len(heaps[class_id]) >= target[class_id]:
                        continue

                    cache_key = (idx, int(class_id), self.pad_ratio, self.min_crop_size)
                    cached = emb_cache.get(cache_key)

                    if cached is not None:
                        e = cached.to(self.device, dtype=torch.float32)
                    else:
                        views, _ = make_crop_views_from_np(
                            img, gt_mask, class_id,
                            pad_ratio=self.pad_ratio,
                            min_size=self.min_crop_size,
                            use_context=self.use_context_view,
                            use_masked=self.use_masked_view,
                        )
                        if views is None:
                            continue
                        try:
                            e = self._compute_crop_embedding(views)
                        except Exception:
                            continue
                        if e is not None:
                            emb_cache.put(cache_key, e.detach().cpu().half())

                    if e is not None and int(e.numel()) == D:
                        qids.append(int(class_id))
                        dsids.append(int(class_id))
                        embs.append(e)

                if not embs:
                    continue

                E = torch.stack(embs, dim=0).float()
                P = proto[torch.tensor(qids, device=self.device)].float()
                scores = (E * P).sum(dim=1).detach().cpu().tolist()

                for qid, dsid, sc in zip(qids, dsids, scores):
                    tie_counter += 1
                    item = (float(sc), tie_counter, int(idx), int(dsid))
                    h = heaps[qid]
                    if len(h) < target[qid]:
                        heapq.heappush(h, item)
                    elif float(sc) > h[0][0]:
                        heapq.heapreplace(h, item)

        # Select top-K representatives per class
        selected_meta: List[List[Tuple[int, int, float]]] = []
        for c in range(C):
            sorted_items = sorted(heaps[c], key=lambda x: x[0], reverse=True)
            K = target[c]
            top = sorted_items[:K] if K > 0 else []
            selected_meta.append([(idx, dsid, float(score)) for (score, _, idx, dsid) in top])

        total_reps = sum(len(selected_meta[c]) for c in range(C))
        print(f"  Stage II done in {time.time()-t_start:.1f}s, "
              f"representative crops: {total_reps}")

        self._cleanup_cuda()

        # =====================================================
        # Stage III: Prompt Scoring & Fusion (aligned with official)
        # =====================================================
        # 1. Per-class batch encoding: encode ALL prompts for a class
        #    together so they share the same token length T (via padding)
        # 2. BTD alignment for consistent fusion
        # 3. Score prompts on representative crops via dice
        # 4. fuse_tokens (weighted average, same T guaranteed)
        # 5. to_raw_from_BTD to restore original model layout [T, B, D]
        print("[Stage III] Scoring prompts on representative crops...")
        t_start = time.time()

        # Per-class batch text encoding (official approach)
        print(f"  Encoding text features per class ({C} classes)...")
        # text_db_class[c] = list of (lf[T, Df], pm[T], le[T, De]) in BTD format
        text_db_class: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        # Also store per-prompt raw features for fallback (original model layout)
        text_db_raw: Dict[int, Dict[str, torch.Tensor]] = {}
        # Store the layout from the first class to use for to_raw conversion
        global_layout: Dict[str, dict] = {}

        with torch.no_grad():
            for class_id in range(C):
                prompt_indices = self.class_to_prompt_indices.get(class_id, [])
                if not prompt_indices:
                    text_db_class.append([])
                    continue

                phrases = [self.prompts["names"][pi] for pi in prompt_indices]
                with self.amp_ctx:
                    out = self.processor.model.backbone.forward_text(
                        phrases, device=self.device
                    )

                B = len(phrases)
                lf_btd, pm_bt, le_btd, layout = _align_language_outputs(out, B)

                if class_id == 0 and not global_layout:
                    global_layout = layout

                # Per-prompt items in BTD format: [T, Df], [T], [T, De]
                items = []
                for i in range(B):
                    items.append((
                        lf_btd[i].detach().cpu(),
                        pm_bt[i].detach().cpu(),
                        le_btd[i].detach().cpu(),
                    ))
                    # Store raw features for fallback: convert BTD [1, T, D] → raw [T, 1, D]
                    pi = prompt_indices[i]
                    text_db_raw[pi] = {
                        "language_features": to_raw_from_BTD(
                            lf_btd[i:i+1].cpu(), layout["language_features"]
                        ),
                        "language_mask": mask_to_raw_from_BT(
                            pm_bt[i:i+1].cpu(), layout["language_mask"]
                        ),
                        "language_embeds": to_raw_from_BTD(
                            le_btd[i:i+1].cpu(), layout["language_embeds"]
                        ),
                    }
                text_db_class.append(items)

            # Also encode remaining prompts not covered by class-to-prompt mapping
            for prompt_idx in range(self.num_prompts):
                if prompt_idx not in text_db_raw:
                    prompt_word = self.prompts["names"][prompt_idx]
                    with self.amp_ctx:
                        out = self.processor.model.backbone.forward_text(
                            [prompt_word], device=self.device
                        )
                    text_db_raw[prompt_idx] = {
                        "language_features": out["language_features"].detach().cpu(),
                        "language_mask": out["language_mask"].detach().cpu(),
                        "language_embeds": out["language_embeds"].detach().cpu(),
                    }

        if not global_layout:
            # Fallback: no classes had prompts
            print("  WARNING: No text features could be encoded")
            return self._build_original_cache()

        fused_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        inference_fn = None
        if self.segmentor is not None:
            inference_fn = self.segmentor._inference_single_view

        for class_id in range(C):
            items = text_db_class[class_id]
            prompt_indices = self.class_to_prompt_indices.get(class_id, [])
            if not items:
                continue

            keep = list(range(min(len(items), self.cand_topk)))
            if not keep:
                continue

            # Get Stage II representative crops for this class
            rep_crops = selected_meta[class_id]

            if not rep_crops:
                # No representative crops: uniform fusion over all kept prompts
                w_uniform = 1.0 / len(keep)
                selected_tensors = [
                    (items[i][0], items[i][1], items[i][2], w_uniform)
                    for i in keep
                ]
                lf_f_btd, pm_f_bt, le_f_btd = fuse_tokens(selected_tensors)
                # Convert [T, Df], [T], [T, De] → raw model layout [T, B=1, D]
                lf_raw = to_raw_from_BTD(
                    lf_f_btd.unsqueeze(0).to(self.device).half(), global_layout["language_features"]
                ).float()
                pm_raw = mask_to_raw_from_BT(
                    pm_f_bt.unsqueeze(0).to(self.device), global_layout["language_mask"]
                ).bool()
                le_raw = to_raw_from_BTD(
                    le_f_btd.unsqueeze(0).to(self.device).half(), global_layout["language_embeds"]
                ).float()
                fused_cache[class_id] = {
                    "language_features": lf_raw,
                    "language_mask": pm_raw,
                    "language_embeds": le_raw,
                }
                phrase_name = self.prompts["names"][prompt_indices[0]]
                print(f"  Class {class_id}: no crops, uniform fusion ({len(keep)} prompts)")
                continue

            # Score each prompt by averaging dice over representative crops
            num_candidates = len(keep)
            dice_scores = torch.zeros((num_candidates,), dtype=torch.float32, device="cpu")

            for crop_rank, (img_idx, dsid, _) in enumerate(rep_crops):
                try:
                    sample = dataset[img_idx]
                    image_path = sample["image_path"]
                    gt_mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else np.asarray(sample["mask"])
                except Exception:
                    continue

                img_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

                box = square_box_from_mask(
                    gt_mask, dsid, self.pad_ratio, self.min_crop_size,
                    self.max_crop_size, self.rng,
                )
                if box is None:
                    continue

                x1, y1, x2, y2 = box
                crop_np = img_np[y1:y2 + 1, x1:x2 + 1].copy()
                gt_crop = (gt_mask[y1:y2 + 1, x1:x2 + 1] == dsid).astype(np.float32)

                if gt_crop.sum() == 0:
                    continue

                crop_h, crop_w = crop_np.shape[:2]
                crop_img = Image.fromarray(crop_np)
                gt_crop_tensor = torch.from_numpy(gt_crop).float()

                try:
                    if inference_fn is not None:
                        seg_logits, _, _, _, _ = inference_fn(
                            crop_img, detailed=False,
                            image_name=f"stage3_{class_id}_crop{crop_rank}",
                        )
                    else:
                        seg_logits, _, _, _, _ = self.engine.inference_batch_view(
                            [crop_img], detailed=False,
                        )
                        seg_logits = seg_logits[0]

                    for pi_local in keep:
                        prompt_idx = prompt_indices[pi_local]
                        prob = seg_logits[prompt_idx].sigmoid().cpu()
                        if prob.shape[0] != crop_h or prob.shape[1] != crop_w:
                            prob = F.interpolate(
                                prob.unsqueeze(0).unsqueeze(0),
                                size=(crop_h, crop_w),
                                mode="bilinear", align_corners=False,
                            ).squeeze()
                        dice_scores[pi_local] += soft_dice_score(prob, gt_crop_tensor)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self._cleanup_cuda()
                    continue
                except Exception:
                    continue

                self._cleanup_cuda()

            dice_scores /= max(len(rep_crops), 1)

            # Top-K + adaptive gating (official)
            order = torch.argsort(dice_scores, descending=True).tolist()
            top = order[:max(1, min(self.fuse_topk, len(order)))]

            s = dice_scores[top].float()
            max_s = s.max() if s.numel() > 0 else torch.tensor(0.0)
            adaptive_mask = s >= (max_s - 0.3)

            if adaptive_mask.sum() > 0:
                s_valid = s[adaptive_mask]
                top_valid = [top[i] for i in range(len(top)) if bool(adaptive_mask[i])]
            else:
                s_valid = s
                top_valid = top

            if s_valid.numel() == 0:
                w_list = [1.0 / len(top_valid)] if top_valid else [1.0]
            else:
                w_list = torch.softmax(s_valid / max(self.tau_w, 1e-6), dim=0).detach().cpu().tolist()

            # Fuse selected prompts (token-level weighted average)
            selected_tensors = [
                (items[idx_local][0], items[idx_local][1], items[idx_local][2], float(w))
                for w, idx_local in zip(w_list, top_valid)
            ]
            lf_f_btd, pm_f_bt, le_f_btd = fuse_tokens(selected_tensors)

            # Convert from BTD [T, D] back to raw model layout [T, B=1, D]
            lf_raw = to_raw_from_BTD(
                lf_f_btd.unsqueeze(0).to(self.device).half(), global_layout["language_features"]
            ).float()
            pm_raw = mask_to_raw_from_BT(
                pm_f_bt.unsqueeze(0).to(self.device), global_layout["language_mask"]
            ).bool()
            le_raw = to_raw_from_BTD(
                le_f_btd.unsqueeze(0).to(self.device).half(), global_layout["language_embeds"]
            ).float()
            fused_cache[class_id] = {
                "language_features": lf_raw,
                "language_mask": pm_raw,
                "language_embeds": le_raw,
            }

            chosen_info = [
                (self.prompts["names"][prompt_indices[top_valid[i]]], float(w_list[i]),
                 float(dice_scores[top_valid[i]]))
                for i in range(len(top_valid))
            ]
            print(f"  Class {class_id}: {len(keep)} candidates, "
                  f"{len(top_valid)} fused, top_dice={dice_scores[top[0]]:.4f}, "
                  f"chosen={[(n, f'{w:.2f}') for n, w, _ in chosen_info]}")

            self._cleanup_cuda()

        # =====================================================
        # Build final text_features_cache (keyed by class_id)
        # =====================================================
        # After fusion, we have one feature vector per class.
        # Store as class_id -> fused features so the segmentor can
        # use num_prompts == num_classes and skip fuse_prompts_to_classes.
        final_cache = {}
        class_names = []  # representative name for each class (for prompts)

        for class_id in range(C):
            if class_id in fused_cache:
                final_cache[class_id] = {
                    "language_features": fused_cache[class_id]["language_features"],
                    "language_mask": fused_cache[class_id]["language_mask"],
                    "language_embeds": fused_cache[class_id]["language_embeds"],
                }
                # Pick the first prompt name belonging to this class as representative
                prompt_indices = self.class_to_prompt_indices.get(class_id, [])
                class_names.append(self.prompts["names"][prompt_indices[0]] if prompt_indices else f"class_{class_id}")
            else:
                # Fallback: find any prompt_idx with this class_id
                for pi, ci in enumerate(self.prompts["indices"]):
                    if ci == class_id and pi in text_db_raw:
                        final_cache[class_id] = {
                            "language_features": text_db_raw[pi]["language_features"].to(self.device),
                            "language_mask": text_db_raw[pi]["language_mask"].to(self.device),
                            "language_embeds": text_db_raw[pi]["language_embeds"].to(self.device),
                        }
                        class_names.append(self.prompts["names"][pi])
                        break

        print(f"\nConcept Bank built in {time.time()-t_start:.1f}s")
        print(f"  Fused classes: {len(fused_cache)}/{C}")
        print(f"  Cache entries: {len(final_cache)}")
        print(f"  Class names: {class_names}")

        self._class_names = class_names
        return final_cache

    def _build_original_cache(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Fallback: return original text features without fusion."""
        from segmentor_lib.prompts import precompute_text_features
        return precompute_text_features(
            self.processor, self.prompts, self.device, self.num_prompts
        )

    def build_and_save(
        self,
        dataset,
        num_calib: int = 24,
        save_path: Optional[str] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Build concept bank, optionally save to disk, and return cache."""
        cache = self.build(dataset, num_calib)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            save_dict = {
                "prompts": self.prompts,
                "class_names": getattr(self, '_class_names', []),
                "cache": {str(k): {
                    "language_features": v["language_features"].cpu(),
                    "language_mask": v["language_mask"].cpu(),
                    "language_embeds": v["language_embeds"].cpu(),
                } for k, v in cache.items()},
                "config": {
                    "tau_w": self.tau_w,
                    "top_k_per_class": self.top_k_per_class,
                    "pad_ratio": self.pad_ratio,
                    "num_calib": num_calib,
                },
            }
            torch.save(save_dict, save_path)
            print(f"Concept Bank saved to {save_path}")

        return cache

    @staticmethod
    def load(save_path: str, device: torch.device) -> dict:
        """Load a saved concept bank from disk.

        Returns:
            dict with keys:
                - 'cache': Dict[int, Dict[str, Tensor]] - class_id -> features
                - 'class_names': List[str] - representative name per class
                - 'prompts': original prompts dict (optional)
                - 'config': build config (optional)
        """
        data = torch.load(save_path, map_location="cpu", weights_only=False)
        cache = {}
        for k, v in data["cache"].items():
            cache[int(k)] = {
                "language_features": v["language_features"].to(device),
                "language_mask": v["language_mask"].to(device),
                "language_embeds": v["language_embeds"].to(device),
            }
        return {
            "cache": cache,
            "class_names": data.get("class_names", []),
            "prompts": data.get("prompts"),
            "config": data.get("config"),
        }

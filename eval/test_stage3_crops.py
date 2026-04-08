"""Test script for Concept Bank Stage III prompt scoring and fusion visualization.

Follows the OFFICIAL Concept Bank approach:
1. Stage I: compute per-class visual prototypes
2. Stage II: mine top-K representative crops per class
3. Stage III: on each representative crop, run inference, compare with GT crop
   to score each prompt, then fuse the best prompts.

This is equivalent to the official sam3_concept_bank.py Stage III which uses
forward_grounding + build_prob_map on cropped regions.  Here we use
_inference_single_view (single-view inference on the crop) instead.

Usage:
    python eval/test_stage3_crops.py \
        --config eval/configs/potsdam_A2_extended.yaml \
        --num_calib 3

Output directory: outputs/build_cbank_stage3/
    summary.txt
"""

from __future__ import annotations

import argparse
import gc
import heapq
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Repository roots
EVAL_DIR = Path(__file__).resolve().parent
SAM3_RS_DIR = EVAL_DIR.parent

if str(SAM3_RS_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_RS_DIR))

from build_concept_bank import load_config, build_dataset  # noqa: E402
from segmentor import SAM3RSSegmentor, InferenceConfig  # noqa: E402
from segmentor_lib.experimental.concept_bank import (  # noqa: E402
    fuse_tokens,
    soft_dice_score,
    square_box_from_mask,
    make_crop_views_from_np,
    extract_sam3_image_embedding,
    LRUEmbCache,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Stage III prompt scoring & fusion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_calib", type=int, default=3,
                        help="Calibration images (used for scoring)")
    parser.add_argument("--tau_w", type=float, default=0.15)
    parser.add_argument("--fuse_topk", type=int, default=999)
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-K representative crops per class from Stage II")
    parser.add_argument("--pad_ratio", type=float, default=0.05)
    parser.add_argument("--min_crop_size", type=int, default=128)
    parser.add_argument("--max_crop_size", type=int, default=1024)
    parser.add_argument("--max_per_class_pass1", type=int, default=10)
    parser.add_argument("--pass2_epochs", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset = build_dataset(cfg)
    N = min(args.num_calib, len(dataset))

    ds_cfg = cfg["dataset"]
    ignore_index = ds_cfg.get("ignore_index", 255)
    cls_file = ds_cfg.get("cls_file", "")

    # Load class names
    class_names = {}
    cls_path = EVAL_DIR / "datasets" / os.path.basename(cls_file)
    if cls_path.exists():
        with open(cls_path, "r") as f:
            for i, line in enumerate(f):
                class_names[i] = line.strip()

    # Determine number of classes
    sample = dataset[0]
    gt_mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else np.asarray(sample["mask"])
    C = int(gt_mask.max()) + 1

    # Output directory
    out_root = Path(args.output) if args.output else SAM3_RS_DIR / "outputs/build_cbank_stage3"
    out_root.mkdir(parents=True, exist_ok=True)

    # =============================================
    # Load SAM3 model
    # =============================================
    print("Loading SAM3 model...")
    seg_cfg = cfg["segmentor"]
    for key in ["semantic_prob_thresholds", "instance_prob_thresholds"]:
        if key in seg_cfg and seg_cfg[key] is not None:
            seg_cfg[key] = {int(k): v for k, v in seg_cfg[key].items()}

    infer_kwargs = dict(seg_cfg)
    for drop_key in ["weight_key", "checkpoint_filename", "debug_memory", "debug_log_file",
                     "analyze_statistics", "compute_boundary_iou", "analyze_presence_score",
                     "prob_thresholds"]:
        infer_kwargs.pop(drop_key, None)

    infer_cfg = InferenceConfig(
        checkpoint_path=infer_kwargs.pop("checkpoint_path"),
        bpe_path=infer_kwargs.pop("bpe_path"),
        device=infer_kwargs.pop("device", "cuda"),
        **infer_kwargs,
    )
    segmentor = SAM3RSSegmentor(infer_cfg)
    device = segmentor.device
    processor = segmentor.processor
    prompts = segmentor.prompts

    num_prompts = len(prompts["names"])

    # Build class -> prompt indices mapping
    class_to_prompt_indices = {}
    for prompt_idx, class_id in enumerate(prompts["indices"]):
        if class_id not in class_to_prompt_indices:
            class_to_prompt_indices[class_id] = []
        class_to_prompt_indices[class_id].append(prompt_idx)

    # =============================================
    # Parameters
    # =============================================
    pad_ratio = args.pad_ratio
    min_size = args.min_crop_size
    max_size = args.max_crop_size
    max_per_class = args.max_per_class_pass1
    top_k = args.top_k

    if C <= 20:
        cap_pass1 = C
    else:
        cap_pass1 = max(3, min(9, int(np.sqrt(C) / 5.0 + 2.0)))

    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if device.type == "cuda"
               else __import__("contextlib").nullcontext())

    print(f"\nConcept Bank Stage III Test")
    print(f"  Dataset: {cfg['dataset']['name']}, Calibration images: {N}")
    print(f"  Classes: {C}, Total prompts: {num_prompts}")
    print(f"  tau_w: {args.tau_w}, fuse_topk: {args.fuse_topk}, top_k: {top_k}")
    print(f"  Output: {out_root}")
    print(f"\n  Prompts:")
    for pi in range(num_prompts):
        print(f"    [{pi}] class={prompts['indices'][pi]:<2} '{prompts['names'][pi]}'")
    print()

    # =============================================
    # Stage I: Prototype Collection
    # =============================================
    print("=" * 60)
    print("[Stage I] Collecting visual prototypes...")
    t_start = time.time()

    D = 0
    sum_embeddings = None
    count_per_class = np.zeros((C,), dtype=np.int32)
    rng = np.random.RandomState(42)

    for idx in tqdm(range(N), desc="Stage I"):
        if (count_per_class >= max_per_class).all():
            break

        try:
            s = dataset[idx]
            image_path = s["image_path"]
            gt = s["mask"].numpy() if torch.is_tensor(s["mask"]) else np.asarray(s["mask"])
        except Exception:
            continue

        img_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        valid_classes = []
        for class_id in range(C):
            if class_id == ignore_index:
                continue
            if count_per_class[class_id] >= max_per_class:
                continue
            if np.any(gt == class_id):
                valid_classes.append(class_id)

        valid_classes.sort(key=lambda c: (count_per_class[c], c))

        for class_id in valid_classes[:cap_pass1]:
            views, _ = make_crop_views_from_np(
                img_np, gt, class_id,
                pad_ratio=pad_ratio, min_size=min_size,
                max_size=max_size,
                use_context=True, use_masked=False, rng=rng,
            )
            if views is None:
                continue
            try:
                with amp_ctx:
                    st = processor.set_image(views[0])
                    e = extract_sam3_image_embedding(st["backbone_out"])
            except Exception:
                continue

            if D == 0:
                D = int(e.numel())
                sum_embeddings = torch.zeros((C, D), device=device, dtype=torch.float32)

            if int(e.numel()) == D:
                sum_embeddings[class_id] += e.float()
                count_per_class[class_id] += 1

    if D == 0:
        print("[ERROR] No valid crops found. Exiting.")
        return

    count_t = torch.tensor(count_per_class, device=device, dtype=torch.float32)
    proto = sum_embeddings / count_t.clip(min=1).unsqueeze(1)
    proto = proto / proto.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    proto_valid = count_per_class > 0

    print(f"  Stage I done in {time.time() - t_start:.1f}s")
    print(f"  Prototypes: {proto_valid.sum()}/{C}, total crops: {count_per_class.sum()}")

    del sum_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =============================================
    # Stage II: Representative Mining
    # =============================================
    print(f"\n[Stage II] Mining representative crops...")
    t_start = time.time()

    heaps = [[] for _ in range(C)]
    tie_counter = 0
    target = [top_k] * C
    emb_cache = LRUEmbCache(max_items=8192)

    def all_full():
        for c in range(C):
            if target[c] > 0 and len(heaps[c]) < target[c] and proto_valid[c]:
                return False
        return True

    for epoch in range(args.pass2_epochs):
        if all_full():
            break
        cap_pass2 = min(12, cap_pass1 * (2 ** epoch))

        for idx in tqdm(range(N), desc=f"Stage II (e{epoch})"):
            if all_full():
                break

            try:
                s = dataset[idx]
                image_path = s["image_path"]
                gt = s["mask"].numpy() if torch.is_tensor(s["mask"]) else np.asarray(s["mask"])
            except Exception:
                continue

            img_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

            cand = []
            for class_id in range(C):
                if target[class_id] <= 0 or not proto_valid[class_id]:
                    continue
                if len(heaps[class_id]) < target[class_id]:
                    if np.any(gt == class_id):
                        cand.append((target[class_id] - len(heaps[class_id]), class_id))

            if not cand:
                continue
            cand.sort(key=lambda x: (-x[0], x[1]))

            qids, dsids, embs = [], [], []
            for _, class_id in cand[:cap_pass2]:
                if len(heaps[class_id]) >= target[class_id]:
                    continue

                cache_key = (idx, int(class_id), pad_ratio, min_size)
                cached = emb_cache.get(cache_key)
                if cached is not None:
                    e = cached.to(device, dtype=torch.float32)
                else:
                    views, _ = make_crop_views_from_np(
                        img_np, gt, class_id,
                        pad_ratio=pad_ratio, min_size=min_size,
                        max_size=max_size,
                        use_context=True, use_masked=False, rng=rng,
                    )
                    if views is None:
                        continue
                    try:
                        with amp_ctx:
                            st = processor.set_image(views[0])
                            e = extract_sam3_image_embedding(st["backbone_out"])
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
            P = proto[torch.tensor(qids, device=device)].float()
            scores = (E * P).sum(dim=1).detach().cpu().tolist()

            for qid, dsid, sc in zip(qids, dsids, scores):
                tie_counter += 1
                item = (float(sc), tie_counter, int(idx), int(dsid))
                h = heaps[qid]
                if len(h) < target[qid]:
                    heapq.heappush(h, item)
                elif float(sc) > h[0][0]:
                    heapq.heapreplace(h, item)

    print(f"  Stage II done in {time.time() - t_start:.1f}s")
    for class_id in range(C):
        n_reps = min(len(heaps[class_id]), top_k)
        cls_label = class_names.get(class_id, f"class_{class_id}")
        print(f"    Class {class_id} ({cls_label}): {n_reps} representatives")

    del emb_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =============================================
    # Pre-compute text features for all prompts
    # =============================================
    print(f"\n  Pre-computing text features for {num_prompts} prompts...")
    text_db = {}
    with torch.no_grad():
        for prompt_idx in range(num_prompts):
            prompt_word = prompts["names"][prompt_idx]
            with amp_ctx:
                text_outputs = processor.model.backbone.forward_text(
                    [prompt_word], device=device
                )
            text_db[prompt_idx] = {
                "language_features": text_outputs["language_features"].detach().cpu(),
                "language_mask": text_outputs["language_mask"].detach().cpu(),
                "language_embeds": text_outputs["language_embeds"].detach().cpu(),
            }
    print(f"  Done: {len(text_db)} prompts")

    # =============================================
    # Stage III: Prompt Scoring on Representative Crops
    # =============================================
    # Official approach: for each class, use Stage II representative crops.
    # On each crop, run inference and compare with GT crop to score prompts.
    # This is equivalent to official _score_prompts_auto_fallback which uses
    # forward_grounding + build_prob_map on cropped regions.
    #
    # We use _inference_single_view which runs the full pipeline (set_image +
    # forward_grounding + dual-head fusion) on the crop.
    print(f"\n{'=' * 60}")
    print(f"[Stage III] Scoring prompts on representative crops...")
    t_start = time.time()

    all_class_results = {}  # class_id -> {prompts, scores, selected, weights}

    for class_id in range(C):
        prompt_indices = class_to_prompt_indices.get(class_id, [])
        if not prompt_indices:
            continue

        keep = prompt_indices[:args.fuse_topk]
        if not keep:
            continue

        cls_label = class_names.get(class_id, f"class_{class_id}")
        print(f"\n--- Class {class_id} ({cls_label}) ---")
        print(f"  Candidate prompts: {len(keep)}")

        # Get Stage II representative crops for this class
        sorted_items = sorted(heaps[class_id], key=lambda x: x[0], reverse=True)
        rep_crops = sorted_items[:top_k]

        if not rep_crops:
            print(f"  No representative crops found, using uniform fusion.")
            w_uniform = 1.0 / len(keep)
            w_list = [w_uniform] * len(keep)
            all_class_results[class_id] = {
                "prompts": [(pi, prompts["names"][pi]) for pi in keep],
                "scores": [0.0] * len(keep),
                "selected": list(range(len(keep))),
                "weights": w_list,
            }
            continue

        print(f"  Using {len(rep_crops)} representative crops from Stage II")

        # Score each prompt by averaging dice over representative crops
        num_candidates = len(keep)
        dice_scores = torch.zeros((num_candidates,), dtype=torch.float32, device="cpu")

        for crop_rank, (crop_score, _, img_idx, dsid) in enumerate(rep_crops):
            try:
                s = dataset[img_idx]
                image_path = s["image_path"]
                gt = s["mask"].numpy() if torch.is_tensor(s["mask"]) else np.asarray(s["mask"])
            except Exception:
                continue

            img_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            img_name = Path(image_path).stem

            # Compute the crop bbox (same as Stage II)
            box = square_box_from_mask(gt, dsid, pad_ratio, min_size, max_size, rng)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            crop_np = img_np[y1:y2 + 1, x1:x2 + 1].copy()
            gt_crop = (gt[y1:y2 + 1, x1:x2 + 1] == dsid).astype(np.float32)

            if gt_crop.sum() == 0:
                continue

            crop_h, crop_w = crop_np.shape[:2]
            crop_img = Image.fromarray(crop_np)
            gt_crop_tensor = torch.from_numpy(gt_crop).float()

            print(f"  Crop #{crop_rank}: {img_name} bbox=({x1},{y1})-({x2},{y2}) "
                  f"size={crop_w}x{crop_h} repscore={crop_score:.3f}")

            # Run single-view inference on the crop
            # This calls: set_image -> forward_grounding for each prompt -> dual-head fusion
            try:
                seg_logits, _, _, _, _ = segmentor._inference_single_view(
                    crop_img, detailed=False, image_name=f"stage3_{class_id}_crop{crop_rank}"
                )
                # seg_logits: [num_prompts, crop_h, crop_w]

                for pi_local, prompt_idx in enumerate(keep):
                    prob = seg_logits[prompt_idx].sigmoid().cpu()
                    # Resize prob to match GT crop if sizes differ
                    if prob.shape[0] != crop_h or prob.shape[1] != crop_w:
                        prob = torch.nn.functional.interpolate(
                            prob.unsqueeze(0).unsqueeze(0),
                            size=(crop_h, crop_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                    dice_scores[pi_local] += soft_dice_score(prob, gt_crop_tensor)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(f"    [WARN] OOM on crop, skipped")
                continue
            except Exception as e:
                print(f"    [WARN] Error on crop: {e}")
                continue

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average over representative crops
        dice_scores /= max(len(rep_crops), 1)

        # Sort by score
        order = torch.argsort(dice_scores, descending=True).tolist()
        top = order[:max(1, min(args.fuse_topk, len(order)))]

        # Adaptive gating
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
            w_list = torch.softmax(s_valid / max(args.tau_w, 1e-6), dim=0).detach().cpu().tolist()

        # Print scores
        print(f"\n  Prompt scores (sorted):")
        for rank, oi in enumerate(order):
            pi = keep[oi]
            prompt_name = prompts["names"][pi]
            score = dice_scores[oi].item()
            selected_mark = " *" if oi in top_valid else ""
            print(f"    [{rank}] dice={score:.4f}  '{prompt_name}'{selected_mark}")

        print(f"\n  Selected for fusion: {len(top_valid)} prompts")
        for i, (wi, oi) in enumerate(zip(w_list, top_valid)):
            pi = keep[oi]
            prompt_name = prompts["names"][pi]
            print(f"    [{i}] weight={wi:.4f}  dice={dice_scores[oi].item():.4f}  '{prompt_name}'")

        all_class_results[class_id] = {
            "prompts": [(keep[oi], prompts["names"][keep[oi]]) for oi in order],
            "scores": [dice_scores[oi].item() for oi in order],
            "selected": top_valid,
            "weights": w_list,
        }

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Stage III done in {elapsed:.1f}s")

    # =============================================
    # Save summary
    # =============================================
    summary_lines = [
        f"Concept Bank Stage III Prompt Scoring Results (Official Crop-Based)\n"
        f"Dataset: {cfg['dataset']['name']}\n"
        f"Calibration images: {N}\n"
        f"Top-K representative crops: {top_k}\n"
        f"tau_w: {args.tau_w}\n"
        f"{'=' * 60}\n",
    ]

    for class_id in sorted(all_class_results.keys()):
        result = all_class_results[class_id]
        cls_label = class_names.get(class_id, f"class_{class_id}")
        summary_lines.append(f"\nClass {class_id} ({cls_label})")
        summary_lines.append(f"  {'Rank':<6}{'Dice':<10}{'Prompt':<40}{'Selected'}")

        for rank, ((pi, name), score) in enumerate(zip(result["prompts"], result["scores"])):
            sel_mark = ""
            for si, oi in enumerate(result["selected"]):
                if rank == oi:
                    sel_mark = f"w={result['weights'][si]:.4f}"
            summary_lines.append(f"  {rank:<6}{score:<10.4f}{name:<40}{sel_mark}")

        summary_lines.append(f"  -> Fused {len(result['selected'])} prompts")

    summary_path = out_root / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nOutput saved to: {out_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

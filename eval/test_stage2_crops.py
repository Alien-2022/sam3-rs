"""Test script for Concept Bank Stage I + Stage II representative mining visualization.

Runs Stage I (prototype collection) and Stage II (top-K selection),
then saves the selected representative crops for visual inspection.

Usage:
    python eval/test_stage2_crops.py \
        --config eval/configs/potsdam_A2_extended.yaml \
        --num_calib 5 --top_k 5

Output directory: outputs/build_cbank_stage2/
    outputs/build_cbank_stage2/
        class_0_impervious_surface/
            rank_0_score_0.85_img_2522.png
            rank_0_score_0.85_img_2522_mask.png
            rank_1_score_0.82_img_2523.png
            ...
        class_1_building/
            rank_0_score_0.91_img_2522.png
            ...
        summary.txt
"""

from __future__ import annotations

import argparse
import gc
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
    square_box_from_mask,
    make_crop_views_from_np,
    extract_sam3_image_embedding,
    LRUEmbCache,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Stage II representative mining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_calib", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5, help="Top-K per class to visualize")
    parser.add_argument("--pad_ratio", type=float, default=0.05)
    parser.add_argument("--min_crop_size", type=int, default=128)
    parser.add_argument("--max_crop_size", type=int, default=1024)
    parser.add_argument("--max_per_class_pass1", type=int, default=10)
    parser.add_argument("--pass2_epochs", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--draw_bbox", action="store_true",
                        help="Draw representative crop bboxes on source images")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Build dataset
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
    gt_mask = sample["mask"].numpy()
    C = int(gt_mask.max()) + 1

    # Output directory
    out_root = Path(args.output) if args.output else SAM3_RS_DIR / "outputs/build_cbank_stage2"
    out_root.mkdir(parents=True, exist_ok=True)

    # =============================================
    # Load SAM3 model (needed for embedding extraction)
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
    print(f"  Device: {device}")

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

    print(f"\nStage I + II Test")
    print(f"  Dataset: {cfg['dataset']['name']}, Images: {N}/{len(dataset)}")
    print(f"  Classes: {C}, cap: {cap_pass1}, max_per_class: {max_per_class}")
    print(f"  top_k: {top_k}, pad_ratio: {pad_ratio}")
    print(f"  Output: {out_root}")
    print(f"  Class names: {class_names}")
    print()

    # =============================================
    # Stage I: Prototype Collection (simplified)
    # =============================================
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

    print(f"  Stage I done in {time.time()-t_start:.1f}s")
    print(f"  Prototypes: {proto_valid.sum()}/{C}, total crops: {count_per_class.sum()}")

    del sum_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =============================================
    # Stage II: Representative Mining
    # =============================================
    print("\n[Stage II] Mining representative crops...")
    t_start = time.time()

    import heapq
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

    print(f"  Stage II done in {time.time()-t_start:.1f}s")

    # =============================================
    # Save representative crops
    # =============================================
    print(f"\n[Saving] Writing representative crops to {out_root} ...")

    summary_lines = [
        f"Concept Bank Stage II Representative Crops\n"
        f"Dataset: {cfg['dataset']['name']}\n"
        f"Calibration images: {N}\n"
        f"Classes: {C}, Top-K: {top_k}\n"
        f"pad_ratio={pad_ratio}, min_size={min_size}, max_size={max_size}\n"
        f"{'='*60}\n",
    ]

    # We need to re-open images to extract crops, so cache them
    img_cache = {}

    def get_img(path):
        if path not in img_cache:
            img_cache[path] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return img_cache[path]

    for class_id in range(C):
        sorted_items = sorted(heaps[class_id], key=lambda x: x[0], reverse=True)
        top = sorted_items[:top_k]

        if not top:
            cls_label = class_names.get(class_id, f"class_{class_id}")
            summary_lines.append(f"\nClass {class_id} ({cls_label}): NO representatives found\n")
            continue

        cls_label = class_names.get(class_id, f"class_{class_id}")
        cls_dir_name = f"{class_id}_{cls_label}" if cls_label else str(class_id)
        cls_dir = out_root / cls_dir_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        summary_lines.append(f"\nClass {class_id} ({cls_label}): {len(top)} representatives")
        summary_lines.append(f"  {'Rank':<6}{'Score':<10}{'Image':<30}{'BBox':<25}{'Size'}")

        for rank, (score, _, img_idx, dsid) in enumerate(top):
            try:
                s = dataset[img_idx]
                image_path = s["image_path"]
                gt = s["mask"].numpy() if torch.is_tensor(s["mask"]) else np.asarray(s["mask"])
            except Exception:
                continue

            img_np = get_img(image_path)
            img_name = Path(image_path).stem

            box = square_box_from_mask(gt, dsid, pad_ratio, min_size, max_size, rng)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            crop_np = img_np[y1:y2 + 1, x1:x2 + 1].copy()
            mask_crop = gt[y1:y2 + 1, x1:x2 + 1].copy()

            # Save crop RGB
            crop_name = f"rank_{rank}_score_{score:.3f}_img_{img_name}.png"
            Image.fromarray(crop_np).save(cls_dir / crop_name)

            # Save crop mask (white = target class)
            mask_vis = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
            mask_vis[mask_crop == dsid] = [255, 255, 255]
            mask_vis[mask_crop != dsid] = [40, 40, 40]
            mask_name = f"rank_{rank}_score_{score:.3f}_img_{img_name}_mask.png"
            Image.fromarray(mask_vis).save(cls_dir / mask_name)

            summary_lines.append(
                f"  {rank:<6}{score:<10.4f}{img_name:<30}"
                f"({x1},{y1})-({x2},{y2})      {crop_np.shape[0]}x{crop_np.shape[1]}"
            )

        # Draw bbox overview for this class
        if args.draw_bbox and top:
            overview = None
            for rank, (score, _, img_idx, dsid) in enumerate(top):
                try:
                    s = dataset[img_idx]
                    image_path = s["image_path"]
                    gt = s["mask"].numpy() if torch.is_tensor(s["mask"]) else np.asarray(s["mask"])
                except Exception:
                    continue

                img_np = get_img(image_path)
                img_name = Path(image_path).stem

                box = square_box_from_mask(gt, dsid, pad_ratio, min_size, max_size, rng)
                if box is None:
                    continue

                x1, y1, x2, y2 = box
                # Draw green box for top representative
                thickness = max(2, min(img_np.shape[0], img_np.shape[1]) // 500)
                vis = img_np.copy()
                vis[y1:y1+thickness, x1:x2+1] = [0, 255, 0]
                vis[y2:y2+thickness, x1:x2+1] = [0, 255, 0]
                vis[y1:y2+1, x1:x1+thickness] = [0, 255, 0]
                vis[y1:y2+1, x2:x2+thickness] = [0, 255, 0]

                # Label
                from PIL import ImageDraw
                pil = Image.fromarray(vis)
                draw = ImageDraw.Draw(pil)
                draw.text((x1 + 3, y1 + 3), f"#{rank} {score:.3f}", fill=(0, 255, 0))
                vis = np.asarray(pil)

                # Save per-image overview
                ov_path = cls_dir / f"overview_{img_name}.png"
                Image.fromarray(vis).save(ov_path)

    # Write summary
    summary_path = out_root / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n{'='*60}")
    print(f"Summary:")
    for class_id in range(C):
        n_reps = min(len(heaps[class_id]), top_k)
        cls_label = class_names.get(class_id, f"class_{class_id}")
        print(f"  Class {class_id} ({cls_label}): {n_reps} representatives")
    print(f"\nOutput saved to: {out_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

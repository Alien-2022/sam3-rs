"""Build Visual Prototype Bank and evaluate segmentation quality.

Usage:
    # Build from scratch + evaluate
    python eval/build_visual_prototypes.py --config eval/configs/potsdam_A2_extended.yaml --eval

    # Load pre-built bank + evaluate
    python eval/build_visual_prototypes.py --config eval/configs/potsdam_A2_extended.yaml --eval \
        --load_bank outputs/visual_prototypes/potsdam.pt

The visual prototype bank extracts per-class visual features from GT calibration
images using SAM3's vision backbone, then injects them via visual_prompt_embed
during inference to guide segmentation.
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
from pathlib import Path
SAM3_RS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAM3_RS_DIR_STR = str(SAM3_RS_DIR)
if SAM3_RS_DIR_STR not in sys.path:
    sys.path.insert(0, SAM3_RS_DIR_STR)

from segmentor_lib.visual_prototype_bank import VisualPrototypeBank
from segmentor_lib.prompts import load_prompts
from eval.datasets.potsdam import PotsdamDataset
from eval.datasets.openearthmap import OpenEarthMapDataset
from eval.datasets.loveda import LoveDADataset
from eval.metrics.seg_metrics import SegmentationMetric
from segmentor import SAM3RSSegmentor, InferenceConfig
import yaml


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Handle base config
    if "base_config" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg["base_config"])
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        # Merge: base first, then override with specific
        for key in base_cfg:
            if key not in cfg:
                cfg[key] = base_cfg[key]

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Build Visual Prototype Bank")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num_calib", type=int, default=4, help="Number of calibration images")
    parser.add_argument("--max_crops", type=int, default=50, help="Max crops per class")
    parser.add_argument("--output", type=str, default="outputs/visual_prototypes/potsdam.pt",
                        help="Output path for prototype bank")
    parser.add_argument("--load_bank", type=str, default=None,
                        help="Load SAM3 FPN prototype bank from .pt file (for geo_box/geo_point/pixel_attn modes)")
    parser.add_argument("--no_bank", action="store_true",
                        help="Run evaluation without visual prototype bank (baseline)")
    parser.add_argument("--dinov3_bank", type=str, default=None,
                        help="Load DINOv3 prototype bank from .pt file (for dinov3_geo_point mode)")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after building")
    parser.add_argument("--mode", type=str, default="geo_point",
                        choices=["geo_box", "geo_point", "pixel_attn", "dinov3_geo_point"],
                        help="Injection mode: "
                             "'geo_box' (bounding boxes from response maps), "
                             "'geo_point' (points from response maps, default), "
                             "'pixel_attn' (spatial attention on backbone_fpn[0] before PixelDecoder), "
                             "'dinov3_geo_point' (use DINOv3 features for response map + geo points)")
    parser.add_argument("--geo_threshold", type=float, default=0.3,
                        help="Threshold for response map binarization in geo_box mode (default: 0.3)")
    parser.add_argument("--geo_topk", type=int, default=10,
                        help="Number of points in geo_point mode (default: 10)")
    parser.add_argument("--geo_point_mode", type=str, default="centroid",
                        choices=["centroid", "topk"],
                        help="Point selection strategy: "
                             "'centroid' = threshold response map, find connected components, "
                             "compute weighted centroid per component (default, more robust for continuous features); "
                             "'topk' = select global top-K highest response locations (original)")
    parser.add_argument("--geo_centroid_thresh_ratio", type=float, default=0.5,
                        help="Fraction of (min+max) range as threshold for centroid mode (default: 0.7). "
                             "Higher = stricter threshold = fewer but more confident regions.")
    parser.add_argument("--geo_centroid_method", type=str, default="peak",
                        choices=["weighted", "peak", "interior_peak"],
                        help="Point selection within CC: 'weighted' (response-weighted centroid), "
                             "'peak' (max response pixel), 'interior_peak' (response × distance-to-boundary)")
    parser.add_argument("--geo_centroid_min_area", type=int, default=4,
                        help="Minimum CC area in pixels to keep (default: 4)")
    parser.add_argument("--geo_topk_suppress_r", type=int, default=0,
                        help="Spatial suppression radius for topk mode (pixels). "
                             "After selecting a point, suppress response within this radius. "
                             "0=disabled (default). Recommended: 5-10%% of feature map size.")
    parser.add_argument("--geo_centroid_area_beta", type=float, default=0.0,
                        help="Area exponent for quality-aware CC ranking in centroid mode. "
                             "Score = mean_response * area^beta. beta=0 uses mean_resp only, "
                             "beta=1 uses total_weight (original). 0.3-0.5 recommended for "
                             "balancing large moderate-confidence vs small high-confidence regions. "
                             "0=disabled (default, uses total_weight like before).")
    parser.add_argument("--geo_correlation", type=str, default="cosine",
                        choices=["cosine", "dot", "euclidean_inv", "channel_attn"],
                        help="Correlation method for response map (default: cosine)")
    parser.add_argument("--geo_presence", type=float, default=0.0,
                        help="Min max-response to add geo prompt; 0=always add (default: 0.0)")
    parser.add_argument("--geo_presence_sam3_thresh", type=float, default=0.0,
                        help="Two-pass mode: only inject geo points when SAM3 presence score < this threshold "
                             "AND DINOv3 response > geo_presence. 0=disable two-pass (default: 0.0, always inject)")
    parser.add_argument("--geo_competition", type=str, default="none",
                        choices=["none", "mean", "max", "weighted", "exclusive"],
                        help="Class competition mode for response maps (dinov3_geo_point only): "
                             "'none' = raw response maps (default); "
                             "'mean' = subtract mean of other classes; "
                             "'max' = subtract max of other classes; "
                             "'weighted' = subtract weighted mean using inter-class prototype similarity; "
                             "'exclusive' = winner-take-all: each pixel only keeps the max-class response")
    parser.add_argument("--geo_resp_fusion_alpha", type=float, default=0.0,
                        help="DINOv3 response map fusion weight (after dual-head fusion). "
                             "0=disabled (default). "
                             "final_logits = (1-alpha)*sam3_logits + alpha*response_map. "
                             "Response maps are upsampled to SAM3 output resolution and scale-normalized. "
                             "Recommended: 0.1-0.3")
    parser.add_argument("--geo_resp_fusion_thresh", type=float, default=0.5,
                        help="Response map threshold (fraction of scale) for masking before fusion. "
                             "Higher = only high-confidence response pixels participate. Default: 0.5")
    parser.add_argument("--geo_only_classes", type=str, default=None,
                        help="Comma-separated class IDs to inject geo prompts for (others get text-only). "
                             "E.g. '3,4' for grass+tree only. Useful for per-class diagnostic. (default: all)")
    parser.add_argument("--geo_only_mode", action="store_true", default=False,
                        help="Geo-only mode (requires geo_presence_sam3_thresh > 0): "
                             "When SAM3 presence < threshold + DINOv3 confident → use point prompt ONLY (no text). "
                             "When both signals low → skip class entirely. "
                             "Default: text + geo point for two-pass rerun.")
    parser.add_argument("--geo_fpn_level", type=int, default=-1,
                        help="FPN level for response map (single-level): -1=72x72, -2=144x144, -3=288x288 (default: -1)")
    parser.add_argument("--feat_level", type=int, default=-1,
                        help="FPN level for prototype extraction: -1=72x72(semantic), -2=144x144, 0=288x288(fine) (default: -1)")
    parser.add_argument("--feat_levels", type=str, default=None,
                        help="Comma-separated FPN levels for multi-level extraction, e.g. '-1,-2,0'. Overrides --feat_level")
    parser.add_argument("--feat_source", type=str, default="backbone_fpn",
                        help="Feature source for prototype construction: 'backbone_fpn' (raw FPN), "
                             "'pixel_decoder' (multi-scale fusion, 288x288), "
                             "'dinov3_sat' (DINOv3 ViT-L/16 SAT, 1024-dim) (default: backbone_fpn)")
    parser.add_argument("--dinov3_weights", type=str, default=None,
                        help="Path to DINOv3 SAT model directory (for dinov3_sat feat_source or dinov3_geo_point mode). "
                             "Default: weights/dinov3/sat_vit_L16")
    parser.add_argument("--dinov3_input_size", type=int, default=1008,
                        help="Input resolution for DINOv3 (must be multiple of 16). "
                             "1008 → 63x63 patches (default: 1008)")
    parser.add_argument("--num_clusters", type=int, default=1,
                        help="Max sub-prototypes per class via adaptive K-means (1=weighted avg, >1=auto-select K in [1,num_clusters] by silhouette score) (default: 1)")
    parser.add_argument("--label_offset", type=int, default=0,
                        help="Offset between class_id and GT label values. "
                             "For OpenEarthMap (GT labels 0=no-data, 1-8=classes), set to 1. "
                             "For Potsdam (GT labels 0-5), use default 0. (default: 0)")
    parser.add_argument("--max_eval_images", type=int, default=0,
                        help="Max number of images to evaluate (0=all). For quick validation. (default: 0)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for batch inference (requires dinov3_geo_point mode, "
                             "no sliding window). 1=single image (default). Recommended: 4-8.")
    parser.add_argument("--fusion", type=str, default="mean",
                        choices=["mean"],
                        help="Multi-level fusion method: 'mean' (response-level average)")
    parser.add_argument("--skip_bg_idx", type=int, default=None,
                        help="Skip geo prompt injection for this class_id (e.g. 0 for background). "
                             "Background is vague; text-only prompt may work better.")
    parser.add_argument("--pixel_attn_alpha", type=float, default=10.0,
                        help="Sharpness of sigmoid for pixel_attn mode (default: 10.0). "
                             "Higher = sharper transition, lower = softer attention.")
    parser.add_argument("--pixel_attn_threshold", type=float, default=0.3,
                        help="Center of sigmoid for pixel_attn mode (default: 0.3). "
                             "Response values above this get amplified, below get suppressed.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Build segmentor
    seg_cfg = cfg["segmentor"].copy()
    # Resolve relative paths
    seg_cfg["checkpoint_path"] = str(SAM3_RS_DIR / seg_cfg["checkpoint_path"])
    seg_cfg["bpe_path"] = str(SAM3_RS_DIR / "sam3/assets" / seg_cfg["bpe_path"])
    seg_cfg["prompts_file"] = str(SAM3_RS_DIR / "eval" / "datasets" / seg_cfg["prompts_file"])

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
        debug_memory=seg_cfg.get("debug_memory", False),
        debug_log_file=seg_cfg.get("debug_log_file", None),
        **infer_kwargs,
    )
    segmentor = SAM3RSSegmentor(infer_cfg)

    device = segmentor.device
    num_classes = segmentor.num_classes

    # Build dataset (select loader based on dataset name)
    ds_cfg = cfg["dataset"]
    ds_name = ds_cfg.get("name", "potsdam")
    ds_kwargs = dict(
        data_root=str(SAM3_RS_DIR / ds_cfg["data_root"]),
        img_dir=ds_cfg["img_dir"],
        mask_dir=ds_cfg["mask_dir"],
        cls_file=str(SAM3_RS_DIR / "eval" / "datasets" / ds_cfg["cls_file"]),
        ignore_index=ds_cfg.get("ignore_index", 255),
        reduce_zero_label=ds_cfg.get("reduce_zero_label", False),
    )
    if ds_name == "openearthmap":
        dataset = OpenEarthMapDataset(**ds_kwargs)
    elif ds_name == "loveda":
        dataset = LoveDADataset(**ds_kwargs)
    else:
        dataset = PotsdamDataset(**ds_kwargs)

    # Build or load visual prototype bank
    if args.no_bank:
        # Baseline: no visual prototype bank
        print("\nRunning baseline (no visual prototype bank)...")
        prototypes = None
        is_multi = False
    elif args.load_bank:
        print(f"\nLoading Visual Prototype Bank from {args.load_bank}...")
        prototypes = VisualPrototypeBank.load(args.load_bank, device)
        is_multi = isinstance(next(iter(prototypes.values())), dict)
        print(f"Loaded {len(prototypes)} class prototypes (multi_level={is_multi})")
    elif args.dinov3_bank:
        # Load DINOv3 prototype bank (standalone .pt with multi-layer features)
        print(f"\nLoading DINOv3 Prototype Bank from {args.dinov3_bank}...")
        bank_data = torch.load(args.dinov3_bank, map_location="cpu", weights_only=False)
        prototypes = bank_data["prototypes"]
        is_multi = False
        bank_config = bank_data.get("config", {})
        dinov3_bank_layers = bank_config.get("layers", None)
        dinov3_bank_inter_sim = bank_data.get("inter_class_sim", None)
        print(f"Loaded {len(prototypes)} class prototypes (feat_dim={bank_config.get('feat_dim', '?')}, "
              f"layers={dinov3_bank_layers})")
    else:
        save_path = args.output
        if save_path and not os.path.isabs(save_path):
            save_path = str(os.path.join(SAM3_RS_DIR, save_path))

        feat_levels_arg = None
        if args.feat_levels is not None:
            feat_levels_arg = [int(x.strip()) for x in args.feat_levels.split(",")]

        # Resolve DINOv3 weights path
        dinov3_weights = args.dinov3_weights
        if args.feat_source == "dinov3_sat" and dinov3_weights is None:
            dinov3_weights = str(os.path.join(SAM3_RS_DIR, "weights", "dinov3", "sat_vit_L16", "model.safetensors"))
        if dinov3_weights is not None and not os.path.isabs(dinov3_weights):
            dinov3_weights = str(os.path.join(SAM3_RS_DIR, dinov3_weights))

        # For dinov3_sat, feat_level/feat_levels are ignored (single output level 0)
        builder = VisualPrototypeBank(
            processor=segmentor.processor,
            device=device,
            num_classes=num_classes,
            max_crops_per_class=args.max_crops,
            crop_size=2048,
            stride=1024,
            feat_level=args.feat_level,
            feat_levels=feat_levels_arg,
            feat_source=args.feat_source,
            num_clusters=args.num_clusters,
            label_offset=args.label_offset,
            dinov3_weights=dinov3_weights,
            dinov3_input_size=args.dinov3_input_size,
        )

        prototypes = builder.build(
            dataset=dataset,
            num_calib=args.num_calib,
            save_path=save_path,
        )

        is_multi = isinstance(next(iter(prototypes.values())), dict)

    # For geo modes, pass multi-level bank directly (core.py handles fusion).
    # For non-geo modes, flatten to single level.
    if is_multi and args.eval and args.mode not in ("geo_box", "geo_point", "pixel_attn", "dinov3_geo_point"):
        selected_level = int(args.feat_levels.split(",")[0].strip()) if args.feat_levels else args.feat_level
        flat_protos = {}
        for k, v in prototypes.items():
            flat_protos[k] = v.get(selected_level, next(iter(v.values())))
        print(f"  Flattened multi-level bank to level {selected_level} for non-geo mode")
        prototypes = flat_protos

    # Apply visual prototype bank to engine
    if prototypes is not None:
        print("\nApplying Visual Prototype Bank to segmentor...")
        segmentor.engine.set_visual_prototype_bank(
            prototypes, mode=args.mode,
            geo_threshold=args.geo_threshold, geo_topk=args.geo_topk,
            geo_correlation=args.geo_correlation, geo_presence_threshold=args.geo_presence,
            geo_fpn_level=args.geo_fpn_level,
            geo_fusion=args.fusion,
            geo_skip_bg_idx=args.skip_bg_idx,
            pixel_attn_alpha=args.pixel_attn_alpha,
            pixel_attn_threshold=args.pixel_attn_threshold,
            geo_point_mode=args.geo_point_mode,
            geo_centroid_thresh_ratio=args.geo_centroid_thresh_ratio,
            geo_centroid_method=args.geo_centroid_method,
            geo_centroid_min_area=args.geo_centroid_min_area,
            geo_topk_suppress_r=args.geo_topk_suppress_r,
            geo_centroid_area_beta=args.geo_centroid_area_beta,
            dinov3_weights=args.dinov3_weights,
            dinov3_input_size=args.dinov3_input_size,
            geo_presence_sam3_thresh=args.geo_presence_sam3_thresh,
            dinov3_layers=dinov3_bank_layers if args.dinov3_bank else None,
            geo_competition=args.geo_competition,
            inter_class_sim=dinov3_bank_inter_sim if args.dinov3_bank else None,
            geo_only_classes=[int(x.strip()) for x in args.geo_only_classes.split(",")] if args.geo_only_classes else None,
            geo_only_mode=args.geo_only_mode,
            geo_resp_fusion_alpha=args.geo_resp_fusion_alpha,
            geo_resp_fusion_thresh=args.geo_resp_fusion_thresh,
        )
        print(f"  Classes with prototypes: {len(prototypes)}/{num_classes}, "
              f"mode={args.mode}, multi_level={is_multi}"
              f"{f', fusion={args.fusion}' if is_multi else ''}")
    else:
        print("\nNo visual prototype bank (baseline mode)")

    if args.eval:
        print("\n" + "=" * 60)
        print("Running evaluation...")
        print("=" * 60)

        # Use A2 config: 12 prompts with fuse_prompts_to_classes
        # (visual prototypes don't change prompt structure)
        ds_cfg_eval = cfg["dataset"]
        ignore_index = ds_cfg_eval.get("ignore_index", 255)
        use_prompted_background = seg_cfg.get("use_prompted_background", False)

        metric = SegmentationMetric(
            num_classes=num_classes,
            ignore_index=ignore_index,
            use_prompted_background=use_prompted_background,
        )

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        max_eval = args.max_eval_images if args.max_eval_images > 0 else len(loader)

        # Collect all image paths and gt masks for batch inference
        all_image_paths = []
        all_gt_masks = []
        for i, batch in enumerate(tqdm(loader, desc="Loading", total=min(len(loader), max_eval))):
            if i >= max_eval:
                break
            all_image_paths.append(batch["image_path"][0])
            all_gt_masks.append(batch["mask"][0])

        use_batch = (args.batch_size > 1
                     and args.mode == "dinov3_geo_point"
                     and seg_cfg.get("slide_crop_size", 0) == 0)
        if use_batch:
            print(f"  Using batch inference: batch_size={args.batch_size}, "
                  f"{len(all_image_paths)} images, ~{len(all_image_paths)//args.batch_size + 1} batches")

        for batch_start in tqdm(range(0, len(all_image_paths), args.batch_size),
                                desc="Evaluating", total=(len(all_image_paths) + args.batch_size - 1) // args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(all_image_paths))
            batch_paths = all_image_paths[batch_start:batch_end]

            if use_batch and len(batch_paths) > 1:
                try:
                    results = segmentor.predict_batch(batch_paths)
                    for b, result in enumerate(results):
                        pred_np = result.seg_pred.cpu().numpy()
                        gt_np = all_gt_masks[batch_start + b].numpy()
                        metric.update(pred_np, gt_np)
                except Exception as e:
                    print(f"  Batch error on {batch_paths[0]}..{batch_paths[-1]}: {e}")
                    import traceback
                    traceback.print_exc()
                    for b in range(len(batch_paths)):
                        try:
                            result = segmentor.predict_single(batch_paths[b])
                            pred_np = result.seg_pred.cpu().numpy()
                            gt_np = all_gt_masks[batch_start + b].numpy()
                            metric.update(pred_np, gt_np)
                        except Exception as e2:
                            print(f"  Fallback error on {batch_paths[b]}: {e2}")
            else:
                for b in range(len(batch_paths)):
                    image_path = batch_paths[b]
                    try:
                        result = segmentor.predict_single(image_path)
                        pred_np = result.seg_pred.cpu().numpy()
                        gt_np = all_gt_masks[batch_start + b].numpy()
                        metric.update(pred_np, gt_np)
                    except Exception as e:
                        print(f"  Error on {image_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        eval_result = metric.compute()
        class_iou = metric.per_class_iou()

        # Build class name list
        prompts = segmentor.prompts
        class_names = []
        for ci in range(num_classes):
            for pi, pid in enumerate(prompts["indices"]):
                if pid == ci:
                    class_names.append(prompts["names"][pi])
                    break
            else:
                class_names.append(f"class_{ci}")

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Evaluation Results")
        print(f"{'=' * 60}")
        print(f"  mIoU  = {eval_result['mIoU']:.4f}")
        print(f"  mAcc  = {eval_result['mAcc']:.4f}")
        print(f"  aAcc  = {eval_result['aAcc']:.4f}")
        print(f"\nPer-class IoU:")
        for ci in range(num_classes):
            print(f"  Class {ci} ({class_names[ci]:<20s}): IoU={class_iou[ci]:.4f}")


if __name__ == "__main__":
    # potsdam
    # python eval/build_visual_prototypes.py --config eval/configs/potsdam_A2_extended.yaml --output outputs/visual_prototypes/potsdam_fpn72_new.pt --geo_correlation cosine --num_calib 4 --max_crops 200 --feat_levels 2
    # python eval/build_visual_prototypes.py --config eval/configs/potsdam_A2_extended.yaml --load_bank outputs/visual_prototypes/potsdam.pt --eval

    # build potsdam with DINOv3 SAT
    # python workspace/core/sam3-rs/eval/build_visual_prototypes.py --config workspace/core/sam3-rs/eval/configs/potsdam_A2_extended.yaml --feat_source dinov3_sat --dinov3_input_size 2048 --output workspace/core/sam3-rs/outputs/visual_prototypes/potsdam_dinov3.pt --dinov3_weights weights/dinov3/sat_vit_L16 --num_calib 4 --max_crops 200
    # eval potsdam
    # python workspace/core/sam3-rs/eval/build_visual_prototypes.py --config workspace/core/sam3-rs/eval/configs/potsdam_A2_extended.yaml --dinov3_bank workspace/core/sam3-rs/outputs/visual_prototypes/potsdam_dinov3.pt --mode dinov3_geo_point --eval --dinov3_weights workspace/core/sam3-rs/weights/dinov3/sat_vit_L16 --dinov3_input_size 2048 --geo_topk 3 --geo_point_mode centroid --geo_presence 0.7 --geo_presence_sam3_thresh 0.5

    # openearthmap
    # python eval/build_visual_prototypes.py --config eval/configs/openearthmap_A2_extended.yaml --load_bank outputs/visual_prototypes/openearthmap_cluster.pt  --geo_fpn_level 2 --geo_topk 1 --eval

    # openearthmap with DINOv3 SAT
    # python eval/build_visual_prototypes.py --config eval/configs/openearthmap_A2_extended.yaml --feat_source dinov3_sat --dinov3_input_size 544 --label_offset 1 --output outputs/visual_prototypes/openearthmap_dinov3.pt --dinov3_weights weights/dinov3/sat_vit_L16 --num_calib 4 --max_crops 50

    # build loveda with DINOv3 SAT
    # python assistant/build_dinov3_prototypes.py \
    # --config workspace/core/sam3-rs/eval/configs/loveda_A2_extended.yaml \
    # --output workspace/core/sam3-rs/outputs/visual_prototypes/loveda_dinov3.pt \
    # --input_size 1024 \
    # --crop_size 1024 --stride 1024 \
    # --label_offset 0 \
    # --num_calib 200 --max_crops 500

    # run loveda with vpb
    # python workspace/core/sam3-rs/eval/build_visual_prototypes.py     --config workspace/core/sam3-rs/eval/configs/loveda_A2_extended.yaml     --dinov3_bank workspace/core/sam3-rs/outputs/visual_prototypes/loveda_dinov3_v2.pt     --mode dinov3_geo_point     --eval     --dinov3_weights workspace/core/sam3-rs/weights/dinov3/sat_vit_L16     --dinov3_input_size 1024     --geo_topk 3     --geo_point_mode topk --geo_competition weighted    --geo_presence_sam3_thresh 0.7  --geo_presence 0.2 --geo_only_classes 0,4,5,6 --max_eval_images 0 --batch_size 4
    main()

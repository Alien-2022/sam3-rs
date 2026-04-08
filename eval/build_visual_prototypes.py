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
                        help="Load pre-built bank instead of building")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after building")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Injection strength: prototype scaled to text_norm * alpha (default: 1.0)")
    parser.add_argument("--mode", type=str, default="vision",
                        choices=["lang", "vision", "concat", "replace_last", "geo_box", "geo_point"],
                        help="Injection mode: 'lang' (add to language_features), "
                             "'vision' (add to backbone_fpn), "
                             "'concat' (append prototype as extra token), "
                             "'replace_last' (replace last token with scaled prototype), "
                             "'geo_box' (bounding boxes from response maps), "
                             "'geo_point' (top-K points from response maps)")
    parser.add_argument("--geo_threshold", type=float, default=0.3,
                        help="Threshold for response map binarization in geo_box mode (default: 0.3)")
    parser.add_argument("--geo_topk", type=int, default=10,
                        help="Number of top-K points in geo_point mode (default: 10)")
    parser.add_argument("--geo_correlation", type=str, default="cosine",
                        choices=["cosine", "dot", "euclidean_inv", "channel_attn"],
                        help="Correlation method for response map (default: cosine)")
    parser.add_argument("--geo_presence", type=float, default=0.0,
                        help="Min max-response to add geo prompt; 0=always add (default: 0.0)")
    parser.add_argument("--geo_fpn_level", type=int, default=-1,
                        help="FPN level for response map: -1=72x72, -2=144x144, -3=288x288 (default: -1)")
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

    # Build dataset
    ds_cfg = cfg["dataset"]
    dataset = PotsdamDataset(
        data_root=str(SAM3_RS_DIR / ds_cfg["data_root"]),
        img_dir=ds_cfg["img_dir"],
        mask_dir=ds_cfg["mask_dir"],
        cls_file=str(SAM3_RS_DIR / "eval" / "datasets" / ds_cfg["cls_file"]),
        ignore_index=ds_cfg.get("ignore_index", 255),
        reduce_zero_label=ds_cfg.get("reduce_zero_label", False),
    )

    # Build or load visual prototype bank
    if args.load_bank:
        print(f"\nLoading Visual Prototype Bank from {args.load_bank}...")
        prototypes = VisualPrototypeBank.load(args.load_bank, device)
        print(f"Loaded {len(prototypes)} class prototypes")
    else:
        save_path = args.output
        if save_path and not os.path.isabs(save_path):
            save_path = str(os.path.join(SAM3_RS_DIR, save_path))

        builder = VisualPrototypeBank(
            processor=segmentor.processor,
            device=device,
            num_classes=num_classes,
            max_crops_per_class=args.max_crops,
            crop_size=2048,
            stride=1024
        )

        prototypes = builder.build(
            dataset=dataset,
            num_calib=args.num_calib,
            save_path=save_path,
        )

    # Apply visual prototype bank to engine
    print("\nApplying Visual Prototype Bank to segmentor...")
    segmentor.engine.set_visual_prototype_bank(
        prototypes, alpha=args.alpha, mode=args.mode,
        geo_threshold=args.geo_threshold, geo_topk=args.geo_topk,
        geo_correlation=args.geo_correlation, geo_presence_threshold=args.geo_presence,
        geo_fpn_level=args.geo_fpn_level,
    )
    print(f"  Classes with prototypes: {len(prototypes)}/{num_classes}")
    for class_id, proto in prototypes.items():
        print(f"    Class {class_id}: shape={proto.shape}, "
              f"norm={proto.norm().item():.4f}, "
              f"mean={proto.mean().item():.4f}")

    # === Diagnostic: compare text_features vs visual prototype scales ===
    print("\n" + "=" * 60)
    print(f"Scale Diagnostic: text_features vs visual prototypes (alpha={args.alpha})")
    print("=" * 60)
    if segmentor.engine.text_features_cache is not None:
        seen_classes = set()
        for pidx, cached in segmentor.engine.text_features_cache.items():
            class_id = segmentor.engine.prompts["indices"][pidx]
            if class_id in seen_classes:
                continue
            seen_classes.add(class_id)
            lf = cached["language_features"].to(device)
            proto = prototypes.get(class_id)
            print(f"  Prompt[{pidx}] '{segmentor.engine.prompts['names'][pidx]}' -> Class {class_id}:")
            print(f"    language_features: shape={lf.shape}, dtype={lf.dtype}")
            text_norm = lf.norm().item()
            print(f"      norm={text_norm:.4f}, mean={lf.mean().item():.6f}, "
                  f"abs_mean={lf.abs().mean().item():.6f}, "
                  f"min={lf.min().item():.4f}, max={lf.max().item():.4f}")
            if proto is not None:
                proto_dev = proto.to(device)
                proto_norm = proto_dev.norm().item()
                print(f"    visual prototype:  shape={proto_dev.shape}, dtype={proto_dev.dtype}")
                print(f"      norm={proto_norm:.4f}, mean={proto_dev.mean().item():.6f}, "
                      f"abs_mean={proto_dev.abs().mean().item():.6f}, "
                      f"min={proto_dev.min().item():.4f}, max={proto_dev.max().item():.4f}")
                scaled_norm = args.alpha * text_norm * proto_norm
                print(f"    raw ratio (proto/text): {proto_norm / text_norm:.6f}")
                print(f"    scaled injection norm: alpha * text_norm * proto_norm = {args.alpha} * {text_norm:.1f} * {proto_norm:.1f} = {scaled_norm:.2f}")
                print(f"    injection/text ratio: {scaled_norm / text_norm:.4f}")
                # Pool language features across token dim for comparison
                lf_pooled = lf.mean(dim=0, keepdim=True)  # [1, 1, 256]
                print(f"    [pooled text] norm={lf_pooled.norm().item():.4f}")
                print(f"    cosine_sim(proto, pooled_text): {torch.nn.functional.cosine_similarity(lf_pooled.flatten().unsqueeze(0), proto_dev.flatten().unsqueeze(0)).item():.4f}")
    else:
        print("  WARNING: text_features_cache not available (not pre-computed)")
    print("=" * 60)

    if args.eval:
        print("\n" + "=" * 60)
        print("Running evaluation with Visual Prototypes...")
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

        for batch in tqdm(loader, desc="Evaluating"):
            image_path = batch["image_path"][0]
            gt_mask = batch["mask"][0]

            try:
                result = segmentor.predict_single(image_path)
                pred_np = result.seg_pred.cpu().numpy()
                gt_np = gt_mask.numpy()
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
    # python eval/build_visual_prototypes.py --config eval/configs/potsdam_A2_extended.yaml --load_bank outputs/visual_prototypes/potsdam.pt --eval
    main()

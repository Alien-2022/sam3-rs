"""Build Concept Bank for SAM3-RS open vocabulary segmentation.

Usage:
    # Build with GT masks on Potsdam (24 calibration images)
    python eval/build_concept_bank.py --config eval/configs/potsdam.yaml --num_calib 24

    # Build with custom parameters
    python eval/build_concept_bank.py --config eval/configs/potsdam.yaml --num_calib 24 \
        --tau_w 0.15 --top_k 10 --pad_ratio 0.05

    # Build and save
    python eval/build_concept_bank.py --config eval/configs/potsdam.yaml --num_calib 24 \
        --output outputs/concept_bank/potsdam.pt

    # Load saved bank and run evaluation
    python eval/build_concept_bank.py --config eval/configs/potsdam.yaml \
        --load_bank outputs/concept_bank/potsdam.pt --eval
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Repository roots
EVAL_DIR = Path(__file__).resolve().parent
SAM3_RS_DIR = EVAL_DIR.parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(SAM3_RS_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_RS_DIR))

from segmentor import SAM3RSSegmentor, InferenceConfig  # noqa: E402
from segmentor_lib.experimental.concept_bank import ConceptBankBuilder  # noqa: E402
from eval.datasets import DATASET_REGISTRY  # noqa: E402
from eval.metrics.seg_metrics import SegmentationMetric  # noqa: E402


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cfg_name: str) -> Dict[str, Any]:
    cfg_path = EVAL_DIR / "configs" / cfg_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    if cfg_raw is None:
        raise ValueError("Empty config file.")

    base_cfg: Dict[str, Any] = {}
    if "base_config" in cfg_raw and cfg_raw["base_config"]:
        base_path = cfg_path.parent / cfg_raw["base_config"]
        if base_path.exists():
            with open(base_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    base_cfg = loaded

    cfg = _deep_merge(base_cfg, {k: v for k, v in cfg_raw.items() if k != "base_config"})

    dataset = cfg.get("dataset", {})
    data_root = dataset.get("data_root", ".")
    data_root = SAM3_RS_DIR / data_root
    dataset["data_root"] = data_root
    dataset["img_dir"] = data_root / dataset["img_dir"]
    dataset["mask_dir"] = data_root / dataset["mask_dir"]
    dataset["cls_file"] = EVAL_DIR / "datasets" / dataset["cls_file"]

    segmentor = cfg.get("segmentor", {})
    segmentor["checkpoint_path"] = SAM3_RS_DIR / segmentor["checkpoint_path"]
    segmentor["bpe_path"] = SAM3_RS_DIR / "sam3/assets" / segmentor["bpe_path"]
    segmentor["prompts_file"] = dataset["cls_file"]

    return cfg


def build_dataset(cfg: Dict[str, Any]) -> torch.utils.data.Dataset:
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"].lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](
        data_root=str(ds_cfg["data_root"]),
        img_dir=str(ds_cfg["img_dir"]),
        mask_dir=str(ds_cfg["mask_dir"]),
        cls_file=str(ds_cfg["cls_file"]),
        ignore_index=ds_cfg.get("ignore_index", 255),
        reduce_zero_label=ds_cfg.get("reduce_zero_label", False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Concept Bank for SAM3-RS")
    parser.add_argument("--config", required=True, help="Dataset config (e.g., potsdam.yaml)")
    parser.add_argument("--num_calib", type=int, default=24, help="Number of calibration images")
    parser.add_argument("--tau_w", type=float, default=0.15, help="Temperature for softmax fusion")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K representative crops per class")
    parser.add_argument("--pad_ratio", type=float, default=0.05, help="Crop padding ratio")
    parser.add_argument("--min_crop_size", type=int, default=128, help="Minimum crop size")
    parser.add_argument("--max_crop_size", type=int, default=1024, help="Maximum crop size")
    parser.add_argument("--pass2_epochs", type=int, default=3, help="Stage II epochs")
    parser.add_argument("--output", type=str, default=None, help="Path to save concept bank (.pt)")
    parser.add_argument("--load_bank", type=str, default=None, help="Load saved bank instead of building")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after building/loading bank")
    parser.add_argument("--debug", action="store_true", help="Debug: test single crop inference and shapes")
    args = parser.parse_args()
    return args

 
def main():
    args = parse_args()
    cfg = load_config(args.config)

    print(f"Config: {args.config}")
    print(f"Dataset: {cfg['dataset']['name']}")
    print(f"Calibration images: {args.num_calib}")

    # Build dataset
    dataset = build_dataset(cfg)
    print(f"Dataset size: {len(dataset)}")

    # Build segmentor (needed for processor and prompts)
    seg_cfg = cfg["segmentor"]
    # Convert string keys to int for threshold dicts
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
    processor = segmentor.processor
    prompts = segmentor.prompts

    if args.load_bank:
        # Load pre-built bank
        print(f"\nLoading Concept Bank from {args.load_bank}...")
        bank_data = ConceptBankBuilder.load(args.load_bank, device)
        bank_cache = bank_data["cache"]
        class_names = bank_data.get("class_names", [])
        print(f"Loaded {len(bank_cache)} cache entries")
        if class_names:
            print(f"  Class names: {class_names}")
    else:
        # Build concept bank
        save_path = args.output
        if save_path and not os.path.isabs(save_path):
            save_path = str(SAM3_RS_DIR / save_path)

        builder = ConceptBankBuilder(
            processor=processor,
            engine=segmentor.engine,
            segmentor=segmentor,
            device=device,
            prompts=prompts,
            pad_ratio=args.pad_ratio,
            min_crop_size=args.min_crop_size,
            max_crop_size=args.max_crop_size,
            top_k_per_class=args.top_k,
            tau_w=args.tau_w,
            pass2_max_epochs=args.pass2_epochs,
        )

        if args.debug:
            from segmentor_lib.experimental.concept_bank import soft_dice_score
            print("\n=== DEBUG: Testing sliding window inference for prompt scoring ===")
            for ci, sample in enumerate(dataset):
                if ci >= 1:
                    break
                image_path = sample["image_path"]
                gt_mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else np.asarray(sample["mask"])
                print(f"  Image {ci}: GT shape={gt_mask.shape}, "
                      f"classes={np.unique(gt_mask)}")

                # Test on a small resized version
                img = Image.open(image_path).convert("RGB")
                H, W = img.height, img.width
                max_size = 1024
                if H > max_size or W > max_size:
                    scale = max_size / max(H, W)
                    new_W, new_H = int(W * scale), int(H * scale)
                    img_small = img.resize((new_W, new_H), Image.BILINEAR)
                    print(f"  Resized: {W}x{H} -> {new_W}x{new_H}")
                else:
                    img_small = img
                    new_H, new_W = H, W

                # Run full inference
                seg_logits, _, _, _, _ = segmentor.engine.inference_batch_view(
                    [img_small], detailed=False
                )
                print(f"  seg_logits shape: {seg_logits.shape}")
                # seg_logits: [1, num_prompts, new_H, new_W]
                for pi in range(seg_logits.shape[1]):
                    class_id = prompts["indices"][pi]
                    prompt_name = prompts["names"][pi]
                    gt_class = (gt_mask == class_id).astype(np.float32)
                    if gt_class.sum() == 0:
                        print(f"    Prompt {pi} '{prompt_name}' (class {class_id}): "
                              f"no GT pixels, skipped")
                        continue
                    prob = seg_logits[0, pi].sigmoid().cpu()
                    dice = soft_dice_score(prob, torch.from_numpy(gt_class).float())
                    print(f"    Prompt {pi} '{prompt_name}' (class {class_id}): "
                          f"dice={dice:.4f}, prob_mean={prob.mean():.4f}, "
                          f"prob_max={prob.max():.4f}")
                break
            print("=== DEBUG END ===\n")
            return

        bank_cache = builder.build_and_save(
            dataset=dataset,
            num_calib=args.num_calib,
            save_path=save_path,
        )
        class_names = getattr(builder, '_class_names', [])

    # ======================================================
    # Apply concept bank to segmentor
    # ======================================================
    # The bank has one fused feature per class (keyed by class_id).
    # We restructure the segmentor to use num_prompts == num_classes
    # so that fuse_prompts_to_classes is skipped during inference.
    print("\nApplying Concept Bank to segmentor...")
    num_classes = segmentor.num_classes
    print(f"  Original: {segmentor.num_prompts} prompts -> {num_classes} classes")

    if not class_names:
        # Fallback: generate class names from original prompts
        class_names = []
        for ci in range(num_classes):
            for pi, pid in enumerate(prompts["indices"]):
                if pid == ci:
                    class_names.append(prompts["names"][pi])
                    break
            else:
                class_names.append(f"class_{ci}")

    assert len(class_names) == num_classes, (
        f"class_names mismatch: {len(class_names)} vs {num_classes}"
    )

    # Override segmentor prompts: one per class
    segmentor.prompts = {
        "names": class_names,
        "indices": list(range(num_classes)),
        "mapping": {name: i for i, name in enumerate(class_names)},
    }
    segmentor.num_prompts = num_classes
    segmentor.query_indices = torch.arange(num_classes, dtype=torch.int64, device=segmentor.device)

    # Also update engine's num_prompts (it allocates logits based on this)
    segmentor.engine.num_prompts = num_classes
    segmentor.engine.prompts = segmentor.prompts

    # Override cache: map class_id (0..C-1) as prompt_idx keys
    # since now prompt_idx == class_id
    bank_cache_by_idx = {ci: features for ci, features in bank_cache.items() if ci < num_classes}
    segmentor.engine.text_features_cache = bank_cache_by_idx

    print(f"  Restructured: {num_classes} prompts (one per class, no synonym fusion needed)")
    print(f"  Concept bank entries: {len(bank_cache_by_idx)}")
    print(f"  Class names: {class_names}")

    if args.eval:
        print("\n" + "=" * 60)
        print("Running evaluation with Concept Bank...")
        print("=" * 60)

        eval_result = run_evaluation(cfg, segmentor, dataset)
        print(f"\nEvaluation result: mIoU={eval_result['mIoU']:.4f}")


def run_evaluation(cfg, segmentor, dataset):
    """Simple evaluation loop."""
    from torch.utils.data import DataLoader

    ds_cfg = cfg["dataset"]
    num_classes = segmentor.num_classes
    ignore_index = ds_cfg.get("ignore_index", 255)
    use_prompted_background = cfg["segmentor"].get("use_prompted_background", False)

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
            continue

    scores = metric.compute()
    return scores


if __name__ == "__main__":
    # python eval/build_concept_bank.py --config potsdam_A2_extended.yaml --eval --num_calib 4
    # python eval/build_concept_bank.py --config potsdam_A2_extended.yaml --load_bank outputs/concept_bank/potsdam.pt --eval
    main()

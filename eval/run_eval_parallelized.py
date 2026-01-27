"""Lightweight evaluation entrypoint for SAM3-RS (no mmengine dependency)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, List

import torch
from PIL import Image
from torch.utils.data import DataLoader
import yaml

# Repository roots
EVAL_DIR = Path(__file__).resolve().parent
SAM3_RS_DIR = EVAL_DIR.parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # .../workspace
if str(SAM3_RS_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_RS_DIR))

from segmentor import SAM3RSSegmentor, InferenceConfig  # type: ignore
from eval.datasets.loveda import LoveDADataset
from eval.metrics.seg_metrics import SegmentationMetric

# Optional workspace-level registry helpers (for NAS paths)
try:
    from workspace.scripts.get_data import get_data_path  # type: ignore
    from workspace.scripts.get_weights import get_weight_path  # type: ignore
except Exception:
    get_data_path = None
    get_weight_path = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cfg_path: str) -> Dict[str, Any]:
    # read the config file
    cfg_path = EVAL_DIR / "configs" / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    if cfg_raw is None:
        raise ValueError("Empty config file.")

    # 1. Process dataset section
    # base config merging
    base_cfg: Dict[str, Any] = {}
    if "base_config" in cfg_raw and cfg_raw["base_config"]:
        base_cfg_path = cfg_raw["base_config"]
        base_path_resolved = cfg_path.parent / base_cfg_path
        if base_path_resolved:
            with open(base_path_resolved, "r", encoding="utf-8") as f:
                loaded_base = yaml.safe_load(f)
                if loaded_base:
                    base_cfg = loaded_base

    cfg = _deep_merge(
        base_cfg, {k: v for k, v in cfg_raw.items() if k != "base_config"}
    )

    # If data_key provided, use the data path on NAS
    dataset = cfg.get("dataset", {})
    data_key = dataset["data_key"]
    if data_key and get_data_path is not None:
        try:
            data_root = Path(get_data_path(data_key))
            dataset["data_root"] = data_root
            dataset["img_dir"] = data_root / dataset["img_dir"]
            dataset["mask_dir"] = data_root / dataset["mask_dir"]
        except Exception:
            pass
    else:
        data_root = dataset.get("data_root", ".")
        data_root = EVAL_DIR / data_root
        dataset["data_root"] = EVAL_DIR / data_root
        dataset["img_dir"] = data_root / dataset["img_dir"]
        dataset["mask_dir"] = data_root / dataset["mask_dir"]
    dataset["cls_file"] = EVAL_DIR / "datasets" / dataset["cls_file"]

    # 2. Process segmentor section
    segmentor = cfg.get("segmentor", {})
    weight_key = segmentor.get("weight_key")
    if weight_key and get_weight_path is not None:
        try:
            weight_path = get_weight_path(weight_key)
            segmentor["checkpoint_path"] = weight_path
        except Exception:
            pass
    segmentor["bpe_path"] = SAM3_RS_DIR / "sam3/assets" / segmentor["bpe_path"]
    # Use dataset class list as prompts file
    segmentor["prompts_file"] = dataset["cls_file"]

    # 3. Process output section
    output = cfg.get("output", {})
    output["metrics_json"] = SAM3_RS_DIR / output.get(
        "metrics_json", "outputs/metrics.json"
    )
    if output.get("save_pred_dir") is not None:
        output["save_pred_dir"] = SAM3_RS_DIR / output["save_pred_dir"]
        os.makedirs(output["save_pred_dir"], exist_ok=True)

    return cfg


def build_segmentor(seg_cfg: Dict[str, Any]) -> SAM3RSSegmentor:
    infer_kwargs = dict(seg_cfg)
    # Drop helper-only keys
    infer_kwargs.pop("weight_key", None)
    infer_kwargs.pop("checkpoint_filename", None)
    infer_cfg = InferenceConfig(
        checkpoint_path=infer_kwargs.pop("checkpoint_path"),
        bpe_path=infer_kwargs.pop("bpe_path"),
        device=infer_kwargs.pop("device", "cuda"),
        **infer_kwargs,
    )
    return SAM3RSSegmentor(infer_cfg)


def save_prediction(pred: torch.Tensor, save_dir: str, image_path: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_pred.png")
    Image.fromarray(pred.cpu().numpy().astype("uint8")).save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM3-RS on a dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--save-pred", default=None, help="Optional dir to save preds")
    parser.add_argument("--device", default=None, help="Override device (e.g., cuda:0)")
    parser.add_argument(
        "--prob-threshold", type=float, default=None, help="Override prob_threshold"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.save_pred is not None:
        cfg.setdefault("output", {})["save_pred_dir"] = args.save_pred
    if args.device is not None:
        cfg.setdefault("segmentor", {})["device"] = args.device
    if args.prob_threshold is not None:
        cfg.setdefault("segmentor", {})["prob_threshold"] = args.prob_threshold

    dataset_cfg = cfg["dataset"]
    dataset = LoveDADataset(
        data_root=dataset_cfg["data_root"],
        img_dir=dataset_cfg["img_dir"],
        mask_dir=dataset_cfg["mask_dir"],
        cls_file=dataset_cfg.get("cls_file"),
        ignore_index=dataset_cfg.get("ignore_index", 255),
        reduce_zero_label=dataset_cfg.get("reduce_zero_label", True),
    )

    dl_cfg = cfg.get("dataloader", {})
    loader = DataLoader(
        dataset,
        batch_size=dl_cfg.get("batch_size", 1),
        num_workers=dl_cfg.get("num_workers", 4),
        pin_memory=dl_cfg.get("pin_memory", True),
        shuffle=False,
    )

    segmentor = build_segmentor(cfg["segmentor"])

    metric = SegmentationMetric(
        num_classes=dataset.num_classes,
        ignore_index=dataset_cfg.get("ignore_index", 255),
    )

    save_pred_dir = cfg.get("output", {}).get("save_pred_dir")
    metrics_json = cfg.get("output", {}).get("metrics_json")

    torch.set_grad_enabled(False)

    total_imgs = len(loader.dataset)
    processed = 0
    for batch in loader:
        # Paths is a tuple of strings (size=batch_size)
        paths = batch["image_path"]
        masks = batch["mask"] # Tensor [B, H, W]

        # Use TRUE BATCH inference!
        # predict_batch handles image loading, batch encoder, and batch decoder
        batch_results = segmentor.predict_batch(list(paths), save_dir=None, detailed=False)

        for i, result in enumerate(batch_results):
            # Paths and order guaranteed to match
            # result = batch_results[i]
            # gt_mask = masks[i]
            
            gt_mask = masks[i]
            pred = result.seg_pred
            pred_np = pred.cpu().numpy().astype(int)
            gt_np = gt_mask.numpy().astype(int)

            # Debug: check value ranges
            # if (pred_np.max() >= dataset.num_classes) or (pred_np.min() < 0):
            #     print(
            #         f"[WARN] pred out of range in {result.image_path}: min={pred_np.min()} max={pred_np.max()} num_classes={dataset.num_classes}"
            #     )
            
            metric.update(pred_np, gt_np)
            if save_pred_dir is not None:
                save_prediction(pred, save_pred_dir, result.image_path)

        processed += len(paths)
        print(f"[eval] {processed}/{total_imgs} images done", end="\r")

    # Ensure the final progress line ends with newline
    if total_imgs > 0:
        print()

    scores = metric.compute()
    per_class_iou = metric.per_class_iou()
    scores_with_detail = dict(scores)
    scores_with_detail["per_class_iou"] = {
        cls_name: float(iou) for cls_name, iou in zip(dataset.classes, per_class_iou)
    }

    print(json.dumps(scores_with_detail, indent=2))

    if metrics_json is not None:
        os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(scores_with_detail, f, indent=2)


if __name__ == "__main__":
    main()

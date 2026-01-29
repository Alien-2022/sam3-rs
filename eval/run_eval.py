"""Lightweight evaluation entrypoint for SAM3-RS (no mmengine dependency)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, List
import numpy as np
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
        data_root = SAM3_RS_DIR / data_root
        dataset["data_root"] = data_root
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
    else:
        segmentor["checkpoint_path"] = SAM3_RS_DIR / segmentor["checkpoint_path"]
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


def save_prediction(pred_np: np.ndarray, save_dir: str, image_path: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_pred.png")
    # 这里得到mask是标签值向左移一位的，且不包含no-data类别:
    # [background:0, building:1, road:2, water:3, barren:4, forest:5, agriculture:6]
    # 但 gt mask 包含no-data类别，且标签值未移动
    # 所以为对齐需要加1恢复原标签值
    pred_np += 1
    Image.fromarray(pred_np).save(out_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM3-RS on a dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--save-pred", default=None, help="Optional dir to save preds")
    parser.add_argument("--device", default=None, help="Override device (e.g., cuda:0)")
    parser.add_argument(
        "--prob-threshold", type=float, default=None, help="Override prob_threshold"
    )
    parser.add_argument(
        "--use-batch", action="store_true", default=True, help="Use high-speed batch inference"
    )
    parser.add_argument(
        "--no-batch", action="store_false", dest="use_batch", help="Use serial predict_single inference"
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
        drop_last=False,
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

    # Timing accumulation
    t_data = 0
    t_infer = 0
    t_eval = 0

    t_start_loop = time.time()
    for batch in loader:
        t_data_ready = time.time()
        t_data += (t_data_ready - t_start_loop)

        paths = batch["image_path"]
        masks = batch["mask"]
        
        # Select inference mode
        t_infer_start = time.time()
        if args.use_batch:
            # High-speed batch inference (B > 1)
            results = segmentor.predict_batch(paths, detailed=False)
        else:
            # Standard serial inference (one by one)
            results = [segmentor.predict_single(p, detailed=False) for p in paths]
        t_infer_end = time.time()
        t_infer += (t_infer_end - t_infer_start)
        
        t_eval_start = time.time()
        for result, gt_mask in zip(results, masks):
            pred = result.seg_pred
            img_path = result.image_path

            pred_np = pred.cpu().numpy().astype(np.uint8)
            gt_np = gt_mask.numpy().astype(np.uint8)

            metric.update(pred_np, gt_np)
            if save_pred_dir is not None:
                save_prediction(pred_np, save_pred_dir, img_path)

            processed += 1
            # Average per image display
            print(f"[eval] {processed}/{total_imgs} images done | Data: {t_data/(processed/8+1e-6):.3f}s/b | Infer: {t_infer/processed:.3f}s/i | Eval: {t_eval/processed:.3f}s/i", end="\r")
        t_eval += (time.time() - t_eval_start)

        if processed >= 40:
            break
        t_start_loop = time.time()

    # Ensure the final progress line ends with newline
    if total_imgs > 0:
        print()
    
    print(f"\n--- Timing Summary (per image) ---")
    print(f"Data loading: {t_data / processed:.3f}s")
    print(f"Inference:    {t_infer / processed:.3f}s")
    print(f"Eval/Save:     {t_eval / processed:.3f}s")
    print(f"Total:        {(t_data + t_infer + t_eval) / processed:.3f}s\n")

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
        scores_with_detail["prob_threshold"] = cfg["segmentor"].get("prob_threshold", 0.1)
        scores_with_detail["confidence_threshold"] = cfg["segmentor"].get(
            "confidence_threshold", 0.5
        )
        # Create directory if it doesn't exist
        # cur_time=time.strftime("%m%d_%H%M", time.localtime())
        os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(scores_with_detail, f, indent=2)


if __name__ == "__main__":
    """
    single模式和batch模式会有微小的差异，原因如下：
    1.计算顺序不同
        Single模式：逐个prompt计算，中间结果会反复读写
            for prompt_idx in range(7):...
            每次200个query的max融合
        Batch模式：一次性处理，向量化的内存访问模式
            inst_current = inst_mask_logits.max(dim=1)[0]  
            向量化max
    2.浮点数累积误差
        bfloat16 的精度是 7-8位有效数字 
        200个query的max融合，加上7个prompt的处理，会累积微小的误差
    """
    main()
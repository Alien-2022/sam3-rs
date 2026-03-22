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
from eval.datasets import DATASET_REGISTRY
from eval.metrics.seg_metrics import SegmentationMetric
from segmentor_lib.analyzers import StatisticsAnalyzer

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

    # Extract debug_memory and debug_log_file separately
    debug_memory = infer_kwargs.pop("debug_memory", False)
    debug_log_file = infer_kwargs.pop("debug_log_file", None)

    # Pop analyzer-related configs (not part of InferenceConfig)
    infer_kwargs.pop("analyze_statistics", None)
    infer_kwargs.pop("compute_boundary_iou", None)
    infer_kwargs.pop("analyze_presence_score", None)

    infer_cfg = InferenceConfig(
        checkpoint_path=infer_kwargs.pop("checkpoint_path"),
        bpe_path=infer_kwargs.pop("bpe_path"),
        device=infer_kwargs.pop("device", "cuda"),
        debug_memory=debug_memory,
        debug_log_file=debug_log_file,
        **infer_kwargs,
    )
    return SAM3RSSegmentor(infer_cfg)


def save_prediction(pred_np: np.ndarray, save_dir: str, image_path: str,
                    gt_mask: Optional[np.ndarray] = None,
                    reduce_zero_label: bool = True,
                    ignore_index: int = 255) -> None:
    """
    保存预测 mask 到磁盘。

    Args:
        pred_np: 预测的 mask 数组（segmentor输出，已reduce）
        save_dir: 保存预测结果的目录
        image_path: 原始图像路径（用于文件名）
        gt_mask: Ground truth mask（可选，用于掩盖 no-data 区域）
                 注意：如果reduce_zero_label=True，gt_mask已经被reduce过了
        reduce_zero_label: 是否需要将预测结果 +1 以恢复原始标签
                           LoveDA 为 True，OpenEarthMap 为 False
        ignore_index: 忽略的标签值（no-data会被设为这个值）
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}.png")

    # segmentor输出格式（reduce后）:
    # [background:0, building:1, road:2, water:3, barren:4, forest:5, agriculture:6]
    #
    # 如果需要保存为原始标签格式（与原始GT对齐），需要：
    # 1. 将预测结果 +1: [0-6] -> [1-7]（1=background, 2=building, ..., 7=agriculture）
    # 2. 将GT中的no-data区域（值为ignore_index=255）应用到预测中

    # 先备份原始预测（未还原）
    pred_original = pred_np.copy()

    if reduce_zero_label:
        # 还原为原始标签格式
        pred_np += 1  # [0-6] -> [1-7]

    # 将 GT 中 no-data 区域（值为ignore_index）应用到预测结果中
    if gt_mask is not None:
        gt_np = gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask
        # 注意：如果reduce_zero_label=True，gt_mask已经是reduce过的格式
        # no-data被设为ignore_index(255)，background为0，其他类为1-6
        pred_np[gt_np == ignore_index] = 0  # 将no-data区域设为0

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
    parser.set_defaults(use_batch=True)
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

    # Get dataset class from registry
    dataset_name = dataset_cfg.get("name", "loveda").lower()
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")

    dataset_class = DATASET_REGISTRY[dataset_name]
    dataset = dataset_class(
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

    # Get segmentor configuration for metric initialization
    seg_cfg = cfg.get("segmentor", {})
    use_prompted_background = seg_cfg.get("use_prompted_background", False)
    bg_idx = seg_cfg.get("bg_idx", 0)

    # Always use dataset.num_classes for metric
    # When use_prompted_background=False:
    #   - GT labels: [1, num_classes] (no semantic background)
    #   - Predictions: [1, num_classes] after filtering out injected background
    #   - We map these to [0, num_classes-1] internally for confusion matrix
    # When use_prompted_background=True:
    #   - GT labels: [0, num_classes-1] (includes background as valid class)
    #   - Predictions: [0, num_classes-1]
    metric_num_classes = dataset.num_classes

    metric = SegmentationMetric(
        num_classes=metric_num_classes,
        ignore_index=dataset_cfg.get("ignore_index", 255),
        use_prompted_background=use_prompted_background,
        bg_idx=bg_idx,
    )

    # Initialize StatisticsAnalyzer for comprehensive analysis (controlled by config)
    analyze_statistics = cfg.get("segmentor", {}).get("analyze_statistics", False)
    stats_analyzer = StatisticsAnalyzer(
        num_classes=metric_num_classes,
        class_names=dataset.classes,
        ignore_index=dataset_cfg.get("ignore_index", 255),
        use_prompted_background=use_prompted_background,
        bg_idx=bg_idx,
        enabled=analyze_statistics,
        compute_boundary_iou=cfg.get("segmentor", {}).get("compute_boundary_iou", False),
    )

    save_pred_dir = cfg.get("output", {}).get("save_pred_dir")
    metrics_json = cfg.get("output", {}).get("metrics_json")

    torch.set_grad_enabled(False)

    total_imgs = len(loader.dataset)
    processed = 0

    # Get batch_size for memory cleanup timing (move outside loop for efficiency)
    batch_size = loader.batch_size if hasattr(loader, 'batch_size') else 1

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
            
            # Update statistics analyzer
            stats_analyzer.on_after_eval({
                'image_name': os.path.basename(img_path),
                'gt_mask': gt_np,
                'pred_mask': pred_np,
            })
            
            if save_pred_dir is not None:
                ignore_index = dataset_cfg.get("ignore_index", 255)
                save_prediction(pred_np, save_pred_dir, img_path, gt_mask,
                               reduce_zero_label=dataset_cfg.get("reduce_zero_label", True),
                               ignore_index=ignore_index)


            processed += 1
            # Average per image display
            print(f"[eval] {processed}/{total_imgs} images done | Data: {t_data/(processed/8+1e-6):.3f}s/b | Infer: {t_infer/processed:.3f}s/i | Eval: {t_eval/processed:.3f}s/i", end="\r")
        t_eval += (time.time() - t_eval_start)

        if processed >= 20:
            break
        t_start_loop = time.time()

        # 定期清理 GPU 缓存 - 每 5 个 batch 清理一次，避免频繁清理影响性能
        # torch.cuda.empty_cache()
        current_batch = (processed // batch_size) + 1
        if current_batch % 2 == 0:
            torch.cuda.empty_cache()

    # Ensure the final progress line ends with newline
    if total_imgs > 0:
        print()

    # Print presence score statistics if enabled in config
    analyze_ps = cfg.get("segmentor", {}).get("analyze_presence_score", False)
    if analyze_ps:
        from segmentor_lib.analyzers import PresenceScoreAnalyzer
        analyzer = segmentor.engine.get_analyzer(PresenceScoreAnalyzer)
        if analyzer:
            analyzer.report()

    # Print comprehensive statistics analysis
    print("\n")
    stats_analyzer.report()

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

    # When use_prompted_background=False, confusion matrix indices [0, num_classes-1]
    # correspond to GT labels [1, num_classes] (no semantic background in GT).
    # The segmentor injects a background channel (index 0) but it's filtered out
    # in metric._fast_hist() before indexing confusion matrix.
    # So per_class_iou indices directly map to dataset.classes.
    if not use_prompted_background:
        per_class_iou_for_classes = per_class_iou
    else:
        per_class_iou_for_classes = per_class_iou





    scores_with_detail["per_class_iou"] = {
        cls_name: float(iou) for cls_name, iou in zip(dataset.classes, per_class_iou_for_classes)
    }

    # Clear progress line and print JSON output
    print()  # Ensure progress line ends properly
    print("=" * 80)  # Separator line
    print("Evaluation Results:")
    print("=" * 80)
    print(json.dumps(scores_with_detail, indent=2))

    if metrics_json is not None:
        scores_with_detail["prob_threshold"] = segmentor.config.prob_threshold
        scores_with_detail["confidence_threshold"] = segmentor.config.confidence_threshold
        scores_with_detail["use_semantic_head"] = segmentor.config.use_semantic_head
        scores_with_detail["use_instance_head"] = segmentor.config.use_instance_head
        scores_with_detail["use_presence_score"]=segmentor.config.use_presence_score

        # Save prompts information
        if segmentor.prompts is not None:
            # Group prompts by class index for better readability
            prompts_by_class = {}
            for name, idx in zip(segmentor.prompts["names"], segmentor.prompts["indices"]):
                if idx not in prompts_by_class:
                    prompts_by_class[idx] = []
                prompts_by_class[idx].append(name)

            # Sort by class index to maintain order
            prompts_list = []
            for class_idx in sorted(prompts_by_class.keys()):
                prompt_list = prompts_by_class[class_idx]
                # Join synonyms with comma, same format as input txt
                prompts_list.append(",".join(prompt_list))

            scores_with_detail["prompts"] = {
                "names": prompts_list,  # e.g., ["background", "building,structure,construction", "road", ...]
                "num_classes": segmentor.num_classes,
                "num_prompts": segmentor.num_prompts,
            }

        cur_time=time.strftime("%m%d_%H%M", time.localtime())
        scores_with_detail["run_time"] = cur_time
        # Create directory if it doesn't exist
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
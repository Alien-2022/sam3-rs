"""Test script for Concept Bank Stage I crop sampling visualization.

Visualizes the crop sampling process of Stage I:
- For each calibration image, creates a folder named after the image
- Inside each image folder, creates subfolders by class ID (from GT mask pixel values)
- Saves the cropped sub-regions for each class

Usage:
    python assistant/test_stage1_crops.py \
        --config eval/configs/potsdam_A2_extended.yaml \
        --num_calib 5 \
        --pad_ratio 0.05 --min_crop_size 128 --max_crop_size 1024

Output directory: outputs/build_cbank_stage1/
    outputs/build_cbank_stage1/
        img_001/
            0/          # class 0 (background)
                crop_0.png
                crop_1.png
            1/          # class 1 (impervious surface)
                crop_0.png
            3/          # class 3 (low vegetation)
                crop_0.png
                crop_1.png
                crop_2.png
        img_002/
            ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Repository roots
SAM3_RS_DIR = Path("/home/ubuntu/research-workspace/workspace/core/sam3-rs")

if str(SAM3_RS_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_RS_DIR))

from build_concept_bank import load_config, build_dataset  # noqa: E402
from segmentor_lib.experimental.concept_bank import square_box_from_mask  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Stage I crop sampling")
    parser.add_argument("--config", required=True, help="Dataset config (e.g., potsdam_A2_extended.yaml)")
    parser.add_argument("--num_calib", type=int, default=5, help="Number of calibration images to visualize")
    parser.add_argument("--pad_ratio", type=float, default=0.05)
    parser.add_argument("--min_crop_size", type=int, default=128)
    parser.add_argument("--max_crop_size", type=int, default=1024)
    parser.add_argument("--max_per_class", type=int, default=10, help="Max crops per class (same as Stage I)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/build_cbank_stage1/)")
    parser.add_argument("--draw_bbox", action="store_true",
                        help="Also draw bbox outline on the original image and save as overview")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset = build_dataset(cfg)
    N = min(args.num_calib, len(dataset))

    ds_cfg = cfg["dataset"]
    ignore_index = ds_cfg.get("ignore_index", 255)
    cls_file = ds_cfg.get("cls_file", "")

    # Load class names for reference
    class_names = {}
    cls_path = SAM3_RS_DIR / "eval/datasets" / os.path.basename(cls_file)
    if cls_path.exists():
        with open(cls_path, "r") as f:
            for i, line in enumerate(f):
                class_names[i] = line.strip()

    # Determine number of classes from dataset
    sample = dataset[0]
    gt_mask = sample["mask"].numpy()
    num_classes = int(gt_mask.max()) + 1

    # Output directory
    out_root = Path(args.output) if args.output else SAM3_RS_DIR / "outputs/build_cbank_stage1"
    out_root.mkdir(parents=True, exist_ok=True)

    # Stage I parameters
    pad_ratio = args.pad_ratio
    min_size = args.min_crop_size
    max_size = args.max_crop_size
    max_per_class = args.max_per_class

    # cap: how many classes to process per image (same formula as Stage I)
    C = num_classes
    if C <= 20:
        cap = C
    else:
        cap = max(3, min(9, int(np.sqrt(C) / 5.0 + 2.0)))

    rng = np.random.RandomState(42)
    count_per_class = np.zeros((C,), dtype=np.int32)

    print(f"Stage I Crop Sampling Visualization")
    print(f"  Dataset: {cfg['dataset']['name']}, Images: {N}/{len(dataset)}")
    print(f"  Classes: {C}, cap_per_image: {cap}, max_per_class: {max_per_class}")
    print(f"  pad_ratio={pad_ratio}, min_size={min_size}, max_size={max_size}")
    print(f"  Output: {out_root}")
    print(f"  Class names: {class_names}")
    print()

    for idx in range(N):
        if (count_per_class >= max_per_class).all():
            print(f"  All classes reached max ({max_per_class}), stopping early.")
            break

        sample = dataset[idx]
        image_path = sample["image_path"]
        gt_mask = sample["mask"].numpy() if torch.is_tensor(sample["mask"]) else np.asarray(sample["mask"])

        img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        img_name = Path(image_path).stem

        # Find valid classes in this mask
        valid_classes = []
        for class_id in range(C):
            if class_id == ignore_index:
                continue
            if count_per_class[class_id] >= max_per_class:
                continue
            if np.any(gt_mask == class_id):
                valid_classes.append(class_id)

        # Prioritize classes with fewer samples (same as Stage I)
        valid_classes.sort(key=lambda c: (count_per_class[c], c))
        classes_this_img = valid_classes[:cap]

        # Image output folder
        img_out_dir = out_root / img_name
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # Collect bboxes for overview (if requested)
        bboxes_for_overview = []

        for class_id in classes_this_img:
            # Get crop box using the same logic as Stage I
            box = square_box_from_mask(gt_mask, class_id, pad_ratio, min_size, max_size, rng)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            crop_np = img[y1:y2 + 1, x1:x2 + 1].copy()
            crop_size = crop_np.shape  # (H, W, 3)

            bboxes_for_overview.append((class_id, box, crop_size))

            # Create class folder
            cls_dir = img_out_dir / str(class_id)
            cls_dir.mkdir(parents=True, exist_ok=True)

            # Save crop image
            crop_count = count_per_class[class_id]
            crop_name = f"crop_{crop_count:02d}.png"
            Image.fromarray(crop_np).save(cls_dir / crop_name)

            # Also save the corresponding mask crop for reference
            mask_crop = gt_mask[y1:y2 + 1, x1:x2 + 1].copy()
            mask_vis = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
            mask_vis[mask_crop == class_id] = [255, 255, 255]  # white = target class
            mask_vis[mask_crop != class_id] = [40, 40, 40]     # dark = others
            Image.fromarray(mask_vis).save(cls_dir / f"crop_{crop_count:02d}_mask.png")

            count_per_class[class_id] += 1

            cls_label = class_names.get(class_id, f"class_{class_id}")
            print(f"  [{idx}] {img_name} / class {class_id} ({cls_label}): "
                  f"bbox=({x1},{y1})-({x2},{y2}), size={crop_size[0]}x{crop_size[1]}")

        # Draw bbox overview on original image
        if args.draw_bbox and bboxes_for_overview:
            import colorsys
            img_overview = img.copy()

            # Generate distinct colors for each class
            for ci, (class_id, (x1, y1, x2, y2), crop_size) in enumerate(bboxes_for_overview):
                hue = class_id / max(C, 1)
                r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
                color = (int(r * 255), int(g * 255), int(b * 255))

                # Draw rectangle
                thickness = max(2, min(img_overview.shape[0], img_overview.shape[1]) // 500)
                # Top edge
                img_overview[y1:y1 + thickness, x1:x2 + 1] = color
                # Bottom edge
                img_overview[y2:y2 + thickness, x1:x2 + 1] = color
                # Left edge
                img_overview[y1:y2 + 1, x1:x1 + thickness] = color
                # Right edge
                img_overview[y1:y2 + 1, x2:x2 + thickness] = color

                # Label
                cls_label = class_names.get(class_id, str(class_id))
                label = f"{class_id}:{cls_label}"
                from PIL import ImageDraw, ImageFont
                img_pil = Image.fromarray(img_overview)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1 + 3, y1 + 3), label, fill=color)
                img_overview = np.asarray(img_pil)

            overview_path = img_out_dir / "bbox_overview.png"
            Image.fromarray(img_overview).save(overview_path)
            print(f"  [{idx}] {img_name}: saved bbox overview -> {overview_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    for class_id in range(C):
        if count_per_class[class_id] > 0:
            cls_label = class_names.get(class_id, f"class_{class_id}")
            print(f"  Class {class_id} ({cls_label}): {count_per_class[class_id]} crops")
        else:
            cls_label = class_names.get(class_id, f"class_{class_id}")
            print(f"  Class {class_id} ({cls_label}): 0 crops (not found in calibration set)")
    print(f"\nOutput saved to: {out_root}")


if __name__ == "__main__":
    main()

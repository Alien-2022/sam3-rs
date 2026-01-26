"""
Simplified demo for SAM3-RS segmentor with head-based visualization.

This demo demonstrates:
- Using segmentor.predict_single() with different head configurations
- Visualizing results based on head selection (semantic/instance/dual)
- Saving results with clear naming for comparison
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from segmentor import SAM3RSSegmentor, InferenceConfig
from workspace.configs.path_conf import ROOT_DIR, TEST_DIR
from workspace.scripts.get_weights import get_weight_path
from workspace.scripts.get_data import get_data_path

SAM_RS_DIR = os.path.join(ROOT_DIR, "workspace/core/sam3-rs")
sys.path.insert(0, SAM_RS_DIR)

# Color palette for visualization (LoveDA style)
COLORS = [
    [0, 0, 0],        # 0: background - black
    [160, 140, 110],   # 1: bareland
    [100, 100, 100],   # 2: road
    [200, 70, 50],     # 3: car
    [40, 100, 50],     # 4: tree
    [50, 120, 180],    # 5: water
    [220, 190, 60],    # 6: cropland
    [140, 80, 70],     # 7: building
]


def save_prediction_mask(seg_pred, output_path):
    """
    Save binary prediction mask.

    Args:
        seg_pred: [H, W] numpy array with class IDs
        output_path: Path to save the mask
    """
    mask = seg_pred.astype(np.uint8)
    # Debug: show ID distribution to verify non-zero presence
    unique_vals = np.unique(mask)
    pred_img = Image.fromarray(mask, mode="L")
    pred_img.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")


def save_colored_mask(seg_pred, colors, output_path):
    """
    Save colored segmentation mask.

    Args:
        seg_pred: [H, W] numpy array with class IDs
        colors: List of RGB colors for each class
        output_path: Path to save the mask
    """
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        mask_area = seg_pred == i
        if mask_area.sum() > 0:
            colored_mask[mask_area] = color

    mask_img = Image.fromarray(colored_mask)
    mask_img.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")


def save_overlay(image, seg_pred, colors, output_path, alpha=0.5):
    """
    Save overlay of segmentation on original image.

    Args:
        image: PIL Image
        seg_pred: [H, W] numpy array with class IDs
        colors: List of RGB colors for each class
        output_path: Path to save the overlay
        alpha: Transparency of the mask (0-1)
    """
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        mask_area = seg_pred == i
        if mask_area.sum() > 0:
            colored_mask[mask_area] = color

    mask_rgb = Image.fromarray(colored_mask)

    if mask_rgb.size != image.size:
        mask_rgb = mask_rgb.resize(image.size, Image.Resampling.NEAREST)

    overlay = Image.blend(image.convert("RGB"), mask_rgb, alpha=alpha)
    overlay.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")


def save_heatmaps(logits, base_path, normalize=True):
    """
    Save per-class heatmaps from logits as grayscale PNGs.
    logits: [C, H, W] tensor on CPU.
    """
    if logits is None:
        return
    logits_np = logits.cpu().numpy()
    c, h, w = logits_np.shape
    for idx in range(c):
        heat = logits_np[idx]
        if normalize:
            vmin, vmax = heat.min(), heat.max()
            if vmax > vmin:
                heat = (heat - vmin) / (vmax - vmin)
            else:
                heat = np.zeros_like(heat)
        heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode="L")
        heat_img.save(f"{base_path}_{idx}.png")


def save_heatmaps_color(logits, base_path, cmap="jet", normalize=True):
    """
    Save per-class heatmaps from logits as colored PNGs using matplotlib colormaps.
    logits: [C, H, W] tensor on CPU.
    """
    if logits is None:
        return
    logits_np = logits.cpu().numpy()
    c, h, w = logits_np.shape
    for idx in range(c):
        heat = logits_np[idx]
        if normalize:
            vmin, vmax = heat.min(), heat.max()
            if vmax > vmin:
                heat = (heat - vmin) / (vmax - vmin)
            else:
                heat = np.zeros_like(heat)
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.tight_layout(pad=0)
        plt.savefig(f"{base_path}_{idx}_color.png", dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()


def save_results_from_single_inference(image, result, base_name, colors=COLORS, save_heatmap=False):
    """
    Save all head results from a single inference call.

    Args:
        image: PIL Image
        result: SegmentationResult from segmentor.predict_single() with dual head enabled
        base_name: Base name for output files (without extension)
        colors: Color palette for visualization
    """
    print(f"\n  Saving results from single inference:")

    # Only save final fused result by default
    dual_pred = result.seg_pred.cpu().numpy()
    base_path = f"{base_name}_dual_head"
    print(f"  Saving dual_head results...")
    # save_prediction_mask(dual_pred, f"{base_path}_pred.png")
    save_colored_mask(dual_pred, colors, f"{base_path}_color.png")
    save_overlay(image, dual_pred, colors, f"{base_path}_overlay.png")

    if save_heatmap:
        # Save semantic and instance logits heatmaps per class
        if result.semantic_logits is not None:
            # save_heatmaps(result.semantic_logits, f"{base_path}_semantic_heat")
            save_heatmaps_color(result.semantic_logits, f"{base_path}_semantic_heat")
        if result.instance_logits is not None:
            # save_heatmaps(result.instance_logits, f"{base_path}_instance_heat")
            save_heatmaps_color(result.instance_logits, f"{base_path}_instance_heat")


def run_single_inference(segmentor, image_path, output_dir, img_name, save_heatmap=False):
    """
    Run inference once with dual head enabled, save all three results.

    Args:
        segmentor: SAM3RSSegmentor instance
        image_path: Path to input image
        output_dir: Directory to save results
        img_name: Base name for output files
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run single inference with dual head
    print("\n" + "=" * 60)
    print("Running Single Inference (Dual Head Enabled)")
    print("=" * 60)
    print(f"use_semantic_head: {segmentor.config.use_semantic_head}")
    print(f"use_instance_head: {segmentor.config.use_instance_head}")

    result = segmentor.predict_single(image_path, detailed=False)

    # Save all results from this single inference
    save_results_from_single_inference(
        image, result,
        f"{output_dir}/{img_name}",
        save_heatmap=save_heatmap,
    )


def main():
    """Main entry point for demo2."""

    # ============ Configuration ============

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Model paths
    checkpoint_path = get_weight_path("sam3") + "/sam3.pt"
    bpe_path = os.path.join(script_dir, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    # Test image path
    dataset_path = get_data_path("LoveDA")
    test_image_path = os.path.join(dataset_path, "Val/Urban/images_png/4168.png")

    # Prompts file for multi-class segmentation
    # prompts_file = os.path.join(script_dir, "configs/loveda_classes.txt")
    prompts_file = os.path.join(script_dir, "configs/prompts_example.txt")

    # Output directory
    output_dir = os.path.join(TEST_DIR, "sam3", "head_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # ============ Initialize SAM3-RS Segmentor ============

    print("\n" + "=" * 60)
    print("Initializing SAM3-RS Segmentor")
    print("=" * 60 + "\n")

    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device="cuda",
        confidence_threshold=0.1,  # Lower threshold to keep more detections
        prob_threshold=0.5,
        use_semantic_head=True,
        use_instance_head=True, 
        use_presence_score=False,
        slide_crop_size=0,        # No sliding window for small images
        slide_stride=1024,
        prompts_file=prompts_file,
        prompt_includes_bg=False
    )

    segmentor = SAM3RSSegmentor(config)

    # ============ Run Single Inference ============

    img_name = os.path.splitext(os.path.basename(test_image_path))[0]

    run_single_inference(segmentor, test_image_path, output_dir, img_name, save_heatmap=True)

    # ============ Summary ============

    print("\n" + "=" * 60)
    print("Demo2 completed!")


if __name__ == "__main__":
    # Set TF32 mode for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run demo
    main()

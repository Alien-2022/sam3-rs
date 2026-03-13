"""
Simplified demo for SAM3-RS segmentor with head-based visualization.

This demo demonstrates:
- Using segmentor.predict_single() with different head configurations
- Visualizing results based on head selection (semantic/instance/dual)
- Saving results with clear naming for comparison

Usage:
    python demo.py --image /path/to/image.png --prompts configs/prompts_example.txt
    python demo.py --image /path/to/image.png --checkpoint weights/sam3.pt --colormap loveda
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from segmentor import SAM3RSSegmentor, InferenceConfig

# Import colormaps
from eval.colormaps import LOVEDA, OPENEARTHMAP, VAIHINGEN, POTSDAM, UAVID, ISAID

# Default colormap - can be changed via command line argument
DEFAULT_COLORMAP = "loveda"

# Available colormaps
COLORMAPS = {
    "loveda": LOVEDA,
    "openearthmap": OPENEARTHMAP,
    "vaihingen": VAIHINGEN,
    "potsdam": POTSDAM,
    "uavid": UAVID,
    "isaid": ISAID,
}

def get_colormap(colormap_name=DEFAULT_COLORMAP):
    """
    Get colormap by name.

    Args:
        colormap_name: Name of the colormap (e.g., "loveda", "openearthmap")

    Returns:
        Dict mapping class IDs to color info
    """
    colormap = COLORMAPS.get(colormap_name.lower(), LOVEDA)
    return colormap

def colors_from_colormap(colormap):
    """
    Extract colors list from colormap dict.

    Args:
        colormap: Dict mapping class IDs to color info

    Returns:
        List of RGB colors for each class
    """
    colors = []
    max_class_id = max(colormap.keys())
    for i in range(max_class_id + 1):
        if i in colormap:
            colors.append(colormap[i]["color"])
        else:
            colors.append([0, 0, 0])  # Black for undefined classes
    return colors


def save_prediction_mask(seg_pred, output_path, reduce_zero_label=True):
    """
    Save prediction mask.

    Args:
        seg_pred: [H, W] numpy array with class IDs (segmentor output format)
        output_path: Path to save the mask
        reduce_zero_label: Whether to restore original label format by adding 1
    """
    # Segmentor output format (reduced):
    # [background:0, building:1, road:2, water:3, barren:4, forest:5, agriculture:6]
    #
    # If need to save as original label format (aligned with original GT), need to:
    # 1. Add 1 to predictions: [0-6] -> [1-7] (1=background, 2=building, ..., 7=agriculture)

    if reduce_zero_label:
        # Restore to original label format
        seg_pred = seg_pred + 1  # [0-6] -> [1-7]

    mask = seg_pred.astype(np.uint8)
    pred_img = Image.fromarray(mask, mode="L")
    pred_img.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")


def save_colored_mask(seg_pred, colors, output_path, reduce_zero_label=True):
    """
    Save colored segmentation mask.

    Args:
        seg_pred: [H, W] numpy array with class IDs
        colors: List of RGB colors for each class
        output_path: Path to save the mask
        reduce_zero_label: Whether seg_pred is in reduced format (0=background) or original format (0=no-data, 1=background)
    """
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        # Adjust class ID based on reduce_zero_label
        # If reduce_zero_label=True: seg_pred is [0:bg, 1:building, ...], need to map to colors[1:bg, 2:building, ...]
        # If reduce_zero_label=False: seg_pred matches colors directly
        if reduce_zero_label:
            # Skip colors[0] (no-data) since seg_pred starts from background at index 0
            class_idx = i - 1  # i=1(bg)->0, i=2(building)->1, etc.
            if class_idx >= 0:
                mask_area = seg_pred == class_idx
                if mask_area.sum() > 0:
                    colored_mask[mask_area] = color
        else:
            mask_area = seg_pred == i
            if mask_area.sum() > 0:
                colored_mask[mask_area] = color

    mask_img = Image.fromarray(colored_mask)
    mask_img.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")


def save_overlay(image, seg_pred, colors, output_path, alpha=0.5, reduce_zero_label=True):
    """
    Save overlay of segmentation on original image.

    Args:
        image: PIL Image
        seg_pred: [H, W] numpy array with class IDs
        colors: List of RGB colors for each class
        output_path: Path to save the overlay
        alpha: Transparency of the mask (0-1)
        reduce_zero_label: Whether seg_pred is in reduced format (0=background) or original format (0=no-data, 1=background)
    """
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        # Adjust class ID based on reduce_zero_label
        if reduce_zero_label:
            # Skip colors[0] (no-data) since seg_pred starts from background at index 0
            class_idx = i - 1
            if class_idx >= 0:
                mask_area = seg_pred == class_idx
                if mask_area.sum() > 0:
                    colored_mask[mask_area] = color
        else:
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


def print_presence_scores(result, segmentor):
    """
    Print presence scores for each class.

    Args:
        result: SegmentationResult from segmentor.predict_single()
        segmentor: SAM3RSSegmentor instance to get class mapping
    """
    if not result.per_class_results or not result.per_class_results:
        print("  No presence scores available (need detailed=True)")
        return

    print("\n  " + "=" * 60)
    print("  Presence Scores:")
    print("=" * 60)

    # Map prompts to class indices
    if segmentor.num_classes != segmentor.num_prompts:
        # Use class mapping to group prompts
        class_scores = {}
        class_names = {}

        for class_id in range(segmentor.num_classes):
            class_scores[class_id] = []

        for prompt_name, class_idx in zip(segmentor.prompts["names"], segmentor.prompts["indices"]):
            presence_score = result.per_class_results.get(prompt_name, {}).get("presence_score", None)
            if presence_score is not None:
                class_scores[class_idx].append(presence_score)
                if class_idx not in class_names:
                    class_names[class_idx] = prompt_name

        # Print all prompts with their presence scores (not averaged)
        print(f"  {'Prompt':<30} {'Class ID':>10} {'Presence':>10}")
        print("  " + "-" * 52)
        for prompt_name, class_idx in zip(segmentor.prompts["names"], segmentor.prompts["indices"]):
            presence_score = result.per_class_results.get(prompt_name, {}).get("presence_score", None)
            if presence_score is not None:
                print(f"  {prompt_name:<30} {class_idx:>10d} {presence_score:>10.4f}")
            else:
                print(f"  {prompt_name:<30} {class_idx:>10d} {'N/A':>10}")
    else:
        # Direct mapping (1 prompt = 1 class)
        print(f"  {'Prompt':<30} {'Class ID':>10} {'Presence':>10}")
        print("  " + "-" * 52)
        for prompt_name in sorted(result.per_class_results.keys()):
            ps = result.per_class_results[prompt_name].get("presence_score", None)
            if ps is not None:
                print(f"  {prompt_name:<30} {result.per_class_results[prompt_name].get('class_idx', '?'):>10} {ps:>10.4f}")

    print("=" * 60)


def save_results_from_single_inference(image, result, base_name, colors, segmentor, save_heatmap=False, reduce_zero_label=True, show_presence_scores=True):
    """
    Save all head results from a single inference call.

    Args:
        image: PIL Image
        result: SegmentationResult from segmentor.predict_single() with dual head enabled
        base_name: Base name for output files (without extension)
        colors: Color palette for visualization
        segmentor: SAM3RSSegmentor instance (for presence score mapping)
        save_heatmap: Whether to save heatmaps
        reduce_zero_label: Whether seg_pred is in reduced format (0=background) or original format (0=no-data, 1=background)
        show_presence_scores: Whether to print presence scores
    """
    print(f"\n  Saving results from single inference:")

    # Only save final fused result by default
    dual_pred = result.seg_pred.cpu().numpy()
    base_path = f"{base_name}_dual_head"
    print(f"  Saving dual_head results...")
    save_prediction_mask(dual_pred, f"{base_path}_pred.png", reduce_zero_label=reduce_zero_label)
    save_colored_mask(dual_pred, colors, f"{base_path}_color.png", reduce_zero_label=reduce_zero_label)
    save_overlay(image, dual_pred, colors, f"{base_path}_overlay.png", reduce_zero_label=reduce_zero_label)

    if save_heatmap:
        # Save semantic and instance logits heatmaps per class
        if result.semantic_logits is not None:
            # save_heatmaps(result.semantic_logits, f"{base_path}_semantic_heat")
            save_heatmaps_color(result.semantic_logits, f"{base_path}_semantic_heat")
        if result.instance_logits is not None:
            # save_heatmaps(result.instance_logits, f"{base_path}_instance_heat")
            save_heatmaps_color(result.instance_logits, f"{base_path}_instance_heat")

    if show_presence_scores:
        print_presence_scores(result, segmentor)


def run_single_inference(segmentor: SAM3RSSegmentor, image_path, output_dir, img_name, colors, save_heatmap=False, reduce_zero_label=True, show_presence_scores=True):
    """
    Run inference once with dual head enabled, save all three results.

    Args:
        segmentor: SAM3RSSegmentor instance
        image_path: Path to input image
        output_dir: Directory to save results
        img_name: Base name for output files
        show_presence_scores: Whether to print presence scores
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run single inference with dual head
    print("\n" + "=" * 60)
    print("Running Single Inference (Dual Head Enabled)")
    print("=" * 60)

    result = segmentor.predict_single(image_path, detailed=True)  # Need detailed=True for presence scores

    # Save all results from this single inference
    save_results_from_single_inference(
        image, result,
        f"{output_dir}/{img_name}",
        colors=colors,
        segmentor=segmentor,
        save_heatmap=save_heatmap,
        reduce_zero_label=reduce_zero_label,
        show_presence_scores=show_presence_scores,
    )


def main():
    """Main entry point for demo."""

    parser = argparse.ArgumentParser(description="SAM3-RS Demo")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "weights/sam3.pt"),
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument(
        "--bpe",
        default=os.path.join(os.path.dirname(__file__), "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        help="Path to BPE vocabulary file",
    )
    parser.add_argument(
        "--prompts",
        default=os.path.join(os.path.dirname(__file__), "configs/prompts_example.txt"),
        help="Path to prompts file",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "outputs/demo"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--colormap",
        default=DEFAULT_COLORMAP,
        choices=list(COLORMAPS.keys()),
        help="Colormap for visualization",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--slide_crop_size",
        type=int,
        default=0,
        help="Sliding window crop size (0 = no sliding window)",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5
    )
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument(
        "--save_heatmap", action="store_true", help="Save per-class heatmaps"
    )
    parser.add_argument(
        "--no_reduce_zero_label",
        action="store_true",
        help="Disable reduce_zero_label (keep 0-indexed predictions as-is)",
    )
    parser.add_argument(
        "--use_prompted_background",
        action="store_true",
        help="Treat background as an explicit prompt class",
    )
    args = parser.parse_args()

    # ============ Configuration ============
    script_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = args.checkpoint
    bpe_path = args.bpe
    test_image_path = args.image_path
    prompts_file = args.prompts
    output_dir = args.output_dir
    reduce_zero_label = not args.no_reduce_zero_label

    os.makedirs(output_dir, exist_ok=True)

    # Get colormap for visualization
    colormap = get_colormap(args.colormap)
    colors = colors_from_colormap(colormap)
    print(f"\nUsing colormap: {args.colormap}")
    print(f"Available colormaps: {', '.join(COLORMAPS.keys())}")

    # ============ Initialize SAM3-RS Segmentor ============

    print("\n" + "=" * 60)
    print("Initializing SAM3-RS Segmentor")
    print("=" * 60 + "\n")

    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        prob_threshold=args.prob_threshold,
        use_semantic_head=True,
        use_instance_head=True,
        use_presence_score=True,
        use_semantic_enhancement=False,
        slide_crop_size=args.slide_crop_size,
        slide_stride=1024,
        prompts_file=prompts_file,
        use_prompted_background=args.use_prompted_background,
    )

    segmentor = SAM3RSSegmentor(config)

    # ============ Run Single Inference ============

    img_name = os.path.splitext(os.path.basename(test_image_path))[0]

    run_single_inference(
        segmentor,
        test_image_path,
        output_dir,
        img_name,
        save_heatmap=args.save_heatmap,
        colors=colors,
        reduce_zero_label=reduce_zero_label,
        show_presence_scores=True,
    )

    # ============ Summary ============

    print("\n" + "=" * 60)
    print(f"Demo completed! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Set TF32 mode for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run demo
    main()

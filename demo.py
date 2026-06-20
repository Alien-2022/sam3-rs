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

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
SAM_RS_DIR = script_dir
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))

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


def generate_random_colors(num_classes, seed=42, use_black_background=True):
    """
    Generate random distinct colors for visualization.

    Args:
        num_classes: Number of classes to generate colors for
        seed: Random seed for reproducibility
        use_black_background: If True, use black for class 0 (background)

    Returns:
        List of RGB colors (num_classes, 3)
    """
    np.random.seed(seed)

    colors = []

    if use_black_background and num_classes > 0:
        # Use black for background (class 0)
        colors.append([0, 0, 0])
        remaining_classes = num_classes - 1
        start_idx = 1
    else:
        remaining_classes = num_classes
        start_idx = 0

    if remaining_classes > 0:
        # Use HSV color space to generate distinct colors for remaining classes
        hues = np.linspace(0, 1, remaining_classes, endpoint=False)
        saturations = np.random.uniform(0.7, 1.0, remaining_classes)
        values = np.random.uniform(0.8, 1.0, remaining_classes)

        for i in range(remaining_classes):
            h, s, v = hues[i], saturations[i], values[i]

            # HSV to RGB conversion
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c

            if 0 <= h * 6 < 1:
                r, g, b = c, x, 0
            elif 1 <= h * 6 < 2:
                r, g, b = x, c, 0
            elif 2 <= h * 6 < 3:
                r, g, b = 0, c, x
            elif 3 <= h * 6 < 4:
                r, g, b = 0, x, c
            elif 4 <= h * 6 < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            rgb = [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]
            colors.append(rgb)

    return colors


def save_prediction_mask(seg_pred, output_path):
    """
    将预测掩码保存为灰度图像文件
    
    Args:
        seg_pred (np.ndarray): 形状为 [H, W] 的 numpy 数组，包含类别 ID（segmentor 输出格式）
        output_path (str): 掩码图像的保存路径
    
    Note:
        直接保存 segmentor 的原始输出，不进行标签格式转换，保持与 prompts 文件一致
        （0=背景，1=第一个类别，以此类推）。
    """
    """
    Save prediction mask (raw segmentor output).

    Args:
        seg_pred: [H, W] numpy array with class IDs (segmentor output format)
        output_path: Path to save the mask

    Note:
        For demo/visualization purposes, we save the raw segmentor output directly
        without any label format transformation. This keeps results consistent with
        the prompts file (0=background, 1=first_class, etc.).
    """
    mask = seg_pred.astype(np.uint8)
    pred_img = Image.fromarray(mask, mode="L")
    pred_img.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")


def save_colored_mask(seg_pred, colors, output_path, label_map=None):
    """
    Save colored segmentation mask with optional labels.

    Args:
        seg_pred: [H, W] numpy array with class IDs (segmentor output: 0=background, 1=class1, ...)
        colors: List of RGB colors for each class (0=background, 1=class1, ...)
        output_path: Path to save the mask
        label_map: Dict mapping class IDs to label names (optional, for legend)

    Note:
        Uses raw segmentor output format where colors[i] corresponds to class i.
    """
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Get unique class IDs in the prediction
    unique_classes = np.unique(seg_pred)

    for class_id in unique_classes:
        if class_id < 0 or class_id >= len(colors):
            continue

        mask_area = seg_pred == class_id
        if mask_area.sum() > 0:
            colored_mask[mask_area] = colors[class_id]

    mask_img = Image.fromarray(colored_mask)

    # Add legend if label_map is provided
    if label_map:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(colored_mask)
        ax.axis('off')

        # Build legend
        legend_elements = []
        for class_id in unique_classes:
            if class_id < 0 or class_id >= len(colors):
                continue
            label = label_map.get(class_id, f"Class {class_id}")
            color = [c/255.0 for c in colors[class_id]]
            legend_elements.append(plt.Patch(facecolor=color, edgecolor='black', label=label))

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                  fontsize=10, framealpha=0.9, edgecolor='black')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        mask_img.save(output_path)

    print(f"  Saved: {os.path.basename(output_path)}")


def save_overlay(image, seg_pred, colors, output_path, alpha=0.5, label_map=None):
    """
    Save overlay of segmentation on original image with optional labels.

    Args:
        image: PIL Image
        seg_pred: [H, W] numpy array with class IDs (segmentor output: 0=background, 1=class1, ...)
        colors: List of RGB colors for each class (0=background, 1=class1, ...)
        output_path: Path to save the overlay
        alpha: Transparency of the mask (0-1)
        label_map: Dict mapping class IDs to label names (optional, for legend)

    Note:
        Uses raw segmentor output format where colors[i] corresponds to class i.
    """
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Get unique class IDs in the prediction
    unique_classes = np.unique(seg_pred)

    for class_id in unique_classes:
        if class_id < 0 or class_id >= len(colors):
            continue

        mask_area = seg_pred == class_id
        if mask_area.sum() > 0:
            colored_mask[mask_area] = colors[class_id]

    mask_rgb = Image.fromarray(colored_mask)

    if mask_rgb.size != image.size:
        mask_rgb = mask_rgb.resize(image.size, Image.Resampling.NEAREST)

    # Create overlay with matplotlib if label_map is provided
    if label_map:
        overlay_array = np.array(Image.blend(image.convert("RGB"), mask_rgb, alpha=alpha))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(overlay_array)
        ax.axis('off')

        # Build legend
        legend_elements = []
        for class_id in unique_classes:
            if class_id < 0 or class_id >= len(colors):
                continue
            label = label_map.get(class_id, f"Class {class_id}")
            color = [c/255.0 for c in colors[class_id]]
            legend_elements.append(plt.Patch(facecolor=color, edgecolor='white', label=label))

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                  fontsize=10, framealpha=0.9, edgecolor='white')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
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
    logits_np = logits.float().cpu().numpy()
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
    logits_np = logits.float().cpu().numpy()
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
    save_prediction_mask(dual_pred, f"{base_path}_pred.png")
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

    if show_presence_scores:
        print_presence_scores(result, segmentor)


def run_single_inference(segmentor: SAM3RSSegmentor, image_path, output_dir, img_name, colors, save_heatmap=False, show_presence_scores=True):
    """
    Run inference once with dual head enabled, save all three results.

    Args:
        segmentor: SAM3RSSegmentor instance
        image_path: Path to input image
        output_dir: Directory to save results
        img_name: Base name for output files
        colors: Color palette for visualization (0=background, 1=class1, ...)
        save_heatmap: Whether to save heatmaps
        show_presence_scores: Whether to print presence scores

    Note:
        Uses raw segmentor output format without label transformation.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run single inference with dual head
    print("\n" + "=" * 60)
    print("Running Single Inference (Dual Head Enabled)")
    print("=" * 60)

    result = segmentor.predict_single(image_path, detailed=False)  # Need detailed=True for presence scores

    # Save all results from this single inference
    save_results_from_single_inference(
        image, result,
        f"{output_dir}/{img_name}",
        colors=colors,
        segmentor=segmentor,
        save_heatmap=save_heatmap,
        show_presence_scores=show_presence_scores,
    )


def main():
    """Main entry point for demo2."""

    # ============ Configuration ============

    # Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    SAM_RS_DIR = script_dir
    TEST_DIR = os.path.join(SAM_RS_DIR, "test", "output")

    # Model paths
    checkpoint_path = os.path.join(SAM_RS_DIR, "weights/sam3/sam3.pt")
    bpe_path = os.path.join(script_dir, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    # Test image path
    test_image_path = os.path.join(SAM_RS_DIR, "test/2522.png")


    # Prompts file for multi-class segmentation
    prompts_file = os.path.join(script_dir, "configs/prompts_example.txt")

    # Check if background is explicitly in prompts
    with open(prompts_file, 'r') as f:
        first_line = f.readline().strip().lower()
        use_prompted_background = first_line == "background" or first_line.startswith("background")

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
        confidence_threshold=0.2,
        prob_threshold=0.1,
        use_semantic_head=True,
        use_instance_head=True,
        use_presence_score=True,
        slide_crop_size=0,        # No sliding window for small images
        slide_stride=1024,
        prompts_file=prompts_file,
        use_prompted_background=use_prompted_background,
    )

    segmentor = SAM3RSSegmentor(config)

    print(f"\nPrompts mode: {'background explicitly prompted' if use_prompted_background else 'background via prob_threshold'}")

    # Determine colormap (use random colors if not matching predefined datasets)
    colormap = None
    colors = None

    # Try to use predefined colormap
    colormap_name = None
    for name in ["loveda", "openearthmap", "vaihingen", "potsdam", "uavid", "isaid"]:
        if name.lower() in os.path.basename(prompts_file).lower():
            colormap_name = name
            break

    if colormap_name:
        colormap = get_colormap(colormap_name)
        colors = colors_from_colormap(colormap)
        print(f"\nUsing predefined colormap: {colormap_name}")
    else:
        # Generate random colors based on number of classes
        # Class 0 = background (use black), class 1+ = other classes (random distinct colors)
        num_classes = segmentor.num_classes
        colors = generate_random_colors(num_classes, seed=42, use_black_background=True)
        print(f"\nUsing random colors for {num_classes} classes (custom prompts)")
        print(f"  colors[0] = background (black), colors[1...] = other classes")

    print(f"Available colormaps: {', '.join(COLORMAPS.keys())}")

    # ============ Run Single Inference ============

    img_name = os.path.splitext(os.path.basename(test_image_path))[0]

    run_single_inference(segmentor, test_image_path, output_dir, img_name, save_heatmap=False, colors=colors, show_presence_scores=False)

    # ============ Summary ============

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    # Set TF32 mode for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run demo
    main()

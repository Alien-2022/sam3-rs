"""
Utility functions for SAM3-RS.

Visualization, file I/O, common operations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Optional, Tuple
import torch


# Color palette for visualization (similar to OpenMMLab)
COLORS = [
    (128, 0, 0),      # 0: Red
    (0, 128, 0),      # 1: Green
    (128, 128, 0),    # 2: Olive
    (0, 0, 128),      # 3: Purple
    (128, 0, 128),    # 4: Teal
    (0, 128, 128),    # 5: Navy
    (128, 128, 0),    # 6: Yellow
    (64, 0, 0),       # 7: Maroon
    (192, 128, 0),    # 8: Orange
]


def visualize_results(
    image_path: str,
    seg_pred: np.ndarray,
    seg_logits: Optional[torch.Tensor] = None,
    per_class_results: Optional[Dict] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.5,
    show: bool = False
):
    """
    Visualize segmentation results.

    Args:
        image_path: Original image path
        seg_pred: Predicted segmentation [H, W] with class IDs
        seg_logits: Optional logits [num_classes, H, W]
        per_class_results: Optional per-class detailed results
        save_path: Where to save visualization
        alpha: Overlay transparency (0-1)
        show: Whether to display plot

    Creates:
        - Original image
        - Segmentation mask
        - Overlay (if save_path provided)
    """
    # Load original image
    original = np.array(Image.open(image_path))

    # Create colored segmentation mask
    h, w = seg_pred.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in np.unique(seg_pred):
        if class_id < len(COLORS):
            colored_mask[seg_pred == class_id] = COLORS[class_id]

    # Create overlay
    overlay = original.copy()
    mask_alpha = (seg_pred > 0).astype(np.uint8) * 255 * alpha
    mask_alpha = mask_alpha.astype(np.uint8)
    mask_rgba = Image.fromarray(colored_mask).convert('RGBA')
    mask_rgba.putalpha(Image.fromarray(mask_alpha, mode='L'))

    from PIL import Image as PILImage
    overlay = PILImage.composite(
        mask_rgba,
        PILImage.fromarray(original),
        Image.fromarray(mask_alpha, mode='L')
    )

    # Plotting
    if save_path or show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(colored_mask)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
    else:
        return overlay


def save_segmentation(
    seg_pred: np.ndarray,
    output_path: str,
    color_palette: Optional[List[Tuple]] = None
):
    """
    Save segmentation mask as colored image.

    Args:
        seg_pred: Segmentation [H, W]
        output_path: Where to save
        color_palette: Optional color palette (default: COLORS)

    Example:
        save_segmentation(
            pred,
            'outputs/mask.png',
            color_palette=[(255, 0, 0), (0, 255, 0)]  # Red/Green
        )
    """
    if color_palette is None:
        color_palette = COLORS

    h, w = seg_pred.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in np.unique(seg_pred):
        if class_id < len(color_palette):
            colored[seg_pred == class_id] = color_palette[class_id]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(colored).save(output_path)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay binary mask on image.

    Args:
        image: RGB image [H, W, 3]
        mask: Binary mask [H, W]
        color: Overlay color (R, G, B)
        alpha: Transparency (0-1)

    Returns:
        Overlayed image
    """
    overlay = image.copy()

    # Create colored mask with alpha
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend
    overlay = (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    return overlay


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_image_info(image_path: str) -> Dict:
    """
    Get basic information about an image.

    Args:
        image_path: Path to image

    Returns:
        Dict with 'width', 'height', 'channels', 'format', 'size_bytes'
    """
    img = Image.open(image_path)

    return {
        'width': img.width,
        'height': img.height,
        'channels': len(img.getbands()),
        'format': img.format,
        'size_bytes': os.path.getsize(image_path)
    }


def batch_resize_images(
    image_paths: List[str],
    output_dir: str,
    target_size: Tuple[int, int]
):
    """
    Resize multiple images to target size.

    Args:
        image_paths: List of image paths
        output_dir: Output directory
        target_size: (width, height)

    Example:
        batch_resize_images(
            ['img1.png', 'img2.png'],
            'resized/',
            (1024, 1024)
        )
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img = Image.open(img_path)
        img = img.resize(target_size, Image.Resampling.BILINEAR)

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        img.save(out_path)

    print(f"✓ Resized {len(image_paths)} images to {target_size}")


def calculate_statistics(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray]
) -> Dict:
    """
    Calculate basic statistics on predictions.

    Args:
        predictions: List of predicted masks
        ground_truths: List of ground truth masks

    Returns:
        Dict with 'mean_pred_pixels', 'mean_gt_pixels', 'overlap_ratio', etc.
    """
    total_pred_pixels = sum(p.sum() for p in predictions)
    total_gt_pixels = sum(g.sum() for g in ground_truths)

    overlaps = []
    for pred, gt in zip(predictions, ground_truths):
        overlap = (pred > 0) & (gt > 0)
        overlap_ratio = overlap.sum() / max(pred.sum(), gt.sum(), 1)
        overlaps.append(overlap_ratio)

    return {
        'mean_pred_pixels': total_pred_pixels / len(predictions),
        'mean_gt_pixels': total_gt_pixels / len(ground_truths),
        'mean_overlap_ratio': np.mean(overlaps),
        'min_overlap_ratio': np.min(overlaps),
        'max_overlap_ratio': np.max(overlaps)
    }


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    output_path: Optional[str] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None
):
    """
    Create a grid of images for comparison.

    Args:
        images: List of images
        titles: Title for each image
        output_path: Optional save path
        rows: Number of rows (auto if None)
        cols: Number of cols (auto if None)

    Example:
        create_comparison_grid(
            [img1, img2, img3, img4],
            ['Original', 'Method A', 'Method B', 'GT'],
            'comparison.png',
            rows=2, cols=2
        )
    """
    n = len(images)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {output_path}")
    else:
        plt.show()
    plt.close()


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON/YAML file.

    Args:
        config_path: Path to config file (.json or .yaml)

    Returns:
        Configuration dictionary
    """
    import json
    import yaml

    ext = os.path.splitext(config_path)[1].lower()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")


def save_config(config: Dict, output_path: str):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save
    """
    import json

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Saved config to {output_path}")

"""
Demo script for SAM3-RS using the new segmentor wrapper.

This script demonstrates:
- Single image inference with single prompt
- Single image inference with multi-class prompts
- Result visualization and saving
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Import from new SAM3-RS structure
from segmentor import SAM3RSSegmentor, InferenceConfig
from utils import visualize_prediction, save_mask

from workspace.configs.path_conf import ROOT_DIR, TEST_DIR
from workspace.scripts.get_weights import get_weight_path
from workspace.scripts.get_data import get_data_path


def save_result_as_instance_seg(image, image_name, result, output_path, mode="mask"):
    """
    Visualize and save instance segmentation result.

    Args:
        image: PIL Image object
        image_name: image name
        result: SAM3 output dict
            - masks: Tensor [N, H, W]
            - boxes: Tensor [N, 4]
            - scores: Tensor [N]
        output_path: Path to save image
        mode: "mask" or "all"
    """
    objects_num = len(result["scores"])

    if mode == "mask":
        # Create combined binary mask
        if len(result["masks"]) > 0:
            first_mask = result["masks"][0].squeeze(0).cpu()
            mask_height, mask_width = first_mask.shape
            combined_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

            for i in range(objects_num):
                mask = result["masks"][i].squeeze(0).cpu().numpy()
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

            # Convert to binary mask (0=background, 255=mask)
            combined_mask = combined_mask * 255
            mask_image = Image.fromarray(combined_mask)
            instance_seg_path = os.path.join(
                output_path, f"{image_name}_instance_seg.png"
            )
            mask_image.save(instance_seg_path)
            print(f"Saved mask: {instance_seg_path}")

        # Save semantic segmentation if available
        if "semantic_seg" in result and result["semantic_seg"] is not None:
            semantic_seg = result["semantic_seg"].squeeze(0).squeeze(0).cpu()
            semantic_seg_binary = (semantic_seg > 0.5).numpy().astype(np.uint8) * 255
            semantic_seg_image = Image.fromarray(semantic_seg_binary, mode="L")
            semantic_seg_path = os.path.join(
                output_path, f"{image_name}_semantic_seg.png"
            )
            semantic_seg_image.save(semantic_seg_path)
            print(f"Saved semantic segmentation: {semantic_seg_path}")

    elif mode == "all":
        plt.figure(figsize=(12, 8))
        plt.imshow(image)

        # Use colors for visualization
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan"]

        for i in range(objects_num):
            color = colors[i % len(colors)]

            # Draw mask
            mask = result["masks"][i].squeeze(0).cpu().numpy()
            plt.imshow(mask, alpha=0.3, cmap="binary", color=color)

            # Draw bbox
            box = result["boxes"][i].cpu().numpy()
            w, h = image.size
            prob = result["scores"][i].item()

            # Convert to matplotlib format
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            plt.gca().add_patch(rect)
            plt.text(
                box[0],
                box[1] - 5,
                f"(id={i}, {prob:.2f})",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved visualization: {output_path}")


def save_result_as_semantic_seg(image, result, output_path, save_class_masks=True):
    """
    Visualize and save semantic segmentation result.

    Args:
        image: PIL Image
        result: SAM3-RS SegmentationResult
        output_path: Path to save output
        save_class_masks: Whether to save per-class masks
    """
    img_width, img_height = image.size

    # Get prediction
    seg_pred = result.seg_pred.cpu().numpy()

    # Save prediction mask
    pred_img = Image.fromarray(seg_pred.astype(np.uint8), mode="L")
    base_name = os.path.splitext(output_path)[0]
    pred_path = f"{base_name}_pred.png"
    pred_img.save(pred_path)
    print(f"Saved prediction: {pred_path}")

    # Save per-class binary masks
    if save_class_masks:
        class_masks_dir = f"{base_name}_class_masks"
        os.makedirs(class_masks_dir, exist_ok=True)

        for class_name in result.per_class_results.keys():
            class_data = result.per_class_results[class_name]
            masks = class_data["masks"]

            if masks.shape[0] > 0:
                # Combine all instance masks for this class
                class_mask = masks.cpu().numpy().max(axis=0)
                binary_mask = (class_mask > 0.5).astype(np.uint8) * 255

                mask_img = Image.fromarray(binary_mask, mode="L")
                class_mask_path = os.path.join(class_masks_dir, f"{class_name}.png")
                mask_img.save(class_mask_path)
                print(f"Saved class mask: {class_mask_path}")

    # Create overlay visualization
    colors = [
        [0, 0, 0],  # 0: background - black
        [160, 140, 110],  # 1: bareland
        [100, 100, 100],  # 2: road
        [200, 70, 50],  # 3: car
        [40, 100, 50],  # 4: tree
        [50, 120, 180],  # 5: water
        [220, 190, 60],  # 6: cropland
        [140, 80, 70],  # 7: building
    ]

    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for i in range(len(colors)):
        mask_area = seg_pred == i
        if mask_area.sum() > 0:
            colored_mask[mask_area] = colors[i]

    mask_rgb = Image.fromarray(colored_mask)
    overlay = Image.blend(image.convert("RGB"), mask_rgb, alpha=0.5)
    overlay_path = f"{base_name}_overlay.png"
    overlay.save(overlay_path)
    print(f"Saved overlay: {overlay_path}")


def single_img_single_prompt(
    segmentor: SAM3RSSegmentor,
    image_path: str,
    prompt: str,
    output_path: str,
    mode: str = "mask",
):
    """
    Single image inference with a single prompt (instance segmentation).

    Args:
        segmentor: SAM3RSSegmentor instance
        image_path: Path to input image
        prompt: Text prompt (e.g., "building", "road")
        output_path: Path to save result
        mode: "mask" or "all"
    """
    print(f"\n{'='*60}")
    print(f"Single Prompt Inference")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*60}\n")

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_name = os.path.basename(image_path)[0]

    # Get instance segmentation from SAM3
    inference_state = segmentor.processor.set_image(image)
    output = segmentor.processor.set_text_prompt(state=inference_state, prompt=prompt)

    if output["masks"].numel() == 0:
        print(f"⚠ No masks found for prompt: '{prompt}'")
        return

    # Save results
    save_result_as_instance_seg(image, image_name, output, output_path, mode)

    print(f"✓ Completed single prompt inference\n")
    return output


def single_img_multi_prompts(
    segmentor: SAM3RSSegmentor, image_path: str, prompts_file: str, output_path: str
):
    """
    Single image inference with multi-class prompts (semantic segmentation).

    Args:
        segmentor: SAM3RSSegmentor instance
        image_path: Path to input image
        prompts_file: Path to prompts config file
        output_path: Path to save result
    """
    print(f"\n{'='*60}")
    print(f"Multi-Class Prompt Inference")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prompts file: {prompts_file}")
    print(f"{'='*60}\n")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Run semantic segmentation with multiple classes
    result = segmentor.predict_single(image_path, detailed=True)

    # Save results
    save_result_as_semantic_seg(image, result, output_path, save_class_masks=True)

    # Print summary
    print(f"\nPrediction shape: {result.seg_pred.shape}")
    print(
        f"Number of classes detected: {len(set(result.seg_pred.cpu().numpy().flatten()))}"
    )
    print(f"✓ Completed multi-class inference\n")

    return result


def main():
    """Main entry point for demo."""

    # ============ Configuration ============

    # Model paths
    checkpoint_path = get_weight_path("sam3") + "/sam3.pt"
    bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    # Test image path (modify as needed)
    dataset_path = get_data_path("LoveDA")
    test_image_path = os.path.join(dataset_path, "Val/Urban/images_png/3549.png")

    # Prompts file for multi-class segmentation
    prompts_file = "configs/prompts_example.txt"

    # Output directory
    output_dir = os.path.join(TEST_DIR, "sam3")
    os.makedirs(output_dir, exist_ok=True)

    # ============ Initialize SAM3-RS Segmentor ============

    print("\n" + "=" * 60)
    print("Initializing SAM3-RS Segmentor")
    print("=" * 60 + "\n")

    # Create configuration
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device="cuda",
        confidence_threshold=0.1,
        prob_threshold=0.0,
        use_semantic_head=True,
        use_instance_head=True,
        use_presence_score=True,
        slide_crop_size=0,  # No sliding window for small images
        slide_stride=512,
        prompts_file=prompts_file,  # Load multi-class prompts
    )

    # Initialize segmentor
    segmentor = SAM3RSSegmentor(config)

    # ============ Run Demo ============

    # Demo 1: Single prompt (instance segmentation)
    print("\n" + "=" * 60)
    print("Demo 1: Single Prompt Inference")
    print("=" * 60)

    output_path_single = os.path.join(output_dir, "single_prompt")
    os.makedirs(output_path_single, exist_ok=True)

    # Note: For single prompt, we use SAM3's raw output
    # This is for demonstration of instance segmentation
    single_img_single_prompt(
        segmentor,
        test_image_path,
        prompt="building",
        output_path=output_path_single,
        mode="mask",
    )

    # Demo 2: Multi-class prompts (semantic segmentation)
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Class Semantic Segmentation")
    print("=" * 60)

    output_path_multi = os.path.join(output_dir, "multi_class")
    os.makedirs(output_path_multi, exist_ok=True)

    img_name = os.path.splitext(os.path.basename(test_image_path))[0]
    single_img_multi_prompts(
        segmentor,
        test_image_path,
        prompts_file=prompts_file,
        output_path=os.path.join(output_path_multi, f"{img_name}_seg"),
    )

    # Demo 3: Large image with sliding window (optional)
    # print("\n" + "=" * 60)
    # print("Demo 3: Large Image with Sliding Window")
    # print("(Skipping - requires large image)")
    # print("=" * 60)

    # Uncomment to test sliding window
    # large_image_path = "data/large_image.tif"
    # config_large = InferenceConfig(
    #     checkpoint_path=checkpoint_path,
    #     bpe_path=bpe_path,
    #     device='cuda',
    #     slide_crop_size=1024,
    #     slide_stride=512,
    #     prompts_file=prompts_file
    # )
    # segmentor_large = SAM3RSSegmentor(config_large)
    # result = segmentor_large.predict_single(large_image_path)

    print("\n" + "=" * 60)
    print("Demo completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Set TF32 mode for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run demo
    main()

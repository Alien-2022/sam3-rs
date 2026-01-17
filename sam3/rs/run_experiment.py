"""
Main experiment runner for SAM3-RS.

Demonstrates complete workflow:
1. Load configuration
2. Initialize dataset and model
3. Run inference
4. Evaluate results
5. Save outputs and metrics
"""

import os
import sys
import argparse
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sam3.rs.config import ExperimentConfig, create_preset_openearthmap, create_preset_loveda
from sam3.rs.segmentor import SAM3RSSegmentor, InferenceConfig
from sam3.rs.data import RSDataLoader, get_dataset_loader
from sam3.rs.prompts import PromptManager, PromptConfig
from sam3.rs.metrics import RSEvaluator, compute_metrics_for_directory
from sam3.rs.utils import visualize_results, save_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAM3-RS Experiment Runner")

    # Preset configs
    parser.add_argument(
        '--preset',
        type=str,
        choices=['openearthmap', 'loveda', 'whu', 'custom'],
        default='openearthmap',
        help='Experiment preset (default: openearthmap)'
    )

    # Dataset
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--data_root', type=str, help='Override data root directory')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (train/val/test)')
    parser.add_argument('--subset', type=int, help='Number of samples to use (None = all)')

    # Model
    parser.add_argument('--checkpoint', type=str, help='Path to SAM3 checkpoint')
    parser.add_argument('--bpe_path', type=str, help='Path to BPE tokenizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    # Inference
    parser.add_argument('--slide_crop', type=int, default=0, help='Sliding window crop size (0 = no sliding)')
    parser.add_argument('--slide_stride', type=int, default=512, help='Sliding window stride')
    parser.add_argument('--conf_thd', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--prob_thd', type=float, default=0.0, help='Probability threshold')
    parser.add_argument('--use_semantic', action='store_true', default=True, help='Use semantic head')
    parser.add_argument('--use_instance', action='store_true', default=True, help='Use instance head')
    parser.add_argument('--use_presence', action='store_true', default=True, help='Use presence score')

    # Prompts
    parser.add_argument('--prompts_file', type=str, help='Path to prompts config file')

    # Output
    parser.add_argument('--save_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--vis', action='store_true', help='Show visualizations')
    parser.add_argument('--save_vis', action='store_true', default=True, help='Save visualizations')

    # Config file
    parser.add_argument('--config', type=str, help='Load experiment from JSON config file')
    parser.add_argument('--save_config', type=str, help='Save experiment config to this path')

    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """Create ExperimentConfig from command line arguments."""
    from sam3.rs.config import (
        ModelConfig, InferenceConfig, PromptConfig,
        DatasetConfig, OutputConfig, ExperimentConfig
    )

    # Start with preset if provided
    if args.preset == 'openearthmap':
        config = create_preset_openearthmap()
    elif args.preset == 'loveda':
        config = create_preset_loveda()
    elif args.preset == 'whu':
        from sam3.rs.config import create_preset_whu
        config = create_preset_whu()
    else:
        config = ExperimentConfig()

    # Override with command line args
    if args.dataset:
        config.dataset.name = args.dataset
    if args.data_root:
        config.dataset.root_dir = args.data_root
    if args.split:
        config.dataset.split = args.split
    if args.subset is not None:
        config.dataset.subset = args.subset

    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if args.bpe_path:
        config.model.bpe_path = args.bpe_path
    if args.device:
        config.model.device = args.device

    if args.slide_crop is not None:
        config.inference.slide_crop_size = args.slide_crop
    if args.slide_stride is not None:
        config.inference.slide_stride = args.slide_stride
    if args.conf_thd is not None:
        config.inference.confidence_threshold = args.conf_thd
    if args.prob_thd is not None:
        config.inference.prob_threshold = args.prob_thd
    config.inference.use_semantic_head = args.use_semantic
    config.inference.use_instance_head = args.use_instance
    config.inference.use_presence_score = args.use_presence

    if args.prompts_file:
        config.prompts.prompts_file = args.prompts_file

    if args.save_dir:
        config.output.save_dir = args.save_dir
    config.output.save_predictions = True
    config.output.save_visualizations = args.save_vis
    config.output.show_plots = args.vis

    return config


def run_experiment(config: ExperimentConfig):
    """
    Run complete experiment workflow.

    Args:
        config: ExperimentConfig object

    Returns:
        results: List of segmentation results
        metrics: Evaluation metrics (if GT available)
    """
    print(f"\n{'='*70}")
    print(f"SAM3-RS: {config.experiment_name}")
    print(f"{config.description}")
    print(f"{'='*70}\n")

    config.print_summary()

    # ============ 1. Initialize Model ============
    print("Step 1: Initializing SAM3-RS model...")

    inference_config = InferenceConfig(
        checkpoint_path=config.model.checkpoint_path,
        bpe_path=config.model.bpe_path,
        device=config.model.device,
        confidence_threshold=config.inference.confidence_threshold,
        prob_threshold=config.inference.prob_threshold,
        bg_idx=0,
        use_semantic_head=config.inference.use_semantic_head,
        use_instance_head=config.inference.use_instance_head,
        use_presence_score=config.inference.use_presence_score,
        slide_crop_size=config.inference.slide_crop_size,
        slide_stride=config.inference.slide_stride
    )

    model = SAM3RSSegmentor(inference_config)

    # ============ 2. Load Dataset ============
    print("\nStep 2: Loading dataset...")

    if config.dataset.name.lower() in RSDataLoader.DATASET_CONFIGS:
        data_loader = get_dataset_loader(
            config.dataset.name,
            root_dir=config.dataset.root_dir
        )
    else:
        # Custom dataset
        data_loader = RSDataLoader(DatasetConfig(
            name=config.dataset.name,
            root_dir=config.dataset.root_dir,
            split=config.dataset.split
        ))

    image_paths = data_loader.get_samples(subset=config.dataset.subset)
    print(f"  Found {len(image_paths)} samples")

    # ============ 3. Run Inference ============
    print(f"\nStep 3: Running inference on {len(image_paths)} images...")

    results = model.predict_batch(
        image_paths=image_paths,
        save_dir=config.output.save_dir if config.output.save_predictions else None,
        detailed=False
    )

    # ============ 4. Evaluate ============
    if config.dataset.is_segmentation:
        print("\nStep 4: Evaluating results...")

        # Check if GT exists
        evaluator = RSEvaluator(
            num_classes=model.num_classes,
            ignore_index=255
        )

        has_gt = False
        for result in results:
            img_path = result.image_path

            # Try to find GT
            # Construct expected GT path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            gt_dir = os.path.join(config.dataset.root_dir, config.dataset.split, 'masks')

            # Try different extensions
            for ext in ['.tif', '.png', '.jpg']:
                gt_path = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(gt_path):
                    # Construct prediction path
                    pred_path = os.path.join(config.output.save_dir, base_name + '_pred.png')
                    evaluator.update(pred_path, gt_path)
                    has_gt = True
                    break

        if has_gt:
            metrics = evaluator.compute()
            metrics.print_summary(class_names=model.prompts.get('class_to_synonyms', {}))

            # Save metrics
            metrics_path = os.path.join(config.output.save_dir, 'metrics.json')
            evaluator.save_results(metrics, metrics_path)
        else:
            print("  No ground truth found, skipping evaluation")

    # ============ 5. Summary ============
    print(f"\n{'='*70}")
    print(f"✓ Experiment completed!")
    print(f"  Results saved to: {config.output.save_dir}")
    print(f"{'='*70}\n")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Load from config file or create from args
    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"✓ Loaded config from {args.config}")
    else:
        config = create_config_from_args(args)

    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"✓ Saved config to {args.save_config}")

    # Run experiment
    try:
        results = run_experiment(config)

        # Example: Analyze results
        print("\nResult Statistics:")
        print(f"  Total images: {len(results)}")
        print(f"  Average prediction shape: {results[0].seg_pred.shape}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Set TF32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    main()

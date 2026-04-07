"""
Demo script for unsupervised threshold calibration

This script demonstrates how to use UnsupervisedThresholdCalibration
to automatically determine optimal confidence and prob thresholds
without requiring labeled data.

==============================================================================
使用说明 / Usage Guide:
==============================================================================

【基本用法 / Basic Usage】
    python eval/calibration_demo.py --config eval/configs/loveda.yaml

【常用参数 / Common Arguments】
    --num_samples N          校准使用的样本数量 (default: 50)
    --confidence_percentile P 置信度阈值百分位 (default: 30.0)
    --prob_percentile P      概率阈值百分位 (default: 50.0)
    --per_class_prob         使用每类独立的概率阈值
    --batch_size B           推理批次大小 (default: 4)
    --force_single_view      强制使用单图模式（大图需要sliding window时使用）

【不同数据集的使用场景 / Dataset-Specific Usage】

1. LoveDA (1024x1024, 小图) - 自动使用批量推理:
    python eval/calibration_demo.py --config eval/configs/loveda.yaml --num_samples 50

2. Potsdam (6000x6000, 大图) - 自动检测大图，使用单图+滑动窗口:
    python eval/calibration_demo.py --config eval/configs/potsdam.yaml --num_samples 24
   或手动强制单图模式:
    python demo_calibration.py --config eval/configs/potsdam.yaml --force_single_view --batch_size 1

【自动模式检测逻辑 / Auto-Detection Logic】
    - 默认使用批量推理(batch_size=4)，提供1.5-2x加速
    - 当图片尺寸 > slide_crop_size 时，自动切换到单图模式
    - 单图模式对大图使用sliding window处理
    - 可通过 --force_single_view 手动覆盖

【输出结果 / Output】
    结果保存在 test/calib/calibration_results.txt
    包含推荐的 confidence_threshold 和 prob_threshold(s)

==============================================================================
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from segmentor import SAM3RSSegmentor, InferenceConfig
from segmentor_lib.experimental.unsupervised_threshold import UnsupervisedThresholdCalibration


def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised Threshold Calibration Demo')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to dataset config file (e.g., eval/configs/loveda.yaml)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of samples for calibration (default: 50)'
    )
    parser.add_argument(
        '--confidence_percentile',
        type=float,
        default=30.0,
        help='Percentile for confidence threshold (default: 30.0)'
    )
    parser.add_argument(
        '--prob_percentile',
        type=float,
        default=50.0,
        help='Percentile for prob threshold (default: 50.0)'
    )
    parser.add_argument(
        '--per_class_prob',
        action='store_true',
        help='Use per-class prob thresholds instead of global'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test/calib/calibration_results.txt',
        help='Output file for calibration results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference (default: 4). Set to 1 for large images requiring sliding window.'
    )
    parser.add_argument(
        '--force_single_view',
        action='store_true',
        help='Force single-view mode (uses sliding window for large images). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--background_names',
        type=str,
        nargs='*',
        default=None,
        help='Class names to exclude from calibration (default: ["background","clutter"]). '
             'Set to empty (e.g. --background_names) to include all classes. '
             'For Potsdam where clutter is a valid foreground class, use --background_names to include it.'
    )
    parser.add_argument(
        '--no_exclude_background',
        action='store_true',
        help='Disable background exclusion entirely (include all classes in calibration)'
    )
    
    parser.add_argument(
        '--head_type',
        type=str,
        default='auto',
        choices=['auto', 'instance', 'semantic'],
        help='Head type for calibration strategy: instance (Non-Zero adaptive), semantic (Top 5%%), auto (detect from config)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Get script directory for path resolution
    script_dir = Path(__file__).parent
    SAM3_RS_DIR = script_dir

    # Process paths
    checkpoint_path = config_dict['segmentor']['checkpoint_path']
    bpe_path = config_dict['segmentor']['bpe_path']
    prompts_file = config_dict['dataset'].get('cls_file') or config_dict['segmentor'].get('prompts_file')

    # Convert to absolute paths
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(SAM3_RS_DIR / checkpoint_path)
    if not os.path.isabs(bpe_path):
        bpe_path = str(SAM3_RS_DIR / "sam3" / "assets" / bpe_path)
    if prompts_file and not os.path.isabs(prompts_file):
        # Try multiple possible locations
        possible_paths = [
            str(SAM3_RS_DIR / "eval" / "datasets" / prompts_file),
            str(SAM3_RS_DIR / "sam3" / "assets" / prompts_file),
            prompts_file  # Use as-is if it's a workspace data path
        ]
        prompts_file = None
        for p in possible_paths:
            if os.path.exists(p):
                prompts_file = p
                break

    # Create InferenceConfig
    inference_config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=config_dict['segmentor'].get('device', 'cuda'),
        confidence_threshold=config_dict['segmentor']['confidence_threshold'],
        # 重要：校准过程中不使用 pre-fusion thresholds，确保基于原始 logits 统计
        instance_prob_thresholds=None,
        semantic_prob_thresholds=None,
        use_semantic_head=config_dict['segmentor'].get('use_semantic_head', True),
        use_instance_head=config_dict['segmentor'].get('use_instance_head', True),
        use_presence_score=config_dict['segmentor'].get('use_presence_score', True),
        presence_score_mode=config_dict['segmentor'].get('presence_score_mode', 'before_fusion'),
        slide_crop_size=config_dict['segmentor'].get('slide_crop_size', 0),
        slide_stride=config_dict['segmentor'].get('slide_stride', 512),
        prompts_file=prompts_file,
        use_prompted_background=config_dict['segmentor'].get('use_prompted_background', False),
    )

    # Initialize segmentor
    print("\n" + "=" * 70)
    print("[Init] Loading segmentor...")
    print("=" * 70)
    segmentor = SAM3RSSegmentor(inference_config)

    # Get calibration images
    dataset_config = config_dict['dataset']
    data_root = dataset_config['data_root']
    img_dir = dataset_config['img_dir']
    images_dir = os.path.join(data_root, img_dir)

    # Collect all image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(images_dir).rglob(f'*{ext}'))

    image_paths = sorted([str(p) for p in image_paths])

    if len(image_paths) == 0:
        print(f"[Error] No images found in {images_dir}")
        return

    # Get prompts and num_classes
    prompts_file = dataset_config.get('prompts_file') or config_dict['segmentor'].get('prompts_file')
    if prompts_file:
        # Try to load from dataset folder first
        prompts_full_path = os.path.join(data_root, prompts_file)
        if not os.path.exists(prompts_full_path):
            # Try eval/datasets/ folder
            prompts_full_path = os.path.join('eval/datasets', prompts_file)
        if not os.path.exists(prompts_full_path):
            # Try sam3/assets folder
            prompts_full_path = os.path.join('sam3/assets', prompts_file)

        from segmentor_lib.prompts import load_prompts
        prompts = load_prompts(prompts_full_path)
        prompt_names = prompts['names']
        num_classes = len(set(prompts['indices']))
    else:
        # Use default prompts
        prompt_names = segmentor.prompts['names']
        num_classes = segmentor.num_classes

    # Auto-detect head_type if set to 'auto'
    head_type = args.head_type
    if head_type == 'auto':
        # 根据配置自动检测：如果只开 instance head，用 instance 策略；否则用 semantic
        use_instance = config_dict['segmentor'].get('use_instance_head', True)
        use_semantic = config_dict['segmentor'].get('use_semantic_head', True)
        if use_instance and not use_semantic:
            head_type = 'instance'
        else:
            head_type = 'semantic'  # 默认或 semantic only 都用 semantic 策略
        print(f"[Auto-detect] Head type: {head_type} (instance={use_instance}, semantic={use_semantic})")

    # Initialize calibrator with head_type
    calibrator = UnsupervisedThresholdCalibration(num_samples=args.num_samples, head_type=head_type)

    # Run calibration
    confidence_threshold, prob_thresholds = calibrator.calibrate(
        segmentor=segmentor,
        image_paths=image_paths,
        prompt_names=prompt_names,
        num_classes=num_classes,
        confidence_percentile=args.confidence_percentile,
        prob_percentile=args.prob_percentile,
        use_per_class_prob=args.per_class_prob,
        batch_size=args.batch_size,
        force_single_view=args.force_single_view,
        exclude_background=not args.no_exclude_background,
        background_names=args.background_names
    )

    # Determine threshold type based on head_type
    if args.head_type == "instance":
        threshold_key = "instance_prob_thresholds"
    else:
        threshold_key = "semantic_prob_thresholds"

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Unsupervised Threshold Calibration Results\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Config file: {args.config}\n")
        f.write(f"Head type: {args.head_type}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Confidence percentile: {args.confidence_percentile}%\n")
        f.write(f"Prob percentile: {args.prob_percentile}%\n")
        f.write(f"Per-class prob thresholds: {args.per_class_prob}\n\n")

        f.write(f"RECOMMENDED THRESHOLDS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"confidence_threshold: {confidence_threshold:.4f}\n")

        if args.per_class_prob and prob_thresholds:
            f.write(f"\n{threshold_key} (pre-fusion per-class):\n")
            for class_idx, threshold in prob_thresholds.items():
                prompt_name = prompt_names[class_idx] if class_idx < len(prompt_names) else f"class_{class_idx}"
                f.write(f"  {class_idx} ({prompt_name}): {threshold:.4f}\n")
        else:
            f.write(f"\nNote: Global threshold is deprecated. Use per-class thresholds.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("To use these thresholds:\n")
        f.write("-" * 70 + "\n")
        if args.per_class_prob and prob_thresholds:
            f.write(f"Add the following to your YAML config file:\n\n")
            f.write("segmentor:\n")
            f.write(f"  confidence_threshold: {confidence_threshold:.4f}\n")
            f.write(f"  {threshold_key}:\n")
            for class_idx, threshold in prob_thresholds.items():
                prompt_name = prompt_names[class_idx] if class_idx < len(prompt_names) else f"class_{class_idx}"
                f.write(f"    {class_idx}: {threshold:.4f}  # {prompt_name}\n")
        else:
            f.write("Warning: Global threshold mode is deprecated.\n")
            f.write("Please use --per_class_prob for per-class thresholds.\n")
        f.write("=" * 70 + "\n")

    print(f"\n[Output] Calibration results saved to: {output_path}")
    print(f"\n[Info] Update your config file with these thresholds for optimal results!")


if __name__ == '__main__':
    # python eval/calibration_demo.py --config eval/configs/potsdam_B2_calib.yaml  --num_samples 24 --per_class_prob --force_single_view --no_exclude_background  --output test/calib/potsdam_B2_calib.txt
    main()

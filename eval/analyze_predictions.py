#!/usr/bin/env python3
"""
分析预测结果，找出导致mIoU偏低的图像

功能：
1. 计算每张图像的mIoU和per-class IoU
2. 找出mIoU最低的N张图像
3. 找出特定类别IoU最低的图像
4. 生成可视化报告，标注问题图像

用法:
    python analyze_predictions.py --gt-dir data/LoveDA/Exp/mask \
                               --pred-dir outputs/preds/loveda \
                               --dataset loveda \
                               --output-dir outputs/analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入metric计算模块
from metrics.seg_metrics import SegmentationMetric
from colormaps import get_colormap


def compute_single_image_metrics(
    gt_path: str,
    pred_path: str,
    metric: SegmentationMetric,
    reduce_zero_label: bool = True,
    ignore_index: int = 255
) -> Tuple[float, Dict[str, float], Dict[str, any]]:
    """
    计算单张图像的指标

    Args:
        gt_path: GT mask路径（原始格式：0=no-data, 1=bg, 2=building...）
        pred_path: 预测mask路径（原始格式：0=no-data, 1=bg, 2=building...）
        metric: SegmentationMetric实例（已初始化）
        reduce_zero_label: 是否需要将标签reduce（与dataloader一致）
        ignore_index: 忽略的标签值

    Returns:
        (mIoU, per_class_iou, metadata)
    """
    # 读取图像名
    img_name = os.path.basename(gt_path)

    # 读取GT和预测（都是原始格式：0=no-data, 1=bg, 2=building...）
    gt_original = np.array(Image.open(gt_path))
    pred_original = np.array(Image.open(pred_path))

    # 确保形状一致
    if gt_original.shape != pred_original.shape:
        print(f"  Warning: {img_name} shape mismatch: GT {gt_original.shape} vs Pred {pred_original.shape}")
        pred_original = np.array(Image.fromarray(pred_original).resize(
            (gt_original.shape[1], gt_original.shape[0]), Image.NEAREST))

    # 处理reduce_zero_label（与dataloader一致）
    # 原始: 0=no-data, 1=bg, 2=building, 3=road, 4=water, 5=barren, 6=forest, 7=agriculture
    # Reduce后: 255=no-data(ignore), 0=bg, 1=building, 2=road, 3=water, 4=barren, 5=forest, 6=agriculture
    if reduce_zero_label:
        gt_shifted = gt_original - 1
        gt_shifted[gt_shifted == -1] = ignore_index

        pred_shifted = pred_original - 1
        pred_shifted[pred_shifted == -1] = ignore_index
    else:
        gt_shifted = gt_original
        pred_shifted = pred_original

    # 计算指标
    metric.reset()
    metric.update(pred_shifted, gt_shifted)

    # 获取结果
    mIoU = metric.compute()['mIoU']
    per_class_iou_array = metric.per_class_iou()

    # 转换为字典格式
    per_class_iou_dict = {i: float(per_class_iou_array[i]) for i in range(len(per_class_iou_array))}

    # 元数据
    metadata = {
        'image_name': img_name,
        'shape': gt_original.shape,
        'gt_unique': np.unique(gt_original),
        'pred_unique': np.unique(pred_original),
        'gt_pixels': np.sum(gt_original != 0),  # 去掉no-data后的像素数
        'gt_original': gt_original,  # 保存原始GT用于可视化
        'pred_original': pred_original,  # 保存原始pred用于可视化
    }

    return mIoU, per_class_iou_dict, metadata


def analyze_dataset(
    gt_dir: str,
    pred_dir: str,
    dataset_name: str,
    reduce_zero_label: bool = True,
    ignore_index: int = 255
) -> Dict[str, any]:
    """
    分析整个数据集的预测结果

    Args:
        gt_dir: GT mask目录
        pred_dir: 预测mask目录
        dataset_name: 数据集名称（用于获取colormap）
        reduce_zero_label: 是否需要reduce_zero_label
        ignore_index: 忽略的标签值

    Returns:
        分析结果字典
    """
    print("=" * 70)
    print("分析预测结果...")
    print("=" * 70)

    # 获取colormap
    colormap = get_colormap(dataset_name)
    class_names = {k: v['name'] for k, v in colormap.items()}

    # 对于LoveDA等reduce_zero_label的数据集：
    # - GT原始: 0=no-data, 1=background, 2=building, ...
    # - reduce后: 0=background, 1=building, 2=road, ... (no-data被设为255)
    # - metric期望的GT包含background类，所以use_prompted_background=True
    if reduce_zero_label:
        # 去掉no-data类别
        metric_num_classes = len(colormap) - 1
        use_prompted_bg = True  # GT包含background类
    else:
        metric_num_classes = len(colormap)
        use_prompted_bg = False  # GT不包含background类

    # 初始化metric
    metric = SegmentationMetric(
        num_classes=metric_num_classes,
        ignore_index=ignore_index,
        use_prompted_background=use_prompted_bg,
        bg_idx=0
    )

    # 收集所有图像的指标
    all_results = []

    # 找到所有pred文件（以pred为准，避免GT目录有更多文件）
    pred_files = sorted(list(Path(pred_dir).glob('*.png')))

    if len(pred_files) == 0:
        print(f"Error: No PNG files found in {pred_dir}")
        return None

    print(f"\n找到 {len(pred_files)} 张图像")
    print(f"GT目录: {gt_dir}")
    print(f"预测目录: {pred_dir}")
    print(f"类别数: {len(colormap)}")
    print(f"reduce_zero_label: {reduce_zero_label}")
    print()

    # 逐张分析（以pred文件为准）
    for idx, pred_path in enumerate(pred_files):
        gt_path = os.path.join(gt_dir, pred_path.name)

        if not os.path.exists(gt_path):
            print(f"  Warning: {pred_path.name} - GT文件不存在，跳过")
            continue

        mIoU, per_class_iou, metadata = compute_single_image_metrics(
            gt_path, str(pred_path), metric, reduce_zero_label, ignore_index
        )

        all_results.append({
            'mIoU': mIoU,
            'per_class_iou': per_class_iou,
            'metadata': metadata,
            'gt_path': str(gt_path),
            'pred_path': str(pred_path)
        })

        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx + 1}/{len(pred_files)}")

    print(f"\n✓ 分析完成！共处理 {len(all_results)} 张图像")

    # 计算整体统计
    overall_miou = np.mean([r['mIoU'] for r in all_results])

    # 排序
    all_results_sorted = sorted(all_results, key=lambda x: x['mIoU'])

    # 找出各类别IoU最低的图像
    worst_by_class = {}
    for class_id in range(metric_num_classes):  # 使用metric_num_classes，跳过no-data
        # reduce_zero_label后，class_id对应的是shifted后的类别
        # 需要映射回原始colormap的class_id用于可视化
        if reduce_zero_label:
            original_class_id = class_id + 1  # 0->1(background), 1->2(building), ...
        else:
            original_class_id = class_id

        class_name = class_names[original_class_id]

        # 筛选包含该类别的图像，且该类别的IoU不是特殊值（如1.0表示无该类别）
        results_with_class = []
        for r in all_results:
            if class_id in r['per_class_iou']:
                iou = r['per_class_iou'][class_id]
                # 排除特殊情况：IoU=1.0（可能表示该图像完全没有这个类别的像素）
                # 或者IoU=0.0（但可能是正常情况，保留）
                # 只排除确实没有这个类别的图像
                gt_has_class = (original_class_id in r['metadata']['gt_unique'])
                if gt_has_class or iou > 0:  # 要么GT有这个类别，要么IoU>0
                    results_with_class.append(r)

        if results_with_class:
            worst_by_class[class_id] = sorted(
                results_with_class,
                key=lambda x: x['per_class_iou'].get(class_id, 0)
            )[0]
            # 保存原始class_id用于可视化
            worst_by_class[class_id]['original_class_id'] = original_class_id
        else:
            # 如果没有找到该类别的图像，跳过
            print(f"  跳过 {class_name}: 没有找到包含该类别的有效图像")

    return {
        'overall_miou': overall_miou,
        'all_results': all_results,
        'all_results_sorted': all_results_sorted,
        'worst_by_class': worst_by_class,
        'colormap': colormap,
        'class_names': class_names,
        'dataset_name': dataset_name,
        'reduce_zero_label': reduce_zero_label,
        'num_images': len(all_results)
    }


def save_analysis_report(analysis: Dict, output_dir: str):
    """
    保存分析报告（JSON格式）
    """
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'dataset': analysis['dataset_name'],
        'overall_mIoU': float(analysis['overall_miou']),
        'num_images': analysis['num_images'],
        'reduce_zero_label': analysis['reduce_zero_label'],
        'class_names': analysis['class_names'],
    }

    # 保存Top-N最差的图像
    top_n = min(10, len(analysis['all_results_sorted']))
    report['worst_images'] = []
    for i in range(top_n):
        result = analysis['all_results_sorted'][i]
        report['worst_images'].append({
            'rank': i + 1,
            'image_name': result['metadata']['image_name'],
            'mIoU': float(result['mIoU']),
            'per_class_iou': result['per_class_iou'],
        })

    # 保存各类别最差的图像
    report['worst_by_class'] = {}
    for class_id, result in analysis['worst_by_class'].items():
        # class_id是shifted后的（0-6），需要映射到原始class_id获取class_name
        original_class_id = result['original_class_id']
        class_name = analysis['class_names'][original_class_id]
        report['worst_by_class'][class_name] = {
            'image_name': result['metadata']['image_name'],
            'IoU': float(result['per_class_iou'].get(class_id, 0)),
        }

    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 分析报告已保存: {report_path}")


def visualize_worst_images(analysis: Dict, output_dir: str, top_n: int = 10):
    """
    可视化最差的N张图像
    """
    print("\n" + "=" * 70)
    print("可视化最差图像...")
    print("=" * 70)

    viz_dir = os.path.join(output_dir, 'worst_images')
    os.makedirs(viz_dir, exist_ok=True)

    colormap = analysis['colormap']
    class_names = analysis['class_names']
    reduce_zero_label = analysis['reduce_zero_label']

    # 可视化Top-N最差的图像
    for i in range(min(top_n, len(analysis['all_results_sorted']))):
        result = analysis['all_results_sorted'][i]
        img_name = result['metadata']['image_name']
        mIoU = result['mIoU']

        print(f"  [{i+1}/{top_n}] {img_name} - mIoU: {mIoU:.4f}")

        # 使用保存的原始GT和pred（未reduce）
        gt = result['metadata']['gt_original']
        pred = result['metadata']['pred_original']

        # 转换为RGB（使用colormap）
        gt_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)

        for class_id, info in colormap.items():
            color = info['color']
            gt_rgb[gt == class_id] = color
            pred_rgb[pred == class_id] = color

        # 创建对比图
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(gt_rgb)
        axes[0].set_title(f'GT\n{img_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred_rgb)
        axes[1].set_title(f'Prediction\nmIoU: {mIoU:.4f}', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 错误热图 - 区分no-data、正确预测和错误区域
        error_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)

        if reduce_zero_label:
            # LoveDA: 0=no-data, 其他是类别
            no_data_mask = (gt == 0)
            correct_mask = (gt == pred) & (gt != 0)
            error_mask = (gt != pred) & (gt != 0)
        else:
            # 其他数据集: 可能没有no-data
            no_data_mask = np.zeros_like(gt, dtype=bool)
            correct_mask = (gt == pred)
            error_mask = (gt != pred)

        # 灰色 = no-data区域
        error_rgb[no_data_mask] = [128, 128, 128]
        # 绿色 = 正确预测
        error_rgb[correct_mask] = [0, 255, 0]
        # 红色 = 错误预测
        error_rgb[error_mask] = [255, 0, 0]

        axes[2].imshow(error_rgb)
        axes[2].set_title(f'Error Map (Overlay)\nGray=No-data, Green=Correct, Red=Error',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # 原始错误热图（仅显示错误区域）
        error_only_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
        error_only_rgb[error_mask] = [255, 0, 0]
        axes[3].imshow(error_only_rgb)
        axes[3].set_title(f'Error Only\nRed = Wrong Prediction (excl. no-data)',
                         fontsize=12, fontweight='bold')
        axes[3].axis('off')

        # 添加per-class IoU信息
        iou_text = "Per-class IoU:\n"
        for class_id, iou in result['per_class_iou'].items():
            # per_class_iou中的key是shifted后的class_id
            # 需要映射回原始colormap的class_id获取class name
            if reduce_zero_label:
                original_class_id = class_id + 1
            else:
                original_class_id = class_id
            name = class_names[original_class_id]
            iou_text += f"  {name}: {iou:.3f}\n"
        fig.text(0.02, 0.02, iou_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(viz_dir, f'worst_{i+1:03d}_{img_name}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ 已保存 {min(top_n, len(analysis['all_results_sorted']))} 张最差图像的可视化")


def visualize_class_worst_images(analysis: Dict, output_dir: str):
    """
    可视化各类别IoU最低的图像
    """
    print("\n" + "=" * 70)
    print("可视化各类别最差图像...")
    print("=" * 70)

    viz_dir = os.path.join(output_dir, 'worst_by_class')
    os.makedirs(viz_dir, exist_ok=True)

    colormap = analysis['colormap']
    class_names = analysis['class_names']
    reduce_zero_label = analysis['reduce_zero_label']

    for class_id, result in analysis['worst_by_class'].items():
        # class_id是shifted后的（metric计算用的）
        # 需要用original_class_id来获取原始的类别名称和colormap
        original_class_id = result['original_class_id']
        class_name = class_names[original_class_id]
        iou = result['per_class_iou'].get(class_id, 0)
        img_name = result['metadata']['image_name']

        print(f"  {class_name}: {img_name} - IoU: {iou:.4f}")

        # 使用保存的原始GT和pred（未reduce）
        gt = result['metadata']['gt_original']
        pred = result['metadata']['pred_original']

        # 只显示该类别的区域（使用原始class_id）
        class_mask_gt = (gt == original_class_id)
        class_mask_pred = (pred == original_class_id)

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. GT - 该类别的mask（二值图）
        axes[0].imshow(class_mask_gt, cmap='gray')
        axes[0].set_title(f'GT - {class_name} (Binary)\nWhite = {class_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 2. Prediction - 该类别的mask（二值图）
        axes[1].imshow(class_mask_pred, cmap='gray')
        axes[1].set_title(f'Prediction - {class_name} (Binary)\nIoU: {iou:.4f}', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 3. 误检/漏检分析
        false_negative = class_mask_gt & (~class_mask_pred)  # 漏检
        false_positive = (~class_mask_gt) & class_mask_pred  # 误检
        error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
        error_map[false_negative] = [255, 0, 0]    # 红色 = 漏检
        error_map[false_positive] = [0, 0, 255]    # 蓝色 = 误检

        axes[2].imshow(error_map)
        axes[2].set_title(f'Error Analysis - {class_name}\nRed=漏检, Blue=误检',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(viz_dir, f'class_{original_class_id:02d}_{class_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ 已保存 {len(analysis['worst_by_class'])} 个类别的最差图像")


def print_summary(analysis: Dict):
    """
    打印分析摘要
    """
    print("\n" + "=" * 70)
    print("分析摘要")
    print("=" * 70)

    print(f"\n数据集: {analysis['dataset_name']}")
    print(f"图像数量: {analysis['num_images']}")
    print(f"整体mIoU: {analysis['overall_miou']:.4f}\n")

    # Top-10最差图像
    print("Top-10 最差图像 (mIoU最低):")
    print("-" * 70)
    for i in range(min(10, len(analysis['all_results_sorted']))):
        result = analysis['all_results_sorted'][i]
        img_name = result['metadata']['image_name']
        mIoU = result['mIoU']
        print(f"  {i+1:2d}. {img_name:30s} mIoU: {mIoU:.4f}")

    # 各类别IoU分布
    print("\n各类别IoU最低的图像:")
    print("-" * 70)
    for class_id, result in analysis['worst_by_class'].items():
        # class_id是shifted后的（0-6），需要映射到原始class_id获取class_name
        original_class_id = result['original_class_id']
        class_name = analysis['class_names'][original_class_id]
        iou = result['per_class_iou'].get(class_id, 0)
        img_name = result['metadata']['image_name']
        print(f"  {class_name:15s}: {img_name:30s} IoU: {iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description='分析预测结果，找出导致mIoU偏低的图像')
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='GT mask目录')
    parser.add_argument('--pred-dir', type=str, required=True,
                       help='预测mask目录')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['loveda', 'openearthmap', 'potsdam', 'vaihingen',
                                'uavid', 'udd5', 'isaid', 'vdd'],
                       help='数据集名称（用于获取colormap）')
    parser.add_argument('--reduce-zero-label', action='store_true', default=True,
                       help='是否需要reduce_zero_label（LoveDA=True）')
    parser.add_argument('--ignore-index', type=int, default=255,
                       help='忽略的标签值')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis',
                       help='输出目录')
    parser.add_argument('--top-n', type=int, default=10,
                       help='可视化Top-N最差图像')
    parser.add_argument('--no-viz', action='store_true',
                       help='跳过可视化（只生成报告）')

    args = parser.parse_args()

    # 分析
    analysis = analyze_dataset(
        args.gt_dir,
        args.pred_dir,
        args.dataset,
        args.reduce_zero_label,
        args.ignore_index
    )

    if analysis is None:
        return

    # 打印摘要
    print_summary(analysis)

    # 保存报告
    save_analysis_report(analysis, args.output_dir)

    # 可视化
    if not args.no_viz:
        visualize_worst_images(analysis, args.output_dir, args.top_n)
        visualize_class_worst_images(analysis, args.output_dir)

    print("\n" + "=" * 70)
    print("✓ 分析完成！")
    print(f"✓ 结果保存在: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # python eval/analyze_predictions.py --gt-dir data/LoveDA/Exp/mask --pred-dir outputs/preds/loveda --dataset loveda --output-dir outputs/analysis
    main()

"""
可视化对比工具使用示例

本脚本演示如何使用 visualization.py 进行 GT mask 和预测 mask 的可视化对比

输出：左右并排的 GT 和预测 mask，包含类别图例
"""

from pathlib import Path
import sys

# 添加 eval 目录到路径
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from visualization import visualize, batch_visualize
from colormaps import get_colormap, load_colormap_from_json


def example_loveda_single():
    """示例 1: LoveDA 单张图片可视化（使用预定义颜色映射）"""
    print("=" * 80)
    print("示例 1: LoveDA 单张图片可视化")
    print("=" * 80)

    # 假设你有以下文件
    gt_mask_path = "path/to/gt_mask.png"
    pred_mask_path = "path/to/pred_mask.png"
    output_path = "path/to/comparison.png"

    # 获取 LoveDA 颜色映射
    colormap = get_colormap("loveda")  # 或 "loveda_pred"

    # 可视化（生成左右并排对比图）
    visualize(
        gt_mask_path,
        pred_mask_path,
        output_path,
        colormap=colormap
    )


def example_loveda_batch():
    """示例 2: LoveDA 批量可视化"""
    print("=" * 80)
    print("示例 2: LoveDA 批量可视化")
    print("=" * 80)

    # 假设你的预测结果保存在 outputs/preds/loveda/
    gt_mask_dir = "path/to/LoveDA/Val/Rural/masks_png"
    pred_mask_dir = "outputs/preds/loveda"
    output_dir = "outputs/visualizations/loveda"

    # 获取颜色映射（使用预测版本）
    colormap = get_colormap("loveda_pred")

    # 批量处理
    results = batch_visualize(
        gt_mask_dir,
        pred_mask_dir,
        output_dir,
        colormap=colormap
    )

    print(f"\n生成了 {len(results)} 张可视化图片")


def example_custom_dataset():
    """示例 3: 自定义数据集可视化（直接定义颜色映射）"""
    print("=" * 80)
    print("示例 3: 自定义数据集")
    print("=" * 80)

    # 1. 定义你的颜色映射
    custom_colormap = {
        0: {"name": "background", "color": [255, 255, 255]},
        1: {"name": "building", "color": [255, 0, 0]},
        2: {"name": "vegetation", "color": [0, 255, 0]},
        3: {"name": "water", "color": [0, 0, 255]},
        255: {"name": "ignore", "color": [0, 0, 0]},
    }

    # 2. 可视化对比
    visualize(
        "path/to/gt_mask.png",
        "path/to/pred_mask.png",
        "path/to/comparison.png",
        colormap=custom_colormap,
        ignore_index=255
    )


def example_custom_dataset_json():
    """示例 4: 从 JSON 加载自定义颜色映射"""
    print("=" * 80)
    print("示例 4: 从 JSON 加载配置")
    print("=" * 80)

    # 从 JSON 文件加载颜色映射
    config_path = "eval/configs/class_info_example.json"
    colormap = load_colormap_from_json(config_path)

    print(f"✓ 从 {config_path} 加载了 {len(colormap)} 个类别的配置")

    # 使用该颜色映射进行可视化
    visualize(
        "path/to/gt_mask.png",
        "path/to/pred_mask.png",
        "path/to/comparison.png",
        colormap=colormap
    )


def example_add_new_dataset():
    """示例 5: 添加新数据集的颜色映射"""
    print("=" * 80)
    print("示例 5: 添加新数据集颜色映射")
    print("=" * 80)

    # 方式 1: 在 colormaps/__init__.py 中添加
    print("方式 1: 在 colormaps/__init__.py 中添加预定义映射")
    print("""
# 在 colormaps/__init__.py 中添加：
MY_DATASET = {
    0: {"name": "class0", "color": [255, 255, 255]},
    1: {"name": "class1", "color": [255, 0, 0]},
    ...
}

COLORMAPS = {
    ...
    "my_dataset": MY_DATASET,
}
    """)

    # 方式 2: 使用 JSON 文件
    print("\n方式 2: 创建 JSON 文件")
    print("""
# 创建 my_dataset.json:
{
  "0": {"name": "class0", "color": [255, 255, 255]},
  "1": {"name": "class1", "color": [255, 0, 0]},
  ...
}

# 使用:
colormap = load_colormap_from_json("my_dataset.json")
visualize(gt_path, pred_path, output_path, colormap=colormap)
    """)


def example_complete_workflow():
    """示例 6: 结合 run_eval.py 完整工作流"""
    print("=" * 80)
    print("示例 6: 结合 run_eval.py 完整工作流")
    print("=" * 80)

    print("步骤 1: 运行评估并保存预测结果")
    print("python core/sam3-rs/eval/run_eval.py \\")
    print("  --config core/sam3-rs/eval/configs/loveda.yaml \\")
    print("  --device cuda:0 \\")
    print("  --save-pred outputs/preds/loveda")
    print()

    print("步骤 2: 可视化对比结果（命令行方式）")
    print("python core/sam3-rs/eval/visualization.py \\")
    print("  --gt path/to/LoveDA/Val/Rural/masks_png \\")
    print("  --pred outputs/preds/loveda \\")
    print("  --output outputs/visualizations/loveda \\")
    print("  --colormap loveda")
    print()

    print("或者使用 Python API:")
    print("""
from visualization import batch_visualize
from colormaps import get_colormap

colormap = get_colormap("loveda")
batch_visualize(
    gt_mask_dir="path/to/gt_masks/",
    pred_mask_dir="outputs/preds/loveda/",
    output_dir="outputs/visualizations/loveda/",
    colormap=colormap
)
    """)
    print()

    print("✓ 步骤 1 会在 outputs/preds/loveda/ 生成预测 mask")
    print("✓ 步骤 2 会在 outputs/visualizations/loveda/ 生成对比图")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("可视化对比工具使用示例")
    print("=" * 80)
    print()
    print("请取消注释以下任一示例函数来运行：")
    print()
    print("  - example_loveda_single()      # LoveDA 单张图片")
    print("  - example_loveda_batch()       # LoveDA 批量处理")
    print("  - example_custom_dataset()       # 自定义数据集（直接定义）")
    print("  - example_custom_dataset_json()  # 自定义数据集（JSON）")
    print("  - example_add_new_dataset()     # 添加新数据集颜色映射")
    print("  - example_complete_workflow()   # 完整工作流")
    print()
    print("=" * 80)
    print()

    # 取消注释以运行示例
    # example_loveda_single()
    # example_loveda_batch()
    # example_custom_dataset()
    # example_custom_dataset_json()
    # example_add_new_dataset()
    # example_complete_workflow()

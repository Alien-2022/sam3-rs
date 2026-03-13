"""
通用语义分割可视化对比工具

支持：
- 可视化对比 GT mask 和预测 mask
- 自定义颜色表（适配不同数据集）
- 生成并排对比图
- 保存为 PNG 格式
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 导入颜色映射表配置
try:
    from .colormaps import COLORMAPS, get_colormap, load_colormap_from_json
except ImportError:
    # 支持直接运行脚本
    from colormaps import COLORMAPS, get_colormap, load_colormap_from_json

# 设置字体支持（优先使用中文字体，回退到 DejaVu Sans）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Serif CJK JP', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SegmentationVisualizer:
    """语义分割结果可视化器"""

    def __init__(
        self,
        class_info: Dict[int, Dict[str, any]],
        ignore_index: int = 255,
        output_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            class_info: 类别信息字典，格式为 {
                0: {"name": "class_name", "color": [R, G, B]},
                1: {"name": "class_name", "color": [R, G, B]},
                ...
            }
            ignore_index: 忽略的标签值（如 255）
            output_size: 输出图像尺寸 (width, height)，None 表示使用原图尺寸
        """
        self.class_info = class_info
        self.ignore_index = ignore_index
        self.output_size = output_size

        # 创建颜色映射
        self.num_classes = max(class_info.keys()) + 1
        self.cmap = self._create_colormap()

    def _create_colormap(self) -> ListedColormap:
        """创建 matplotlib colormap"""
        colors = np.zeros((self.num_classes, 4))

        for class_id, info in self.class_info.items():
            colors[class_id] = info["color"] + [255]  # RGBA

        # 对于 ignore_index 设置透明或黑色
        if self.ignore_index < len(colors):
            colors[self.ignore_index] = [0, 0, 0, 0]

        return ListedColormap(colors / 255.0)

    def _apply_colormap(self, mask: np.ndarray) -> np.ndarray:
        """应用颜色映射到 mask"""
        # 处理 ignore_index
        mask_display = mask.copy()
        mask_display[mask == self.ignore_index] = -1

        # 创建 RGB 图像
        rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for class_id, info in self.class_info.items():
            mask_class = (mask == class_id)
            rgb_image[mask_class] = info["color"]

        # ignore_index 设置为透明（黑色）
        mask_ignore = (mask == self.ignore_index)
        rgb_image[mask_ignore] = [0, 0, 0]

        return rgb_image

    def load_mask(self, mask_path: str) -> np.ndarray:
        """加载 mask 文件"""
        mask = np.array(Image.open(mask_path))

        # 处理特殊情况：如果是 3 通道 RGB，转换为灰度
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return mask.astype(np.int32)

    def visualize_comparison(
        self,
        gt_mask_path: str,
        pred_mask_path: str,
        output_path: str,
        show_legend: bool = True,
        dpi: int = 150
    ) -> None:
        """
        可视化对比 GT 和预测 mask

        Args:
            gt_mask_path: GT mask 文件路径
            pred_mask_path: 预测 mask 文件路径
            output_path: 输出图片保存路径
            show_legend: 是否显示图例
            dpi: 输出图像分辨率
        """
        # 加载 masks
        gt_mask = self.load_mask(gt_mask_path)
        pred_mask = self.load_mask(pred_mask_path)

        # 应用颜色映射
        gt_rgb = self._apply_colormap(gt_mask)
        pred_rgb = self._apply_colormap(pred_mask)

        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=dpi)

        # GT Mask
        axes[0].imshow(gt_rgb)
        axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Predicted Mask
        axes[1].imshow(pred_rgb)
        axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')


        if show_legend:
            self._add_legend(fig)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved to: {output_path}")

    def _add_legend(self, fig) -> None:
        """添加图例"""
        # 准备图例项
        class_names = []
        colors = []

        for class_id in sorted(self.class_info.keys()):
            info = self.class_info[class_id]
            class_names.append(f"{class_id}: {info['name']}")
            colors.append(np.array(info['color']) / 255.0)

        # 创建图例
        patches = []
        for name, color in zip(class_names, colors):
            from matplotlib.patches import Patch
            patch = Patch(color=color, label=name)
            patches.append(patch)

        fig.legend(
            handles=patches,
            loc='center',
            bbox_to_anchor=(0.5, -0.05), # 表示水平居中，位于图表下方 15% 处
            fontsize=10,
            framealpha=0.9,
            ncol=len(patches)  # 横向排布所有图例项
        )

    def batch_visualize(
        self,
        gt_mask_dir: str,
        pred_mask_dir: str,
        output_dir: str,
        show_legend: bool = True
    ) -> Dict[str, str]:
        """
        批量可视化对比

        Args:
            gt_mask_dir: GT mask 目录
            pred_mask_dir: 预测 mask 目录
            output_dir: 输出目录
            show_legend: 是否显示图例

        Returns:
            生成的文件路径字典 {input_name: output_path}
        """
        gt_dir = Path(gt_mask_dir)
        pred_dir = Path(pred_mask_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 获取匹配的文件
        gt_files = sorted(gt_dir.glob("*.png"))
        pred_files = sorted(pred_dir.glob("*.png"))

        gt_names = {f.stem: f for f in gt_files}
        pred_names = {f.stem: f for f in pred_files}

        common_names = sorted(set(gt_names.keys()) & set(pred_names.keys()))

        if not common_names:
            raise ValueError(
                f"No matching files found between {gt_dir} and {pred_dir}"
            )

        results = {}
        for name in common_names:
            gt_path = gt_names[name]
            pred_path = pred_names[name]
            output_path = out_dir / f"{name}_comparison.png"

            self.visualize_comparison(
                str(gt_path),
                str(pred_path),
                str(output_path),
                show_legend=show_legend
            )

            results[name] = str(output_path)

        print(f"\n✓ Completed {len(results)} visualizations")
        print(f"✓ Output directory: {out_dir}")

        return results


# ============ 通用可视化函数 ============

def visualize(
    gt_mask_path: str,
    pred_mask_path: str,
    output_path: str,
    colormap: Dict[int, Dict[str, any]],
    ignore_index: int = 255,
    show_legend: bool = True,
    dpi: int = 150
) -> None:
    """
    通用的语义分割可视化函数（数据集无关）

    Args:
        gt_mask_path: GT mask 文件路径
        pred_mask_path: 预测 mask 文件路径
        output_path: 输出图片保存路径
        colormap: 颜色映射表 {class_id: {"name": "xxx", "color": [R, G, B]}}
        ignore_index: 忽略的标签值
        show_legend: 是否显示图例
        dpi: 输出图像分辨率

    Example:
        >>> from visualization import visualize
        >>> from colormaps import get_colormap
        >>> colormap = get_colormap("loveda")
        >>> visualize("gt.png", "pred.png", "comparison.png", colormap)
    """
    visualizer = SegmentationVisualizer(
        class_info=colormap,
        ignore_index=ignore_index
    )

    visualizer.visualize_comparison(
        gt_mask_path,
        pred_mask_path,
        output_path,
        show_legend=show_legend,
        dpi=dpi
    )


def batch_visualize(
    gt_mask_dir: str,
    pred_mask_dir: str,
    output_dir: str,
    colormap: Dict[int, Dict[str, any]],
    ignore_index: int = 255,
    show_legend: bool = True
) -> Dict[str, str]:
    """
    通用的批量可视化函数（数据集无关）

    Args:
        gt_mask_dir: GT mask 目录
        pred_mask_dir: 预测 mask 目录
        output_dir: 输出目录
        colormap: 颜色映射表 {class_id: {"name": "xxx", "color": [R, G, B]}}
        ignore_index: 忽略的标签值
        show_legend: 是否显示图例

    Returns:
        生成的文件路径字典 {input_name: output_path}

    Example:
        >>> from visualization import batch_visualize
        >>> from colormaps import get_colormap
        >>> colormap = get_colormap("loveda")
        >>> results = batch_visualize("gt_dir/", "pred_dir/", "output_dir/", colormap)
    """
    visualizer = SegmentationVisualizer(
        class_info=colormap,
        ignore_index=ignore_index
    )

    return visualizer.batch_visualize(
        gt_mask_dir,
        pred_mask_dir,
        output_dir,
        show_legend=show_legend
    )


# ============ 命令行接口 ============

def main():
    import argparse

    parser = argparse.ArgumentParser(description="语义分割可视化对比工具")
    parser.add_argument("--gt", type=str, required=True, help="GT mask 路径或目录")
    parser.add_argument("--pred", type=str, required=True, help="预测 mask 路径或目录")
    parser.add_argument("--output", type=str, required=True, help="输出路径或目录")
    parser.add_argument("--colormap", type=str, default="loveda",
                        help="颜色映射表名称（如 loveda, loveda_pred）或 custom")
    parser.add_argument("--colormap-file", type=str, default=None,
                        help="自定义颜色映射 JSON 文件路径（仅当 colormap=custom 时使用）")
    parser.add_argument("--ignore-index", type=int, default=255,
                        help="忽略的标签值（默认 255）")
    parser.add_argument("--list-colormaps", action="store_true",
                        help="列出所有可用的颜色映射表")

    args = parser.parse_args()

    # 列出颜色映射表
    if args.list_colormaps:
        print("可用的颜色映射表:")
        for name in sorted(COLORMAPS.keys()):
            print(f"  - {name}")
        return

    # 获取颜色映射表
    if args.colormap == "custom":
        if args.colormap_file is None:
            raise ValueError("自定义颜色映射需要提供 --colormap-file 参数")
        colormap = load_colormap_from_json(args.colormap_file)
    else:
        colormap = get_colormap(args.colormap)

    # 判断是单文件还是批量
    gt_is_dir = Path(args.gt).is_dir()
    pred_is_dir = Path(args.pred).is_dir()

    if gt_is_dir != pred_is_dir:
        raise ValueError("GT 和预测必须同为目录或同为文件")

    if gt_is_dir:
        # 批量处理
        batch_visualize(
            args.gt,
            args.pred,
            args.output,
            colormap=colormap,
            ignore_index=args.ignore_index
        )
    else:
        # 单张处理
        visualize(
            args.gt,
            args.pred,
            args.output,
            colormap=colormap,
            ignore_index=args.ignore_index
        )


if __name__ == "__main__":
    r"""
    # 查看可用颜色映射
    python visualization.py --list-colormaps
    # LoveDA 单张
    python visualization.py --gt gt.png --pred pred.png --output out.png --colormap loveda
    python eval/visualization.py --gt data/LoveDA/Exp/mask/2522.png --pred outputs/preds/loveda/2522.png --output outputs/compare/2522.png --colormap loveda
    # LoveDA 批量
    python visualization.py --gt gt_dir/ --pred pred_dir/ --output out_dir/ --colormap loveda_pred
    python eval/visualization.py --gt data/isAID/Exp/mask --pred outputs/preds/isAID --output outputs/compare --colormap isaid
    python eval/visualization.py --gt data/Potsdam/pre_slice/mask --pred outputs/preds/potsdam --output outputs/compare --colormap potsdam
    # 自定义数据集
    python visualization.py --gt gt.png --pred pred.png --output out.png --colormap custom --colormap-file custom.json
    """
    main()

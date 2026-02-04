# SAM3-RS Eval (Lightweight)

简洁版评测入口，默认无 mmengine 依赖，支持本地路径与 NAS 注册表路径。

## 目录
- `run_eval.py`：评测入口
- `configs/`：示例 YAML 配置（含 LoveDA），支持 `base_config` 继承通用默认
- `datasets/`：数据集加载（当前包含 LoveDA）
- `metrics/`：流式 mIoU/mAcc/aAcc 指标
- `colormaps/`：颜色映射表配置（支持扩展新数据集）
- `visualization.py`：可视化对比工具（GT vs 预测）

## 快速开始
```bash
python core/sam3-rs/eval/run_eval.py \
  --config core/sam3-rs/eval/configs/loveda.yaml \
  --device cuda:0 \
  --save-pred outputs/loveda_preds
```
参数说明：
- `--config`：YAML 路径
- `--device`：覆盖设备（如 `cuda:0`）
- `--save-pred`：可选，保存预测 PNG
- `--prob-threshold`：可选，覆盖前景阈值

## 配置要点（loveda.yaml）
- `dataset`
  - `data_root`：本地/云端数据根
  - `img_dir`/`mask_dir`：可为路径或列表/通配符，按 `data_root` 拼接，如 LoveDA 官方布局可用 `Val/Urban/images_png/*.png` 与 `Val/Rural/...`
  - `cls_file`：类名列表
  - `ignore_index`：忽略标签的哨兵值（LoveDA 推荐 255，uint8 掩码中不会与有效类冲突）
  - `reduce_zero_label`：是否将标签整体减 1 并把原始 0 设为忽略（适配原始 0=无数据，1..7 为有效类的标注）
  - `data_key`（可选）：若提供且存在 `workspace.scripts.get_data.get_data_path`，将用注册表返回的路径覆盖 `data_root`
- `segmentor`
  - `checkpoint_path`：权重文件路径
  - `bpe_path`：BPE 词表路径
  - `prompts_file`：提示词/类别文件
  - `bg_idx` / `use_prompted_background`：背景通道控制
    - `bg_idx`：输出 mask 中背景类的标签值（应与 GT mask 一致）
    - `use_prompted_background`：背景处理方式
      - `false` (默认)：SAM 不分割背景，通过 `prob_threshold` 过滤低置信度像素归为 `bg_idx`
      - `true`：SAM 显式分割背景类别，`bg_idx` 应与 prompts.txt 中 background 行的索引一致
  - `slide_crop_size` / `slide_stride`：滑窗配置（0 关闭滑窗）
  - `weight_key`（可选）：若提供且存在 `workspace.scripts.get_weights.get_weight_path`，将用注册表路径 + `checkpoint_filename` 覆盖 `checkpoint_path`
  - `checkpoint_filename`（可选）：与 `weight_key` 搭配使用
- `output`
  - `save_pred_dir`：可选，保存预测 PNG
  - `metrics_json`：保存指标 JSON

  ### base_config 继承
  - 在子配置中写 `base_config: ./base.yaml`，其余字段覆盖同名键。
  - `base.yaml` 提供 dataloader/segmentor/output 的常用默认值，可按需新增全局默认。

## NAS / 本地路径策略
1) 优先解析为相对配置文件目录；
2) 若不存在则尝试 workspace 根；
3) 若配置提供 `data_key`/`weight_key` 且注册表函数可用，则用注册表返回的路径覆盖。

## 扩展新数据集
1) 在 `configs/` 复制一份 YAML，改 `data_root/img_dir/mask_dir/cls_file`（或设置 `data_key`）。
2) 如类别文件不同，替换 `prompts_file`/`cls_file`。
3) 若目录结构与 LoveDA 不同，可按需新增 Dataset（放入 `datasets/`）并在入口替换。
4) 在 `datasets/__init__.py` 的 `DATASET_REGISTRY` 中注册新数据集。
5) 如需可视化，在 `colormaps/__init__.py` 中添加颜色映射。

### 已支持的数据集
- **LoveDA**: 城市场景语义分割（7类）
- **OpenEarthMap**: 地物分类（8类）
- **iSAID**: 遥感航拍目标检测/分割（15类）
- **Potsdam**: 高分辨率遥感图像分割（6类）
- **Vaihingen**: 高分辨率遥感图像分割（6类，与 Potsdam 类别相同）
- **UAVid**: 无人机视频语义分割（8类，包含 moving car 和 static car）

## 指标
- `mIoU` / `mAcc` / `aAcc`，流式混淆矩阵实现，内存占用低。
- 终端与 `metrics_json` 会输出逐类 IoU（键为类名）；内部使用 `SegmentationMetric.per_class_iou()`。

## 可视化对比
提供 GT mask 与预测 mask 的可视化对比功能，支持多种数据集。

### 快速使用

**查看可用颜色映射**
```bash
python core/sam3-rs/eval/visualization.py --list-colormaps
```

**LoveDA 数据集**
```bash
# 单张图片对比（使用 GT 原始标签 0-7）
python core/sam3-rs/eval/visualization.py \
  --gt /path/to/gt_mask.png \
  --pred /path/to/pred_mask.png \
  --output /path/to/comparison.png \
  --colormap loveda

# 批量对比（使用预测标签 0-6）
python core/sam3-rs/eval/visualization.py \
  --gt /path/to/gt_masks/ \
  --pred /path/to/pred_masks/ \
  --output /path/to/output_dir/ \
  --colormap loveda_pred
```

**自定义数据集**
```bash
# 1. 创建颜色映射 JSON（参考 configs/class_info_example.json）
# 2. 运行可视化
python core/sam3-rs/eval/visualization.py \
  --gt /path/to/gt_masks/ \
  --pred /path/to/pred_masks/ \
  --output /path/to/output_dir/ \
  --colormap custom \
  --colormap-file /path/to/custom_colormap.json
```

### Python API

```python
from visualization import visualize, batch_visualize
from colormaps import get_colormap, load_colormap_from_json

# 使用预定义颜色映射
colormap = get_colormap("loveda")
visualize("gt.png", "pred.png", "comparison.png", colormap=colormap)

# 从 JSON 加载自定义颜色映射
colormap = load_colormap_from_json("custom_colormap.json")
batch_visualize("gt_dir/", "pred_dir/", "output_dir/", colormap=colormap)
```

### 颜色映射格式

```json
{
  "0": {"name": "background", "color": [255, 255, 255]},
  "1": {"name": "building", "color": [128, 0, 0]},
  "2": {"name": "road", "color": [128, 128, 128]},
  ...
}
```

### 添加新数据集颜色映射

在 `colormaps/__init__.py` 中添加：
```python
MY_DATASET = {
    0: {"name": "class0", "color": [255, 255, 255]},
    1: {"name": "class1", "color": [255, 0, 0]},
    ...
}

COLORMAPS = {
    ...
    "my_dataset": MY_DATASET,
}
```

### 输出说明
- 生成 2 子图对比：GT、预测
- 包含完整类别图例
- 支持批量处理整个目录

## 依赖
- PyTorch、Pillow、PyYAML、NumPy
- SAM3-RS 自身依赖（见项目 `requirements.txt`）

## 已知默认
- LoveDA 标签 0 视为无效区；`reduce_zero_label=True` 时把有效类整体减 1 并将 0 设为 `ignore_index`。
- 滑窗模式仅返回融合 logits，未累积各头的中间 logits，以节省显存。

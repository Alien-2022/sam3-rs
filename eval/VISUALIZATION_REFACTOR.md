# 可视化工具重构说明

## 改进概述

根据建议，对可视化工具进行了重构，主要改进：

1. **统一颜色映射管理**：将所有颜色映射表集中到 `colormaps/` 模块
2. **数据集无关设计**：移除数据集特定的函数（如 `visualize_loveda`），统一使用 `visualize` 和 `batch_visualize`
3. **易于扩展**：添加新数据集只需在 `colormaps/__init__.py` 中定义颜色映射

## 文件结构

```
eval/
├── colormaps/
│   └── __init__.py          # 颜色映射表集中管理
├── visualization.py          # 统一的可视化接口
├── visualization_example.py   # 使用示例
├── configs/
│   └── class_info_example.json  # 自定义颜色映射示例
└── README.md               # 使用文档
```

## 核心改进

### 1. 颜色映射表集中管理

**旧方式**：颜色映射散落在各个函数中
```python
# 在 visualization.py 中
LOVEDA_CLASS_INFO = {...}
```

**新方式**：集中到 `colormaps/__init__.py`
```python
# 在 colormaps/__init__.py 中
LOVEDA = {...}
LOVEDA_PRED = {...}
COLORMAPS = {
    "loveda": LOVEDA,
    "loveda_pred": LOVEDA_PRED,
}
```

### 2. 统一的可视化接口

**旧方式**：每个数据集一个函数
```python
visualize_loveda(gt, pred, output, is_gt_original=True)
batch_visualize_loveda(gt_dir, pred_dir, output_dir, is_gt_original=True)
```

**新方式**：数据集无关的统一接口
```python
from visualization import visualize, batch_visualize
from colormaps import get_colormap

colormap = get_colormap("loveda")
visualize(gt, pred, output, colormap=colormap)
batch_visualize(gt_dir, pred_dir, output_dir, colormap=colormap)
```

### 3. 易于扩展新数据集

**步骤 1**：在 `colormaps/__init__.py` 中添加
```python
MY_DATASET = {
    0: {"name": "class0", "color": [255, 255, 255]},
    1: {"name": "class1", "color": [255, 0, 0]},
}

COLORMAPS = {
    "loveda": LOVEDA,
    "my_dataset": MY_DATASET,
}
```

**步骤 2**：直接使用
```python
colormap = get_colormap("my_dataset")
visualize(gt, pred, output, colormap=colormap)
```

## 使用对比

### 命令行使用

**旧方式**：
```bash
python visualization.py --gt gt.png --pred pred.png --output out.png --dataset loveda --is-gt-original
```

**新方式**：
```bash
python visualization.py --gt gt.png --pred pred.png --output out.png --colormap loveda
```

### Python API

**旧方式**：
```python
from visualization import visualize_loveda
visualize_loveda("gt.png", "pred.png", "out.png", is_gt_original=True)
```

**新方式**：
```python
from visualization import visualize
from colormaps import get_colormap

colormap = get_colormap("loveda")
visualize("gt.png", "pred.png", "out.png", colormap=colormap)
```

## 新功能

### 1. 列出可用颜色映射
```bash
python visualization.py --list-colormaps
```

### 2. 从 JSON 加载自定义颜色映射
```bash
python visualization.py \
  --gt gt.png --pred pred.png --output out.png \
  --colormap custom \
  --colormap-file custom_colormap.json
```

### 3. 支持自定义 ignore_index
```bash
python visualization.py \
  --gt gt.png --pred pred.png --output out.png \
  --colormap loveda \
  --ignore-index 255
```

## 向后兼容性

**注意**：旧版本的数据集特定函数（`visualize_loveda` 等）已被移除，请使用新的统一接口。

迁移示例：
```python
# 旧代码
visualize_loveda("gt.png", "pred.png", "out.png", is_gt_original=True)

# 新代码
from visualization import visualize
from colormaps import get_colormap

colormap = get_colormap("loveda")  # 或 "loveda_pred"
visualize("gt.png", "pred.png", "out.png", colormap=colormap)
```

## 优势总结

| 方面 | 旧方式 | 新方式 |
|------|--------|--------|
| **代码复用** | ❌ 每个数据集一个函数 | ✅ 统一接口 |
| **扩展性** | ❌ 需修改多处 | ✅ 只需添加颜色映射 |
| **维护性** | ❌ 颜色映射分散 | ✅ 集中管理 |
| **灵活性** | ❌ 预定义数据集有限 | ✅ 支持 JSON 自定义 |
| **用户使用** | ❌ 需记忆特定函数 | ✅ 统一 API |

## 快速参考

### 获取预定义颜色映射
```python
from colormaps import get_colormap, list_colormaps

# 列出所有可用映射
print(list_colormaps())  # ['loveda', 'loveda_pred', ...]

# 获取特定映射
colormap = get_colormap("loveda")
```

### 从 JSON 加载
```python
from colormaps import load_colormap_from_json

colormap = load_colormap_from_json("custom_colormap.json")
```

### 可视化
```python
from visualization import visualize, batch_visualize

# 单张
visualize("gt.png", "pred.png", "out.png", colormap=colormap)

# 批量
batch_visualize("gt_dir/", "pred_dir/", "out_dir/", colormap=colormap)
```

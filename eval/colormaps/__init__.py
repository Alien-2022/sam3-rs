"""
颜色映射表配置模块

用于定义和存储各种语义分割数据集的颜色映射表
"""

from pathlib import Path
from typing import Dict, Optional, List
import json


# LoveDA 数据集颜色映射
# GT 原始标签: 0: no-data, 1: background, 2: building, 3: road, 4: water, 5: barren, 6: forest, 7: agriculture
LOVEDA = {
    0: {"name": "no-data", "color": [0, 0, 0]},
    1: {"name": "background", "color": [255, 255, 255]},
    2: {"name": "building", "color": [128, 0, 0]},
    3: {"name": "road", "color": [128, 128, 128]},
    4: {"name": "water", "color": [0, 0, 128]},
    5: {"name": "barren", "color": [128, 128, 0]},
    6: {"name": "forest", "color": [0, 128, 0]},
    7: {"name": "agriculture", "color": [0, 128, 128]},
}


# OpenEarthMap 数据集颜色映射
# GT 原始标签: 0: no-data, 1: Bareland, 2: Rangeland, 3: Developed space, 4: Road, 5: Tree, 6: Water, 7: Agriculture land, 8: Building
OPENEARTHMAP = {
    0: {"name": "no-data", "color": [0, 0, 0]},
    1: {"name": "Bareland", "color": [128, 0, 0]},          # #800000
    2: {"name": "Rangeland", "color": [0, 255, 36]},       # #00FF24
    3: {"name": "Developed space", "color": [148, 148, 148]},  # #949494
    4: {"name": "Road", "color": [255, 255, 255]},         # #FFFFFF
    5: {"name": "Tree", "color": [34, 97, 38]},            # #226126
    6: {"name": "Water", "color": [0, 69, 255]},           # #0045FF
    7: {"name": "Agriculture land", "color": [75, 181, 73]},    # #4BB549
    8: {"name": "Building", "color": [222, 31, 7]},        # #DE1F07
}


# Potsdam 数据集颜色映射
# GT 原始标签: 0: background, 1: impervious surface, 2: building, 3: low vegetation, 4: tree, 5: car
# Potsdam 数据集颜色映射（官方颜色定义）
# GT 原始标签: 0: background, 1: impervious surface, 2: building, 3: low vegetation, 4: tree, 5: car
POTSDAM = {
    0: {"name": "background", "color": [255, 0, 0]},           # #FF0000 (Clutter/background)
    1: {"name": "impervious surface", "color": [255, 255, 255]},  # #FFFFFF
    2: {"name": "building", "color": [0, 0, 255]},           # #0000FF
    3: {"name": "low vegetation", "color": [0, 255, 255]},       # #00FFFF
    4: {"name": "tree", "color": [0, 255, 0]},              # #00FF00
    5: {"name": "car", "color": [255, 255, 0]},            # #FFFF00
}


# Vaihingen 数据集颜色映射（与 Potsdam 相同的官方颜色定义）
# GT 原始标签: 0: background, 1: impervious surface, 2: building, 3: low vegetation, 4: tree, 5: car
VAIHINGEN = {
    0: {"name": "background", "color": [255, 0, 0]},           # #FF0000 (Clutter/background)
    1: {"name": "impervious surface", "color": [255, 255, 255]},  # #FFFFFF
    2: {"name": "building", "color": [0, 0, 255]},           # #0000FF
    3: {"name": "low vegetation", "color": [0, 255, 255]},       # #00FFFF
    4: {"name": "tree", "color": [0, 255, 0]},              # #00FF00
    5: {"name": "car", "color": [255, 255, 0]},            # #FFFF00
}


# UAVid 数据集颜色映射（官方颜色定义）
# GT 原始标签: 0: background, 1: building, 2: road, 3: tree, 4: low vegetation, 5: moving car, 6: static car, 7: human
UAVID = {
    0: {"name": "background", "color": [0, 0, 0]},           # #000000
    1: {"name": "building", "color": [128, 0, 0]},           # #800000
    2: {"name": "road", "color": [128, 64, 128]},            # #804080
    3: {"name": "tree", "color": [0, 128, 0]},               # #008000
    4: {"name": "low vegetation", "color": [128, 128, 0]},       # #808000
    5: {"name": "moving car", "color": [64, 0, 128]},           # #400080
    6: {"name": "static car", "color": [192, 0, 192]},          # #C000C0
    7: {"name": "human", "color": [64, 64, 0]},             # #404000
}


# UDD5 数据集颜色映射（使用 New Label 的官方 RGB 颜色）
# GT 重映射标签 (New Label): 0: other, 1: vegetation, 2: building, 3: road, 4: vehicle
# 原始 GT Labels -> New Labels:
#   0(Vegetation) -> 1
#   1(Building)   -> 2
#   2(Road)       -> 3
#   3(Vehicle)    -> 4
#   4(Other)      -> 0 (background)
UDD5 = {
    0: {"name": "other", "color": [0, 0, 0]},               # #000000 (原始 4)
    1: {"name": "vegetation", "color": [107, 142, 35]},       # #6B8E23 (原始 0)
    2: {"name": "building", "color": [102, 102, 156]},          # #66669C (原始 1)
    3: {"name": "road", "color": [128, 64, 128]},             # #804080 (原始 2)
    4: {"name": "vehicle", "color": [0, 0, 142]},             # #00008E (原始 3)
}


# 注册表：支持的数据集颜色映射
COLORMAPS = {
    "loveda": LOVEDA,
    "openearthmap": OPENEARTHMAP,
    "potsdam": POTSDAM,
    "vaihingen": VAIHINGEN,
    "uavid": UAVID,
    "udd5": UDD5,
}


def get_colormap(name: str) -> Dict[int, Dict[str, any]]:
    """
    获取颜色映射表

    Args:
        name: 颜色映射表名称

    Returns:
        颜色映射字典

    Raises:
        ValueError: 如果名称不存在
    """
    if name not in COLORMAPS:
        available = list(COLORMAPS.keys())
        raise ValueError(
            f"颜色映射 '{name}' 不存在。"
            f"可用映射: {available}"
        )
    return COLORMAPS[name]


def load_colormap_from_json(json_path: str) -> Dict[int, Dict[str, any]]:
    """
    从 JSON 文件加载颜色映射表

    Args:
        json_path: JSON 配置文件路径

    Returns:
        颜色映射字典 {class_id: {"name": "xxx", "color": [R, G, B]}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容两种格式：
    # 1. 直接格式: {0: {"name": "xxx", "color": [R, G, B]}}
    # 2. 嵌套格式: {"class_info": {...}}
    if "class_info" in data:
        return data["class_info"]
    else:
        return data


def list_colormaps() -> List[str]:
    """列出所有可用的颜色映射表名称"""
    return list(COLORMAPS.keys())

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


# 注册表：支持的数据集颜色映射
COLORMAPS = {
    "loveda": LOVEDA,
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

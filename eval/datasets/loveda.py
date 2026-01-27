"""LoveDA dataset loader for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_class_names(cls_file: Optional[str], fallback: Optional[List[str]] = None) -> List[str]:
    if cls_file is None:
        if fallback is None:
            raise ValueError("Either cls_file or fallback class names must be provided.")
        return fallback
    names: List[str] = []
    with open(cls_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"No class names found in {cls_file}")
    return names


def _collect_files(root: Path, specs: Union[str, Path]) -> List[Path]:
    """Collect image paths when specs are directories (no glob patterns)."""
    # Simplify to single directory path
    base = Path(specs)
    if not base.is_absolute():
        base = root / base
    if not base.is_dir():
        raise FileNotFoundError(f"Expected directory but got: {base}")
    files = sorted([p for p in base.iterdir() if p.suffix.lower() in IMG_EXTS])
    return files


def _index_by_stem(paths: Iterable[Path]):
    """
    :param paths: 
        [
            WindowsPath('D:/data/LoveDA/Val/Rural/images_png/2617.png'),
            WindowsPath('D:/data/LoveDA/Val/Rural/images_png/2618.png')
        ]
    :return:
        {
            '2617': WindowsPath('D:/data/LoveDA/Val/Rural/images_png/2617.png'),
            '2618': WindowsPath('D:/data/LoveDA/Val/Rural/images_png/2618.png')
        }
    """
    return {p.stem: p for p in paths}


class LoveDADataset(Dataset):
    def __init__(
        self,
        data_root: str,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        cls_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        reduce_zero_label: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

        img_paths = _collect_files(self.data_root, img_dir)
        mask_paths = _collect_files(self.data_root, mask_dir)

        if len(img_paths) == 0 or len(mask_paths) == 0:
            raise FileNotFoundError(
                f"Image or mask files not found. data_root={self.data_root}, img_dir={img_dir}, mask_dir={mask_dir}"
            )

        img_map = _index_by_stem(img_paths)
        mask_map = _index_by_stem(mask_paths)

        common_keys = sorted(set(img_map.keys()) & set(mask_map.keys()))
        if not common_keys:
            raise ValueError("No overlapping image/mask pairs found.")

        self.pairs = [(img_map[k], mask_map[k]) for k in common_keys]
        # fallback class names if cls_file is not provided
        # In the constructor of LoveDADataset, users can pass in a text file path through cls_file 
        # or a string list directly through class_names.
        self.classes = _load_class_names(cls_file, fallback=class_names)
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        mask = np.array(Image.open(mask_path), dtype=np.int64)
        if self.reduce_zero_label:
            mask = mask - 1
            mask[mask == -1] = self.ignore_index

        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image_path": str(img_path),
            "mask": mask_tensor,
        }

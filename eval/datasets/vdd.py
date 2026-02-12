"""VDD dataset loader for evaluation."""

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
    base = Path(specs)
    if not base.is_absolute():
        base = root / base
    if not base.is_dir():
        raise FileNotFoundError(f"Expected directory but got: {base}")
    files = sorted([p for p in base.iterdir() if p.suffix.lower() in IMG_EXTS])
    return files


def _index_by_stem(paths: Iterable[Path]):
    """
    Index files by their stem (filename without extension).

    Args:
        paths: Iterable of file paths

    Returns:
        Dictionary mapping stem to file path
    """
    return {p.stem: p for p in paths}


class VDDDataset(Dataset):
    """VDD dataset loader.

    The VDD dataset contains 7 semantic classes:
        0: other (clutter/background)
        1: wall
        2: road
        3: vegetation
        4: vehicle
        5: roof
        6: water

    GT Labels (0-6):
        0 = other (background/clutter)
        1 = wall
        2 = road
        3 = vegetation
        4 = vehicle
        5 = roof
        6 = water

    Since cls_vdd.txt includes background (other) as first line (label 0),
    and GT masks use labels 0-6 directly,
    no label shifting is needed (reduce_zero_label=False).
    """

    def __init__(
        self,
        data_root: str,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        cls_file: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        reduce_zero_label: bool = False,
    ) -> None:
        """
        VDD Dataset for semantic segmentation evaluation.

        Args:
            data_root: Root directory of dataset
            img_dir: Directory containing images (relative to data_root)
            mask_dir: Directory containing masks (relative to data_root)
            cls_file: Path to text file containing class names
            class_names: Fallback list of class names if cls_file is not provided
            ignore_index: Value to use for ignored pixels (default: 255)
            reduce_zero_label: Whether to subtract 1 from labels to map class 0 to -1
                             VDD uses labels 0-6 (0=other, 1-6=classes),
                             set to True to map to -1/0-5 and ignore class 0 in metrics
        """
        self.data_root = Path(data_root)
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

        img_paths = _collect_files(self.data_root, img_dir)
        mask_paths = _collect_files(self.data_root, mask_dir)

        if len(img_paths) == 0 or len(mask_paths) == 0:
            raise FileNotFoundError(
                f"Image or mask files not found. data_root={self.data_root}, "
                f"img_dir={img_dir}, mask_dir={mask_dir}"
            )

        img_map = _index_by_stem(img_paths)
        mask_map = _index_by_stem(mask_paths)

        common_keys = sorted(set(img_map.keys()) & set(mask_map.keys()))
        if not common_keys:
            raise ValueError("No overlapping image/mask pairs found.")

        self.pairs = [(img_map[k], mask_map[k]) for k in common_keys]
        self.classes = _load_class_names(cls_file, fallback=class_names)
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # VDD masks are stored as PNG format with dtype uint8
        # Labels are 0-6 (0=other, 1=wall, 2=road, ..., 6=water)
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if self.reduce_zero_label:
            # Map labels 0-6 to -1/0-5:
            #   - other(0) -> -1 (will be marked as ignore_index)
            #   - wall(1) -> 0
            #   - ...
            #   - water(6) -> 5
            mask = mask - 1
            mask[mask == -1] = self.ignore_index

        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image_path": str(img_path),
            "mask": mask_tensor,
        }

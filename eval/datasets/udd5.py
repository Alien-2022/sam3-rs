"""UDD5 dataset loader for evaluation."""

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

    :param paths:
        [
            WindowsPath('D:/data/UDD5/Exp/img/image_001.png'),
            WindowsPath('D:/data/UDD5/Exp/img/image_002.png')
        ]
    :return:
        {
            'image_001': WindowsPath('D:/data/UDD5/Exp/img/image_001.png'),
            'image_002': WindowsPath('D:/data/UDD5/Exp/img/image_002.png')
        }
    """
    return {p.stem: p for p in paths}


class UDD5Dataset(Dataset):
    """UDD5 dataset loader.

    The UDD5 dataset contains 4 semantic classes after remapping:
        0: other (clutter/background) - remapped from original 4
        1: vegetation - remapped from original 0
        2: building - remapped from original 1
        3: road - remapped from original 2
        4: vehicle - remapped from original 3

    Original GT Labels (0-4):
        0 = Vegetation  -> New Label 1
        1 = Building    -> New Label 2
        2 = Road        -> New Label 3
        3 = Vehicle     -> New Label 4
        4 = Other       -> New Label 0 (background)

    Since cls_udd5.txt includes background as first line (mapped label 0),
    and the GT masks have been pre-remapped to New Labels,
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

        mask = np.array(Image.open(mask_path), dtype=np.int64)

        # UDD5 masks have been pre-remapped to New Labels (0-4):
        #   Original 0(Vegetation) -> 1
        #   Original 1(Building)   -> 2
        #   Original 2(Road)        -> 3
        #   Original 3(Vehicle)     -> 4
        #   Original 4(Other)       -> 0 (background)
        # Since cls_udd5.txt includes background (other) as first line (label 0),
        # no shifting is needed unless reduce_zero_label=True for compatibility
        if self.reduce_zero_label:
            mask = mask - 1
            mask[mask == -1] = self.ignore_index

        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image_path": str(img_path),
            "mask": mask_tensor,
        }

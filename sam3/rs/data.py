"""
Remote sensing dataset loader with flexible dataset support.

Supports common RS datasets:
- OpenEarthMap
- LoveDA
- iSAID
- Potsdam/Vaihingen
- UAVid
- WHU (building extraction)
- etc.

Designed to be:
1. Easy to add new datasets
2. Consistent API across datasets
3. Compatible with SAM3-RS inference
"""

import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    root_dir: str
    split: str = "val"  # train/val/test
    num_classes: int = 9

    # Dataset-specific options
    image_suffix: str = ".tif"
    mask_suffix: str = ".tif"
    reduce_zero_label: bool = False


class RSDataLoader:
    """
    Flexible remote sensing dataset loader.

    Features:
    - Automatic directory scanning
    - Support for multiple dataset formats
    - Easy to extend with new datasets
    - Returns (image_path, mask_path) pairs
    """

    # Predefined dataset configurations
    DATASET_CONFIGS = {
        'openearthmap': DatasetConfig(
            name='OpenEarthMap',
            root_dir='data/OpenEarthMap',
            split='val',
            num_classes=9,
            image_suffix='.tif',
            mask_suffix='.tif'
        ),
        'loveda': DatasetConfig(
            name='LoveDA',
            root_dir='data/LoveDA',
            split='val',
            num_classes=7,
            image_suffix='.png',
            mask_suffix='.png'
        ),
        'isaid': DatasetConfig(
            name='iSAID',
            root_dir='data/iSAID',
            split='val',
            num_classes=16,
            image_suffix='.png',
            mask_suffix='.png'
        ),
        'potsdam': DatasetConfig(
            name='Potsdam',
            root_dir='data/Potsdam',
            split='val',
            num_classes=6,
            image_suffix='.png',
            mask_suffix='.png'
        ),
        'vaihingen': DatasetConfig(
            name='Vaihingen',
            root_dir='data/Vaihingen',
            split='val',
            num_classes=6,
            image_suffix='.png',
            mask_suffix='.png'
        ),
        'uavid': DatasetConfig(
            name='UAVid',
            root_dir='data/UAVid',
            split='val',
            num_classes=8,
            image_suffix='.png',
            mask_suffix='.png'
        ),
        'whu': DatasetConfig(
            name='WHU',
            root_dir='data/WHU',
            split='test',
            num_classes=2,
            image_suffix='.png',
            mask_suffix='.png'
        ),
    }

    def __init__(self, config: DatasetConfig):
        """
        Args:
            config: DatasetConfig object
        """
        self.config = config
        self.samples = self._scan_dataset()

        print(f"✓ Loaded {config.name}: {len(self.samples)} samples")

    def _scan_dataset(self) -> List[Tuple[str, Optional[str]]]:
        """
        Scan dataset directory to find image-mask pairs.

        Returns:
            List of (image_path, mask_path) tuples
        """
        root = self.config.root_dir
        img_suffix = self.config.image_suffix
        mask_suffix = self.config.mask_suffix

        # Standard structure: root/split/images/ and root/split/masks/
        img_dir = os.path.join(root, self.config.split, "images")
        mask_dir = os.path.join(root, self.config.split, "masks")

        if not os.path.exists(img_dir):
            print(f"Warning: Image directory not found: {img_dir}")
            return []

        samples = []
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(img_suffix)])

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            img_name = os.path.splitext(img_file)[0]

            # Try to find corresponding mask
            mask_name = img_name + mask_suffix
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                samples.append((img_path, mask_path))
            else:
                # Some datasets might not have masks (inference-only mode)
                samples.append((img_path, None))

        return samples

    def get_samples(self, subset: Optional[int] = None) -> List[str]:
        """
        Get image paths for inference.

        Args:
            subset: Number of samples to return (None = all)
                    Useful for quick testing with small subset

        Returns:
            List of image paths
        """
        samples = [img_path for img_path, _ in self.samples]

        if subset is not None and subset < len(samples):
            return samples[:subset]

        return samples

    def get_sample(self, idx: int) -> Tuple[str, Optional[str], Optional[np.ndarray]]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            (image_path, mask_path, mask_array)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range (total: {len(self.samples)})")

        img_path, mask_path = self.samples[idx]

        # Load mask if exists
        mask_array = None
        if mask_path and os.path.exists(mask_path):
            mask_array = np.array(Image.open(mask_path))

        return img_path, mask_path, mask_array

    def add_dataset_config(self, name: str, config: DatasetConfig):
        """
        Add a new dataset configuration dynamically.

        Useful for custom datasets not in predefined configs.

        Example:
            loader.add_dataset_config('my_dataset', DatasetConfig(
                name='MyCustomDataset',
                root_dir='data/my_dataset',
                split='val',
                num_classes=10,
                image_suffix='.jpg',
                mask_suffix='.png'
            ))
        """
        self.DATASET_CONFIGS[name.lower()] = config
        print(f"✓ Added custom dataset: {name}")


def get_dataset_loader(dataset_name: str, root_dir: Optional[str] = None) -> RSDataLoader:
    """
    Factory function to create dataset loader.

    Args:
        dataset_name: Name of dataset (e.g., 'openearthmap', 'loveda')
        root_dir: Optional override for root directory

    Returns:
        RSDataLoader instance

    Example:
        loader = get_dataset_loader('openearthmap')
        samples = loader.get_samples()

        # With custom root
        loader = get_dataset_loader('loveda', root_dir='/path/to/loveda')
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in RSDataLoader.DATASET_CONFIGS:
        available = ', '.join(RSDataLoader.DATASET_CONFIGS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available: {available}"
        )

    config = RSDataLoader.DATASET_CONFIGS[dataset_name]

    # Override root if provided
    if root_dir is not None:
        config.root_dir = root_dir

    return RSDataLoader(config)


# ============ Convenience Functions ============

def scan_images(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Scan directory for images with given extensions.

    Args:
        directory: Path to scan
        extensions: List of extensions (default: common image formats)

    Returns:
        Sorted list of image paths
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

    images = []
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return images

    for f in os.listdir(directory):
        if any(f.lower().endswith(ext) for ext in extensions):
            images.append(os.path.join(directory, f))

    return sorted(images)


def split_dataset(samples: List[str], train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    Split samples into train/val sets.

    Args:
        samples: List of image paths
        train_ratio: Fraction for training set

    Returns:
        (train_samples, val_samples)
    """
    import random
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    return samples[:split_idx], samples[split_idx:]

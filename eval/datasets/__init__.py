"""Dataset loaders for SAM3-RS evaluation.

Usage:
    # Recommended: use registry for dynamic dataset loading
    from eval.datasets import DATASET_REGISTRY
    dataset_class = DATASET_REGISTRY["loveda"]
    dataset = dataset_class(...)

    # Alternative: import specific dataset for testing/debugging
    from eval.datasets.loveda import LoveDADataset
"""

from .loveda import LoveDADataset
from .openearthmap import OpenEarthMapDataset
from .isaid import iSAIDDataset

# Dataset registry: maps dataset names to dataset classes
DATASET_REGISTRY = {
    "loveda": LoveDADataset,
    "openearthmap": OpenEarthMapDataset,
    "isaid": iSAIDDataset,
}

# Export registry for dynamic dataset loading
# Specific dataset classes can be imported directly for testing:
#   from eval.datasets.loveda import LoveDADataset
__all__ = ["DATASET_REGISTRY"]

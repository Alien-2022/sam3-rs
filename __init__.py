"""
SAM3-RS: Lightweight SAM3 wrapper for remote sensing zero-shot semantic segmentation.

Main exports:
    - SAM3RSSegmentor: Main segmentation model
    - ExperimentConfig: Configuration dataclass
    - RemoteSensingDataset: Dataset loader
    - PromptManager: Prompt management
    - SegmentationMetrics: Evaluation metrics
"""

from segmentor import SAM3RSSegmentor
from config import (
    ExperimentConfig,
    ModelConfig,
    InferenceConfig,
    PromptConfig,
    DatasetConfig,
    OutputConfig,
    create_preset_openearthmap,
    create_preset_loveda,
    create_preset_whu,
)
from data import RemoteSensingDataset
from prompts import PromptManager
from metrics import SegmentationMetrics
from utils import visualize_prediction, save_mask

__version__ = "0.1.0"
__all__ = [
    # Models
    "SAM3RSSegmentor",
    
    # Configs
    "ExperimentConfig",
    "ModelConfig",
    "InferenceConfig",
    "PromptConfig",
    "DatasetConfig",
    "OutputConfig",
    "create_preset_openearthmap",
    "create_preset_loveda",
    "create_preset_whu",
    
    # Data
    "RemoteSensingDataset",
    
    # Prompts
    "PromptManager",
    
    # Metrics
    "SegmentationMetrics",
    
    # Utils
    "visualize_prediction",
    "save_mask",
]

__doc__ = """
SAM3-RS: Lightweight SAM3 wrapper for remote sensing zero-shot semantic segmentation.

Quick Start:
    >>> from sam3_rs import SAM3RSSegmentor, ExperimentConfig
    >>> config = ExperimentConfig(
    ...     checkpoint_path='weights/sam3.pt',
    ...     bpe_path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    ... )
    >>> model = SAM3RSSegmentor(config)
    >>> result = model.predict_single('image.png')

Documentation:
    - Quick Start: See docs/START_HERE.md
    - Architecture: See docs/ARCHITECTURE.md
"""

"""
SAM3-RS: Lightweight SAM3 wrapper for remote sensing zero-shot semantic segmentation.

Main exports:
    - SAM3RSSegmentor: Main segmentation model
    - InferenceConfig: Inference configuration dataclass
    - SegmentationResult: Result container
"""

from segmentor import SAM3RSSegmentor, InferenceConfig, SegmentationResult

__version__ = "0.1.0"
__all__ = [
    "SAM3RSSegmentor",
    "InferenceConfig",
    "SegmentationResult",
]

__doc__ = """
SAM3-RS: Lightweight SAM3 wrapper for remote sensing zero-shot semantic segmentation.

Quick Start:
    >>> from segmentor import SAM3RSSegmentor, InferenceConfig
    >>> config = InferenceConfig(
    ...     checkpoint_path='weights/sam3.pt',
    ...     bpe_path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    ...     prompts_file='configs/prompts_example.txt',
    ... )
    >>> model = SAM3RSSegmentor(config)
    >>> result = model.predict_single('image.png')

Documentation:
    - README.md: Project overview and usage
    - segmentor_lib/: Modular inference components
    - eval/: Evaluation scripts and datasets
"""

"""
Configuration management for SAM3-RS experiments.

Centralized configuration for easy experiment management.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import json


@dataclass
class ModelConfig:
    """SAM3 model configuration"""
    checkpoint_path: str = "weights/sam3/sam3.pt"
    bpe_path: str = "assets/bpe_simple_vocab_16e6.txt.gz"

    # Device settings
    device: str = "cuda"
    precision: str = "bfloat16"  # or "float32"


@dataclass
class InferenceConfig:
    """Inference configuration (SegEarthOV3 style)"""
    # Confidence thresholds
    confidence_threshold: float = 0.5
    prob_threshold: float = 0.0
    bg_idx: int = 0

    # Head selection (SegEarthOV3's dual-head fusion)
    use_semantic_head: bool = True
    use_instance_head: bool = True
    use_presence_score: bool = True

    # Large image handling
    slide_crop_size: int = 0  # 0 = no sliding window
    slide_stride: int = 512

    # Batch settings
    batch_size: int = 1


@dataclass
class PromptConfig:
    """Prompt configuration"""
    prompts_file: str = "configs/prompts.txt"
    use_synonyms: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "openearthmap"
    root_dir: str = "data/OpenEarthMap"
    split: str = "val"
    subset: Optional[int] = None  # None = all samples

    # Dataset type
    is_segmentation: bool = True  # True = seg, False = binary (e.g., building extraction)


@dataclass
class OutputConfig:
    """Output configuration"""
    save_dir: str = "outputs"
    save_predictions: bool = True
    save_visualizations: bool = True
    save_metrics: bool = True
    save_per_class_masks: bool = False

    # Visualization settings
    vis_overlay_alpha: float = 0.5
    show_plots: bool = False


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Usage:
        config = ExperimentConfig(
            dataset=DatasetConfig(name='loveda', root_dir='data/LoveDA'),
            inference=InferenceConfig(slide_crop_size=1024, slide_stride=512),
            output=OutputConfig(save_dir='outputs/loveda_1024crop')
        )

        # Save for reproducibility
        config.save('configs/experiment_loveda.json')

        # Run experiment
        results = run_experiment(config)
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Experiment metadata
    experiment_name: str = "sam3_rs_experiment"
    description: str = ""
    seed: int = 42

    def save(self, output_path: str):
        """
        Save configuration to JSON file.

        Args:
            output_path: Path to save config

        Example:
            config.save('configs/experiment_001.json')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        print(f"✓ Saved experiment config to {output_path}")

    @classmethod
    def load(cls, config_path: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to config file

        Returns:
            ExperimentConfig instance

        Example:
            config = ExperimentConfig.load('configs/experiment_001.json')
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            data = json.load(f)

        # Reconstruct from dict
        return cls(
            model=ModelConfig(**data.get('model', {})),
            inference=InferenceConfig(**data.get('inference', {})),
            prompts=PromptConfig(**data.get('prompts', {})),
            dataset=DatasetConfig(**data.get('dataset', {})),
            output=OutputConfig(**data.get('output', {})),
            experiment_name=data.get('experiment_name', 'sam3_rs_experiment'),
            description=data.get('description', ''),
            seed=data.get('seed', 42)
        )

    def print_summary(self):
        """Print configuration summary"""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset.name}")
        print(f"  Root: {self.dataset.root_dir}")
        print(f"  Split: {self.dataset.split}")
        print(f"  Subset: {self.dataset.subset if self.dataset.subset else 'all'}")
        print(f"\nModel:")
        print(f"  Checkpoint: {self.model.checkpoint_path}")
        print(f"  Device: {self.model.device}")
        print(f"\nInference:")
        print(f"  Slide crop: {self.inference.slide_crop_size}")
        print(f"  Slide stride: {self.inference.slide_stride}")
        print(f"  Prob threshold: {self.inference.prob_threshold}")
        print(f"  Use semantic head: {self.inference.use_semantic_head}")
        print(f"  Use instance head: {self.inference.use_instance_head}")
        print(f"  Use presence score: {self.inference.use_presence_score}")
        print(f"\nOutput:")
        print(f"  Save dir: {self.output.save_dir}")
        print(f"{'='*60}\n")


# ============ Preset Configurations ============

def create_preset_openearthmap(save_dir: str = None) -> ExperimentConfig:
    """Preset for OpenEarthMap evaluation."""
    return ExperimentConfig(
        experiment_name="sam3_rs_openearthmap",
        description="SAM3-RS on OpenEarthMap dataset",
        dataset=DatasetConfig(
            name="openearthmap",
            root_dir="data/OpenEarthMap",
            split="val"
        ),
        inference=InferenceConfig(
            confidence_threshold=0.1,
            prob_threshold=0.0,
            slide_crop_size=0  # OpenEarthMap images are small
        ),
        output=OutputConfig(
            save_dir=save_dir or "outputs/openearthmap"
        )
    )


def create_preset_loveda(save_dir: str = None) -> ExperimentConfig:
    """Preset for LoveDA evaluation."""
    return ExperimentConfig(
        experiment_name="sam3_rs_loveda",
        description="SAM3-RS on LoveDA dataset",
        dataset=DatasetConfig(
            name="loveda",
            root_dir="data/LoveDA",
            split="val"
        ),
        inference=InferenceConfig(
            confidence_threshold=0.1,
            prob_threshold=0.0,
            slide_crop_size=1024,  # LoveDA has larger images
            slide_stride=512
        ),
        output=OutputConfig(
            save_dir=save_dir or "outputs/loveda"
        )
    )


def create_preset_whu(save_dir: str = None) -> ExperimentConfig:
    """Preset for WHU building extraction."""
    return ExperimentConfig(
        experiment_name="sam3_rs_whu",
        description="SAM3-RS on WHU building extraction",
        dataset=DatasetConfig(
            name="whu",
            root_dir="data/WHU",
            split="test",
            is_segmentation=False  # Binary task
        ),
        inference=InferenceConfig(
            confidence_threshold=0.5,
            prob_threshold=0.5,
            slide_crop_size=512,
            slide_stride=256
        ),
        prompts=PromptConfig(
            prompts_file="configs/prompts_binary.txt"  # Just 'building'
        ),
        output=OutputConfig(
            save_dir=save_dir or "outputs/whu"
        )
    )

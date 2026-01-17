#!/usr/bin/env python3
"""
Quick start script for SAM3-RS.

This script helps you get started quickly with minimal setup.
"""

import os
import sys

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    missing = []

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("  ✗ PyTorch not found")

    try:
        import torchvision
        print(f"  ✓ TorchVision {torchvision.__version__}")
    except ImportError:
        missing.append("torchvision")
        print("  ✗ TorchVision not found")

    try:
        import PIL
        from PIL import Image
        print("  ✓ Pillow (PIL)")
    except ImportError:
        missing.append("Pillow")
        print("  ✗ Pillow not found")

    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
        print("  ✗ NumPy not found")

    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")
        print("  ✗ Matplotlib not found")

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("  ✓ All dependencies installed!\n")
    return True


def create_sample_configs():
    """Create sample configuration files."""
    print("Creating sample configuration files...")

    # Create configs directory
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)

    # Sample prompts file
    if not os.path.exists(os.path.join(configs_dir, "prompts.txt")):
        prompts_path = os.path.join(configs_dir, "prompts.txt")
        with open(prompts_path, 'w') as f:
            f.write("background\n")
            f.write("bareland,barren\n")
            f.write("grass\n")
            f.write("road,route\n")
            f.write("car,vehicle\n")
            f.write("tree,forest\n")
            f.write("water,river\n")
            f.write("cropland,farmland\n")
            f.write("building,roof,house\n")
        print(f"  ✓ Created {prompts_path}")

    # Sample experiment config (OpenEarthMap style)
    config_path = os.path.join(configs_dir, "experiment_openearthmap.json")
    if not os.path.exists(config_path):
        import json
        sample_config = {
            "experiment_name": "sam3_rs_openearthmap",
            "description": "SAM3-RS on OpenEarthMap dataset",
            "dataset": {
                "name": "openearthmap",
                "root_dir": "data/OpenEarthMap",
                "split": "val",
                "num_classes": 9
            },
            "model": {
                "checkpoint_path": "weights/sam3/sam3.pt",
                "bpe_path": "assets/bpe_simple_vocab_16e6.txt.gz",
                "device": "cuda"
            },
            "inference": {
                "confidence_threshold": 0.1,
                "prob_threshold": 0.0,
                "bg_idx": 0,
                "use_semantic_head": True,
                "use_instance_head": True,
                "use_presence_score": True,
                "slide_crop_size": 0,
                "slide_stride": 512
            },
            "prompts": {
                "prompts_file": "configs/prompts.txt"
            },
            "output": {
                "save_dir": "outputs/openearthmap",
                "save_predictions": True,
                "save_visualizations": True,
                "save_metrics": True,
                "vis_overlay_alpha": 0.5,
                "show_plots": False
            },
            "seed": 42
        }

        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"  ✓ Created {config_path}")

    # Sample experiment config (LoveDA style - with sliding window)
    config_path = os.path.join(configs_dir, "experiment_loveda.json")
    if not os.path.exists(config_path):
        sample_config = {
            "experiment_name": "sam3_rs_loveda",
            "description": "SAM3-RS on LoveDA dataset with sliding window",
            "dataset": {
                "name": "loveda",
                "root_dir": "data/LoveDA",
                "split": "val",
                "num_classes": 7
            },
            "model": {
                "checkpoint_path": "weights/sam3/sam3.pt",
                "bpe_path": "assets/bpe_simple_vocab_16e6.txt.gz",
                "device": "cuda"
            },
            "inference": {
                "confidence_threshold": 0.1,
                "prob_threshold": 0.0,
                "bg_idx": 0,
                "use_semantic_head": True,
                "use_instance_head": True,
                "use_presence_score": True,
                "slide_crop_size": 1024,  # Larger images
                "slide_stride": 512
            },
            "prompts": {
                "prompts_file": "configs/prompts.txt"
            },
            "output": {
                "save_dir": "outputs/loveda",
                "save_predictions": True,
                "save_visualizations": True,
                "save_metrics": True,
                "vis_overlay_alpha": 0.5,
                "show_plots": False
            },
            "seed": 42
        }

        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"  ✓ Created {config_path}")

    print()


def print_next_steps():
    """Print next steps for the user."""
    print("="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. Prepare your data:")
    print("   mkdir -p data/OpenEarthMap/val/images")
    print("   mkdir -p data/OpenEarthMap/val/masks")
    print("   # Copy your images and ground truth to these directories")
    print("\n2. Download SAM3 checkpoint:")
    print("   # Place 'sam3.pt' in weights/sam3/")
    print("   # Get it from: https://huggingface.co/facebook/sam3")
    print("\n3. Run experiments:")
    print("   # Using preset config:")
    print("   python -m sam3.rs.run_experiment --preset openearthmap --save_vis")
    print("\n   # Using custom config:")
    print("   python -m sam3.rs.run_experiment --config configs/experiment_openearthmap.json")
    print("\n4. View results:")
    print("   ls outputs/openearthmap/")
    print("   cat outputs/openearthmap/metrics.json")
    print("\n5. Modify and experiment:")
    print("   # Edit sam3/rs/segmentor.py for new fusion strategies")
    print("   # Edit sam3/rs/prompts.py for prompt optimization")
    print("   # Edit sam3/rs/config.py to add presets")
    print("="*70)


def main():
    """Main entry point for quick start."""
    print("\n" + "="*70)
    print("SAM3-RS: Quick Start")
    print("="*70 + "\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create sample configs
    create_sample_configs()

    # Print next steps
    print_next_steps()

    print("\n✓ Quick start setup complete!")
    print("  You're ready to run SAM3-RS experiments.\n")


if __name__ == '__main__':
    main()

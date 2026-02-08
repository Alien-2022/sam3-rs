# SAM3-RS

Lightweight SAM3 wrapper for remote sensing zero-shot semantic segmentation.

## Features

✨ **Core capabilities**
- Dual-head fusion (instance + semantic) - SegEarthOV3 innovation
- Presence score filtering
- Sliding window inference for large images
- Multi-class prompt handling with synonym support
- Remote sensing dataset integration
- No heavy dependencies (no MMSegmentation required!)

🚀 **Easy to use**
- Simple Python API
- Configuration-driven experiments
- Extensible for custom datasets
- Built-in evaluation metrics

## Installation

```bash
# Basic dependencies
pip install torch torchvision pillow numpy matplotlib

# Install SAM3-RS
cd workspace/core/sam3-rs
pip install -e .
```

## Quick Start

### 1. Prepare prompts file

Create `configs/prompts.txt`:
```
background
bareland,barren
grass
road,route
car,vehicle
tree,forest
water,river
cropland,farmland
building,roof,house
```

### 2. Run inference with preset config

```bash
# OpenEarthMap (small images, no sliding window)
python run_experiment.py --preset openearthmap

# LoveDA (larger images, with sliding window)
python run_experiment.py --preset loveda --save_vis

# WHU building extraction (binary task)
python run_experiment.py --preset whu --conf_thd 0.5 --prob_thd 0.5
```

### 3. Custom configuration

```bash
# Custom dataset
python run_experiment.py \
    --dataset my_dataset \
    --data_root /path/to/data \
    --slide_crop 1024 \
    --slide_stride 512 \
    --prompts_file configs/my_prompts.txt \
    --save_dir outputs/custom_run

# Load from config file
python run_experiment.py \
    --config configs/experiment_001.json \
    --save_dir outputs/run_001
```

## Project Structure

```
sam3-rs/
├── segmentor.py          # Core SAM3-RS wrapper
├── data.py              # Dataset loader
├── prompts.py            # Prompt management
├── metrics.py            # Evaluation metrics
├── config.py             # Configuration management
├── utils.py              # Visualization utilities
├── run_experiment.py     # Main experiment runner
├── configs/              # Configuration files
│   └── prompts_example.txt
├── docs/                 # Documentation
│   ├── START_HERE.md     # Quick start guide
│   └── ARCHITECTURE.md   # Architecture details
├── sam3/                # Original SAM3 code
└── README.md            # This file
```

## Usage Examples

### Basic Inference

```python
from segmentor import SAM3RSSegmentor
from config import ExperimentConfig

# Initialize model
config = ExperimentConfig(
    checkpoint_path='weights/sam3.pt',
    bpe_path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    device='cuda',
)

model = SAM3RSSegmentor(config)

# Predict single image
result = model.predict_single('data/image.tif')

# Save result
result['seg_pred'].save('output_pred.png')
```

### Batch Inference

```python
from data import RemoteSensingDataset
from config import DatasetConfig

# Load dataset
dataset_config = DatasetConfig(
    name='OpenEarthMap',
    root_dir='data/OpenEarthMap',
    split='val'
)
dataset = RemoteSensingDataset(dataset_config)

# Predict batch
for img_path, mask_path in dataset:
    result = model.predict_single(img_path)
    # Process result...
```

### Sliding Window for Large Images

```python
from config import InferenceConfig

# Large image (e.g., 10000x10000)
config = ExperimentConfig(
    model=ModelConfig(
        checkpoint_path='weights/sam3.pt',
        bpe_path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    ),
    inference=InferenceConfig(
        slide_crop_size=512,    # Crop size
        slide_stride=256,        # Overlap
    )
)

model = SAM3RSSegmentor(config)

# Automatically uses sliding window
result = model.predict_single('large_image.tif')
```

## Comparison with SegEarthOV3

| Feature | SegEarthOV3 | SAM3-RS (this) |
|---------|---------------|-------------------|
| **Dependencies** | MMSeg + MMCV (heavy!) | Only PyTorch (light!) |
| **Dual-head fusion** | ✅ | ✅ |
| **Presence score** | ✅ | ✅ |
| **Sliding window** | ✅ | ✅ |
| **Multi-dataset support** | ✅ (hardcoded) | ✅ (easy to extend) |
| **Config-driven** | ✅ (complex) | ✅ (simple) |
| **Evaluation metrics** | Via MMSeg | Built-in (no MMSeg) |
| **Easy to modify** | ❌ (deep inheritance) | ✅ (clean code) |

## Documentation


## Design Principles

1. **No heavy dependencies**: Only requires PyTorch, PIL, NumPy
2. **Easy to extend**: Add new datasets, metrics, prompt strategies
3. **Clear APIs**: Simple functions, no complex inheritance
4. **Configuration-driven**: Change behavior via config files, not code
5. **Research-friendly**: Easy to experiment with different settings

## Roadmap

- [x] Core SAM3 wrapping
- [x] Sliding window inference
- [x] Dual-head fusion
- [x] Remote sensing datasets
- [ ] Prompt optimization (CLIP-based ranking)
- [ ] Distributed inference
- [ ] Automatic prompt generation
- [ ] Ensemble methods

## Citation


## License

Based on:
- SAM3 (Meta AI) - Apache 2.0
- SegEarthOV3 - Check original repository

## Contact

For issues, questions, or contributions, please open an issue on GitHub.

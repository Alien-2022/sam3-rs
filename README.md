# SAM3-RS

Lightweight SAM3 wrapper for remote sensing zero-shot semantic segmentation.

## Features

✨ **Core capabilities**
- Dual-head fusion (instance + semantic) — SegEarthOV3 innovation
- Presence score filtering
- Sliding window inference for large remote sensing images
- Multi-scale inference for improved cross-scale generalisation
- Multi-class prompt handling with synonym support
- No heavy dependencies (no MMSegmentation required!)

🚀 **Easy to use**
- Simple Python API
- YAML / JSON configuration files
- Extensible for custom datasets
- Built-in evaluation metrics

## Installation

```bash
# Core dependencies
pip install torch torchvision pillow numpy matplotlib tqdm pyyaml

# Install the SAM3 sub-package (editable)
cd sam3
pip install -e .
```

## Quick Start

### 1. Prepare prompts file

Create `configs/prompts.txt` (one class per line, synonyms comma-separated):
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

### 2. Run the demo

```bash
python demo.py \
    --image /path/to/image.tif \
    --checkpoint weights/sam3.pt \
    --prompts configs/prompts_example.txt \
    --colormap loveda \
    --output_dir outputs/demo
```

### 3. Python API

```python
from segmentor import SAM3RSSegmentor, InferenceConfig

config = InferenceConfig(
    checkpoint_path="weights/sam3.pt",
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    device="cuda",
    prompts_file="configs/prompts_example.txt",
)

model = SAM3RSSegmentor(config)
result = model.predict_single("image.tif")
pred_mask = result.seg_pred.cpu().numpy()   # [H, W] class IDs
```

### 4. Load config from YAML

```python
from segmentor_lib.config_loader import load_inference_config
from segmentor import SAM3RSSegmentor

config = load_inference_config("configs/loveda.yaml")
model = SAM3RSSegmentor(config)
```

### 5. Sliding window (large images)

```python
config = InferenceConfig(
    ...,
    slide_crop_size=1024,  # crop size in pixels
    slide_stride=512,       # stride (overlap = crop_size - stride)
)
```

### 6. Multi-scale inference

```python
config = InferenceConfig(
    ...,
    multi_scale_factors=[0.5, 1.0, 1.5],  # scale factors
    multi_scale_merge_mode="avg",           # "avg" or "max"
)
```

### 7. Fine-tune on your dataset

```bash
python train_rs.py \
    --data_root data/LoveDA/train \
    --val_root  data/LoveDA/val \
    --prompts_file configs/loveda_classes.txt \
    --checkpoint weights/sam3.pt \
    --output_dir outputs/finetune_loveda \
    --epochs 10 --batch_size 2 --lr 1e-4 \
    --freeze_backbone
```

## Project Structure

```
sam3-rs/
├── segmentor.py              # Main SAM3-RS wrapper + InferenceConfig
├── segmentorV1.py            # Legacy V1 wrapper (kept for reference)
├── demo.py                   # CLI demo with visualization
├── train_rs.py               # Remote sensing fine-tuning script
├── metrics.py                # Lightweight RSEvaluator (no MMSeg)
│
├── segmentor_lib/            # Modular inference components
│   ├── core.py               # InferenceEngine (single-image forward pass)
│   ├── prompts.py            # load_prompts(), precompute_text_features()
│   ├── postprocess.py        # fuse_prompts_to_classes(), logits_to_pred()
│   ├── sliding_window.py     # SlidingWindowInference
│   ├── multi_scale.py        # MultiScaleInference ← new
│   ├── config_loader.py      # load/save InferenceConfig from YAML/JSON ← new
│   ├── analyzers.py          # PresenceScoreAnalyzer, StatisticsAnalyzer
│   ├── debug.py              # MemoryDebugger
│   └── experimental/
│       ├── semantic_enhancement.py
│       └── adaptive_threshold.py
│
├── eval/                     # Evaluation scripts
│   ├── metrics/
│   │   └── seg_metrics.py    # Fast streaming SegmentationMetric
│   └── colormaps.py          # Dataset-specific colour palettes
│
├── configs/                  # Configuration files
│   ├── prompts_example.txt   # Example prompts (LoveDA)
│   └── loveda.yaml           # Example YAML config ← new
│
├── tests/                    # Unit tests ← new
│   └── test_core.py
│
├── sam3/                     # SAM3 upstream code
├── requirements.txt
└── README.md
```

## Comparison with SegEarthOV3

| Feature | SegEarthOV3 | SAM3-RS |
|---------|-------------|---------|
| **Dependencies** | MMSeg + MMCV (heavy) | PyTorch only (light) |
| **Dual-head fusion** | ✅ | ✅ |
| **Presence score** | ✅ | ✅ |
| **Sliding window** | ✅ | ✅ |
| **Multi-scale inference** | ❌ | ✅ |
| **YAML / JSON configs** | ❌ | ✅ |
| **Fine-tuning script** | ❌ | ✅ |
| **Unit tests** | ❌ | ✅ |
| **Easy to modify** | ❌ (deep inheritance) | ✅ (flat, modular) |

## Design Principles

1. **No heavy dependencies** – only PyTorch, PIL, NumPy, tqdm
2. **Modular** – each concern lives in its own file under `segmentor_lib/`
3. **Config-driven** – change behaviour via YAML/JSON, not code edits
4. **Research-friendly** – easy to add new datasets, metrics, prompt strategies

## Roadmap

- [x] Core SAM3 wrapping
- [x] Sliding window inference
- [x] Dual-head fusion
- [x] Multi-scale inference
- [x] YAML/JSON config support
- [x] Fine-tuning script
- [x] Unit tests
- [ ] CLIP-based prompt ranking
- [ ] Distributed multi-GPU inference
- [ ] Automatic prompt generation

## License

Based on:
- SAM3 (Meta AI) – Apache 2.0
- SegEarthOV3 – check the original repository

## Contact

For issues, questions, or contributions, please open an issue on GitHub.


# SAM3-RS Demo 使用说明

## 概述

`demo.py` 演示了 SAM3-RS 的两种主要使用方式：
1. **单提示词推理**（实例分割）- 检测单个类别
2. **多类别提示词推理**（语义分割）- 同时检测多个类别

## 主要改进

### 旧版 demo.py 的问题
```python
# 旧版：直接使用 SAM3 的原始 API
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(...)
processor = Sam3Processor(model)

# 手动管理推理状态
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="building")
```

### 新版 demo.py 的优势
```python
# 新版：使用封装好的 segmentor
from segmentor import SAM3RSSegmentor, InferenceConfig

# 配置驱动
config = InferenceConfig(
    checkpoint_path='weights/sam3/sam3.pt',
    bpe_path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',
    prompts_file='configs/prompts_example.txt',
    confidence_threshold=0.1,
    slide_crop_size=0,
)

# 初始化 segmentor
segmentor = SAM3RSSegmentor(config)

# 一行推理
result = segmentor.predict_single(image_path, detailed=True)
```

## 使用方式

### 1. 准备环境

```bash
cd workspace/core/sam3-rs

# 安装依赖
pip install -r requirements.txt

# 下载 SAM3 模型权重（如果没有）
# 将 sam3.pt 放到 weights/sam3/ 目录
```

### 2. 准备数据

```bash
# 准备测试图像
mkdir -p data/OpenEarthMap/Val/images

# 将你的测试图像放到该目录
cp /path/to/your/test.png data/OpenEarthMap/Val/images/
```

### 3. 运行 demo

```bash
# 直接运行（使用默认配置）
python demo.py
```

## Demo 功能说明

### Demo 1: 单提示词推理（实例分割）

```python
single_img_single_prompt(
    segmentor,
    test_image_path,
    prompt="building",  # 只检测建筑物
    output_path="outputs/demo/single_prompt_result/building_mask.png",
    mode="mask"
)
```

**输出**：
- `building_mask.png` - 建筑物的二值掩码
- `building_mask_semantic_seg.png` - 语义分割（如果可用）

**使用场景**：
- 只关心某个特定类别的检测
- 需要实例级别的分割（多个独立的建筑物）

### Demo 2: 多类别提示词推理（语义分割）

```python
single_img_multi_prompts(
    segmentor,
    test_image_path,
    prompts_file="configs/prompts_example.txt",  # 包含 8 个类别
    output_path="outputs/demo/multi_class_result/xxx_seg"
)
```

**输出**：
- `xxx_seg_pred.png` - 语义分割预测（每个像素一个类别 ID）
- `xxx_seg_overlay.png` - 原图 + 分割叠加
- `xxx_seg_class_masks/` - 每个类别的二值掩码
  - `background.png`
  - `bareland.png`
  - `road.png`
  - `car.png`
  - ...

**使用场景**：
- 需要完整的语义分割
- 每个像素都分配一个类别标签

### Demo 3: 大图像滑动窗口（可选）

```python
config = InferenceConfig(
    checkpoint_path=checkpoint_path,
    bpe_path=bpe_path,
    device='cuda',
    slide_crop_size=1024,  # 滑动窗口大小
    slide_stride=512,      # 滑动步长（有重叠）
    prompts_file=prompts_file
)

segmentor = SAM3RSSegmentor(config)
result = segmentor.predict_single(large_image_path)
```

**使用场景**：
- 处理超大图像（如 10000x10000）
- GPU 内存不足时

## 配置说明

### InferenceConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_path` | str | 必填 | SAM3 模型权重路径 |
| `bpe_path` | str | 必填 | BPE tokenizer 路径 |
| `device` | str | "cuda" | 设备：'cuda' 或 'cpu' |
| `confidence_threshold` | float | 0.5 | 实例检测置信度阈值 |
| `prob_threshold` | float | 0.0 | 像素概率阈值 |
| `bg_idx` | int | 0 | 背景类别 ID |
| `use_semantic_head` | bool | True | 使用语义分割头 |
| `use_instance_head` | bool | True | 使用实例分割头 |
| `use_presence_score` | bool | True | 使用存在性分数过滤 |
| `slide_crop_size` | int | 0 | 滑动窗口大小（0=不使用） |
| `slide_stride` | int | 512 | 滑动窗口步长 |
| `prompts_file` | str | None | 提示词配置文件路径 |

### 提示词文件格式

```
background
bareland,barren
road,path
car
tree,forest
water,river
agricultural land
building,roof,house
```

- 每行一个类别
- 逗号分隔同义词（如 "bareland,barren"）
- 顺序对应类别 ID（0, 1, 2, ...）

## 输出结构

```
outputs/demo/
├── single_prompt_result/
│   └── building_mask.png
│   └── building_mask_semantic_seg.png
└── multi_class_result/
    ├── image_pred.png              # 语义分割预测
    ├── image_overlay.png          # 叠加可视化
    └── image_class_masks/        # 每个类别的二值掩码
        ├── background.png
        ├── bareland.png
        ├── road.png
        ├── car.png
        ├── tree.png
        ├── water.png
        ├── agricultural land.png
        └── building.png
```

## 常见问题

### Q1: 如何修改要检测的类别？

**A**: 修改 `configs/prompts_example.txt` 文件：
```bash
# 只检测建筑物和道路
vim configs/prompts_example.txt
```

```
background
building,roof,house
road,path,highway
```

### Q2: 如何调整检测阈值？

**A**: 修改 `demo.py` 中的配置：
```python
config = InferenceConfig(
    confidence_threshold=0.2,  # 提高阈值减少误检
    prob_threshold=0.1,          # 提高阈值过滤低置信像素
    ...
)
```

### Q3: 如何处理非常大的图像？

**A**: 启用滑动窗口：
```python
config = InferenceConfig(
    slide_crop_size=1024,  # 每个窗口 1024x1024
    slide_stride=512,      # 重叠 512 像素
    ...
)
```

### Q4: 如何只使用语义分割头（不用实例头）？

**A**: 设置配置：
```python
config = InferenceConfig(
    use_instance_head=False,  # 只用语义头
    use_semantic_head=True,
    ...
)
```

### Q5: 结果不理想怎么办？

**A**: 尝试以下调优：
1. **优化提示词**：使用更具体的描述（如 "tall building" vs "building"）
2. **添加同义词**：如 "building,roof,house,structure"
3. **调整阈值**：
   - `confidence_threshold`：控制实例检测
   - `prob_threshold`：控制像素级置信度
4. **调整滑动窗口**：增大 crop_size 或减小 stride

## 对比：原始 SAM3 vs SAM3-RS

| 功能 | 原始 SAM3 | SAM3-RS（新版 demo） |
|------|-----------|---------------------|
| **单提示词推理** | ✅ 手动管理状态 | ✅ 一行调用 |
| **多类别推理** | ❌ 需要循环调用 | ✅ 自动批量处理 |
| **双头融合** | ❌ 不支持 | ✅ 自动融合 |
| **滑动窗口** | ❌ 需要手动实现 | ✅ 自动检测 |
| **同义词处理** | ❌ 不支持 | ✅ 自动合并 |
| **结果可视化** | ❌ 需要自己写 | ✅ 内置保存函数 |

## 下一步

1. **自定义数据集**：
   - 修改 `data.py` 添加新的数据集
   - 参考 `RemoteSensingDataset` 类

2. **批量推理**：
   - 使用 `segmentor.predict_batch(image_paths, save_dir)`

3. **评估指标**：
   - 使用 `metrics.SegmentationMetrics` 计算 mIoU、F1 等

4. **实验管理**：
   - 使用 `run_experiment.py` 进行完整的实验流程

## 相关文档

- **[README.md](README.md)** - 项目总览
- **[docs/START_HERE.md](docs/START_HERE.md)** - 快速开始
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - 架构详情
- **[segmentor.py](segmentor.py)** - 核心实现
- **[config.py](config.py)** - 配置管理

## 引用

如果你使用了 SAM3-RS，请引用：

```bibtex
@article{segearthov3,
  title={SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images},
  author={Li, Kaiyu and Zhang, Shengqi and Deng, Yupeng and Wang, Zhi and Meng, Deyu and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2512.08730},
  year={2025}
}
```

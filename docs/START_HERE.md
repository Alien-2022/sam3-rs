# SAM3-RS 快速开始指南

## ✅ 框架已创建完成！

所有核心文件已在 `sam3/rs/` 目录下创建完成。

---

## 📁 文件结构

```
sam3/rs/
├── __init__.py              # 主入口（统一导入）
├── segmentor.py             # 核心：SAM3 封装 + SegEarthOV3 特性
├── data.py                 # 数据集加载（多数据集支持）
├── prompts.py               # 提示词管理（同义词、映射）
├── metrics.py               # 评估指标（mIoU, F1, 无 MMSeg 依赖）
├── config.py                # 配置管理（预设、实验配置）
├── utils.py                # 工具函数（可视化、文件 I/O）
├── run_experiment.py        # 主运行脚本（完整工作流）
├── example_prompts.txt      # 示例提示词文件
├── ARCHITECTURE.md         # 框架架构文档
└── quick_start.py           # 快速开始脚本
```

---

## 🚀 5 分钟快速开始

### 1. 运行快速开始脚本

```bash
cd workspace/core/sam3-rs
python quick_start.py
```

这个脚本会：
- ✓ 检查依赖（PyTorch, NumPy, PIL, Matplotlib）
- ✓ 创建示例配置文件（`configs/prompts.txt`, `configs/experiment_*.json`）
- ✓ 打印下一步指引

### 2. 准备数据

创建数据目录结构（以 OpenEarthMap 为例）：

```bash
mkdir -p data/OpenEarthMap/val/images
mkdir -p data/OpenEarthMap/val/masks

# 将你的图像和标签放入对应目录
# images/: 0001.tif, 0002.tif, ...
# masks/: 0001.tif, 0002.tif, ...
```

### 3. 运行实验

#### 方式 A：使用预设配置（推荐）

```bash
# OpenEarthMap（小图像，不需要滑动窗口）
python -m sam3.rs.run_experiment --preset openearthmap --save_vis

# LoveDA（大图像，使用滑动窗口）
python -m sam3.rs.run_experiment --preset loveda \
    --slide_crop 1024 \
    --slide_stride 512 \
    --save_vis

# WHU building extraction（二值任务）
python -m sam3.rs.run_experiment --preset whu \
    --conf_thd 0.5 \
    --prob_thd 0.5
```

#### 方式 B：使用自定义配置

```bash
# 修改配置文件
vim configs/experiment_custom.json

# 运行
python -m sam3.rs.run_experiment \
    --config configs/experiment_custom.json \
    --save_dir outputs/custom_run
```

#### 方式 C：完整自定义（所有参数）

```bash
python -m sam3.rs.run_experiment \
    --dataset my_dataset \
    --data_root /path/to/data \
    --split val \
    --subset 100 \
    --checkpoint weights/sam3/sam3.pt \
    --bpe_path assets/bpe_simple_vocab_16e6.txt.gz \
    --device cuda \
    --slide_crop 1024 \
    --slide_stride 512 \
    --conf_thd 0.1 \
    --prob_thd 0.0 \
    --use_semantic \
    --use_instance \
    --use_presence \
    --prompts_file configs/prompts.txt \
    --save_dir outputs/my_experiment \
    --save_vis
```

### 4. 查看结果

```bash
# 输出目录结构
outputs/my_experiment/
├── 0001_pred.png              # 分割预测（主输出）
├── 0001_logits.npy            # 分割 logtis
├── 0001_masks/               # 每个类别的二值掩码
│   ├── class_0.png
│   ├── class_1.png
│   └── ...
├── 0001_pred_visual.png      # 可视化结果（原图+预测+叠加）
└── metrics.json                # 评估指标（如果有 GT）
```

查看指标：
```bash
cat outputs/my_experiment/metrics.json
```

---

## 🔧 代码示例

### 基础推理

```python
from sam3.rs import SAM3RSSegmentor, InferenceConfig

# 初始化模型
config = InferenceConfig(
    checkpoint_path='weights/sam3.pt',
    bpe_path='assets/bpe_simple_vocab_16e6.txt.gz',
    device='cuda',
    confidence_threshold=0.1,
    prob_threshold=0.0
)

model = SAM3RSSegmentor(config)

# 推理单张图片
result = model.predict_single('data/image.tif')

print(f"Prediction shape: {result.seg_pred.shape}")
print(f"Class IDs found: {result.seg_pred.unique()}")
```

### 批量推理

```python
from sam3.rs.data import get_dataset_loader
from sam3.rs import SAM3RSSegmentor, InferenceConfig

# 加载数据集
loader = get_dataset_loader('openearthmap')
image_paths = loader.get_samples(subset=100)  # 前 100 张

# 初始化模型
model = SAM3RSSegmentor(InferenceConfig(...))

# 批量推理
results = model.predict_batch(
    image_paths=image_paths,
    save_dir='outputs/',
    detailed=True
)

print(f"Processed {len(results)} images")
```

### 滑动窗口（大图像）

```python
from sam3.rs import SAM3RSSegmentor, InferenceConfig

# 大图像（例如 10000x10000）
config = InferenceConfig(
    checkpoint_path='weights/sam3.pt',
    slide_crop_size=512,    # 裁剪大小
    slide_stride=256,        # 步长（重叠）
    device='cuda'
)

model = SAM3RSSegmentor(config)

# 自动使用滑动窗口
result = model.predict_single('large_image.tif')

print(f"Large image segmentation complete!")
```

### 自定义数据集

```python
from sam3.rs.data import RSDataLoader, DatasetConfig
from sam3.rs import SAM3RSSegmentor, InferenceConfig

# 定义自定义数据集
config = DatasetConfig(
    name='MyCustomDataset',
    root_dir='data/my_dataset',
    split='val',
    num_classes=10,
    image_suffix='.jpg',
    mask_suffix='.png'
)

loader = RSDataLoader(config)
image_paths = loader.get_samples()

# 推理
model = SAM3RSSegmentor(InferenceConfig(...))
results = model.predict_batch(image_paths, 'outputs/')
```

### 评估（有 GT 时）

```python
from sam3.rs.metrics import RSEvaluator

evaluator = RSEvaluator(num_classes=9, ignore_index=255)

# 累积预测
for pred_path, gt_path in zip(preds, gts):
    evaluator.update(pred_path, gt_path)

# 计算指标
results = evaluator.compute()
results.print_summary()

# 保存结果
evaluator.save_results(results, 'outputs/metrics.json')
```

---

## 🎯 SegEarthOV3 特性已实现

✅ **双头融合**（Dual-head Fusion）
   - Instance head (Transformer decoder)
   - Semantic head
   - Element-wise MAX 融合

✅ **Presence Score 过滤**
   - 对不存在场景的类别进行惩罚
   - 减少大词汇量导致的虚假正例

✅ **滑动窗口推理**
   - 支持超大图像（10000x10000+）
   - 自动边界调整
   - 结果融合（平均重叠区域）

✅ **多类别提示词处理**
   - 同义词映射（如 "building,roof,house" → 类别 8）
   - 支持复杂提示词

✅ **概率阈值控制**
   - confidence_threshold: 实例检测阈值
   - prob_threshold: 最终分割阈值
   - 独立控制，灵活调整

---

## 🔬 实验指南

### 实验 1：对比不同头配置

```bash
# 仅使用 instance head
python -m sam3.rs.run_experiment \
    --preset loveda \
    --no-use_semantic \
    --save_dir outputs/loveda_instance_only

# 仅使用 semantic head
python -m sam3.rs.run_experiment \
    --preset loveda \
    --no-use_instance \
    --save_dir outputs/loveda_semantic_only

# 双头融合
python -m sam3.rs.run_experiment \
    --preset loveda \
    --save_dir outputs/loveda_dual_head

# 对比指标
diff <(cat outputs/loveda_instance_only/metrics.json) \
      <(cat outputs/loveda_dual_head/metrics.json)
```

### 实验 2：不同阈值对比

```bash
# 高阈值（严格）
python -m sam3.rs.run_experiment \
    --preset loveda \
    --conf_thd 0.7 \
    --save_dir outputs/loveda_high_thd

# 低阈值（宽松）
python -m sam3.rs.run_experiment \
    --preset loveda \
    --conf_thd 0.1 \
    --save_dir outputs/loveda_low_thd

# 无阈值
python -m sam3.rs.run_experiment \
    --preset loveda \
    --prob_thd 0.0 \
    --save_dir outputs/loveda_no_thd
```

### 实验 3：不同滑动窗口配置

```bash
# 小窗口（更多计算，更精确）
python -m sam3.rs.run_experiment \
    --preset loveda \
    --slide_crop 512 \
    --slide_stride 256 \
    --save_dir outputs/loveda_512window

# 大窗口（更少计算，可能精度降低）
python -m sam3.rs.run_experiment \
    --preset loveda \
    --slide_crop 2048 \
    --slide_stride 1024 \
    --save_dir outputs/loveda_2048window

# 无滑动窗口（原图）
python -m sam3.rs.run_experiment \
    --preset loveda \
    --slide_crop 0 \
    --save_dir outputs/loveda_noslide
```

---

## 📚 文档索引

- **README_RS.md**: 完整文档和 API 参考
- **ARCHITECTURE.md**: 框架架构详细设计
- **example_prompts.txt**: 提示词文件格式示例

---

## 🤔 常见问题

### Q: 如何添加新的数据集？

**A**: 在 `data.py` 中添加配置：

```python
DATASET_CONFIGS['my_dataset'] = DatasetConfig(
    name='MyDataset',
    root_dir='data/my_dataset',
    split='val',
    num_classes=10
)
```

然后使用：`get_dataset_loader('my_dataset')`

---

### Q: 如何修改双头融合策略？

**A**: 编辑 `segmentor.py` 的 `_inference_single_view` 方法，在 `# ===== SegEarthOV3's Dual-Head Fusion =====` 下修改融合逻辑。

---

### Q: 如何添加新的评估指标？

**A**: 在 `metrics.py` 的 `RSEvaluator.compute()` 方法中添加新的计算，并更新 `MetricResult` 数据类。

---

### Q: 是否需要安装 MMSegmentation？

**A**: **不需要！** SAM3-RS 是完全独立的轻量级框架，只依赖 PyTorch、NumPy、PIL、Matplotlib。

---

### Q: 如何处理超大图像（>20000x20000）？

**A**: 使用滑动窗口配置：

```bash
python -m sam3.rs.run_experiment \
    --preset loveda \
    --slide_crop 1024 \
    --slide_stride 512
```

会自动分块处理并融合结果。

---

## 🎓 下一步

1. **运行快速开始脚本**：`python quick_start.py`
2. **准备数据集**：按照目录结构放置文件
3. **运行第一个实验**：使用预设配置快速测试
4. **分析结果**：查看指标和可视化
5. **修改和迭代**：基于结果调整参数和策略

---

## 💡 开发建议

### 扩展功能（可选）

1. **提示词优化**：
   - 在 `prompts.py` 中添加 CLIP-based 提示词排序
   - 自动生成变体

2. **分布式推理**：
   - 添加多 GPU 支持
   - 批量并行处理

3. **集成其他模型**：
   - 添加 Grounding DINO
   - 添加 OpenSeg
   - 支持模型对比实验

4. **自动后处理**：
   - CRF 优化边界
   - 形态学操作
   - 小区域过滤

---

## 📞 遇到问题？

1. 检查 `README_RS.md` 完整文档
2. 查看 `ARCHITECTURE.md` 理解设计
3. 查看代码注释
4. 查看示例配置文件

祝实验顺利！🚀

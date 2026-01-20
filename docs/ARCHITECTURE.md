# SAM3-RS 框架架构文档

## 目录
- [总体架构](#总体架构)
- [核心模块详解](#核心模块详解)
- [数据流](#数据流)
- [关键算法](#关键算法)
- [扩展指南](#扩展指南)

---

## 总体架构

```
sam3-rs/
├── sam3/                    # SAM3 原始代码
│   ├── model/              # 模型定义
│   ├── config/             # 模型配置
│   └── __init__.py
│
└── rs/                     # 遥感专用扩展（SAM3-RS）
    ├── __init__.py        # 统一入口
    ├── segmentor.py       # 核心推理引擎
    ├── data.py            # 数据集管理
    ├── prompts.py         # 提示词优化
    ├── metrics.py         # 评估指标
    ├── config.py          # 配置管理
    ├── utils.py           # 工具函数
    └── run_experiment.py  # 实验运行脚本
```

### 架构设计原则

1. **轻量化**：最小依赖（仅 PyTorch + Pillow）
2. **模块化**：每个模块职责单一
3. **可扩展**：预留扩展接口
4. **易调试**：清晰的代码结构和日志

---

## 核心模块详解

### 1. Segmentor（推理引擎）- `segmentor.py`

**职责**：封装 SAM3 模型，提供高级推理接口

**核心功能**：
```python
class SAM3RSSegmentor:
    def __init__(self, config: ExperimentConfig):
        # 加载 SAM3 模型
        # 初始化处理器
        # 配置双头融合参数
        
    def predict(self, image, prompts):
        # 单图推理
        # 支持双头融合
        # 支持滑动窗口
        # 返回分割结果
        
    def predict_batch(self, images, prompts):
        # 批量推理
```

**关键算法**：
- **双头融合**：Transformer Decoder 头 + Semantic Segmentation 头
- **滑动窗口**：大图自动分块、边界调整、结果融合
- **Presence Score**：抑制不存在类别的误检

---

### 2. Data（数据集管理）- `data.py`

**职责**：统一加载和预处理遥感数据集

**核心功能**：
```python
class RemoteSensingDataset:
    def __init__(self, name, root, config):
        # 加载数据集配置
        # 构建文件列表
        # 加载类别映射
        
    def __len__(self):
        # 返回数据集大小
        
    def __getitem__(self, idx):
        # 返回 (image_path, mask_path, class_id)
        
    def get_prompts(self, class_id):
        # 返回类别对应的文本提示
```

**支持的数据集**：
- OpenEarthMap（语义分割）
- LoveDA（城市场景）
- WHU（建筑提取）
- Inria（建筑提取）
- xBD（灾害评估）
- CHN6-CUG（道路提取）
- UAVid（无人机影像）
- DLRSD（语义分割）

---

### 3. Prompts（提示词管理）- `prompts.py`

**职责**：优化和生成遥感地物类别提示词

**核心功能**：
```python
class PromptManager:
    def __init__(self, prompt_file=None):
        # 加载提示词配置
        # 构建同义词映射
        
    def get_prompt(self, class_id):
        # 返回类别的优化提示词
        # 支持同义词融合
        
    def load_from_file(self, filepath):
        # 从文件加载提示词
        # 格式: "class_id:prompt1,prompt2,..."
        
    def optimize_prompt(self, base_prompt, context):
        # 根据上下文优化提示词（预留）
```

**提示词格式**：
```txt
# example_prompts.txt
0:background
1:building,roof,house,construction
2:road,highway,street,path
3:water,river,lake,sea
4:vegetation,forest,tree,grass
5:car,vehicle,truck
```

---

### 4. Metrics（评估指标）- `metrics.py`

**职责**：计算分割质量指标

**核心功能**：
```python
class SegmentationMetrics:
    def update(self, pred, gt):
        # 更新混淆矩阵
        
    def compute(self):
        # 计算指标
        # 返回: mIoU, F1, Pixel Acc, Per-class IoU
```

**支持指标**：
- **mIoU**（Mean Intersection over Union）：核心指标
- **F1-score**：精确率和召回率的调和平均
- **Pixel Accuracy**：像素级准确率
- **Per-class IoU**：每类别的 IoU

**实现方式**：
- 基于混淆矩阵计算
- 无 MMSeg 依赖
- 支持 GPU 加速

---

### 5. Config（配置管理）- `config.py`

**职责**：管理实验配置参数

**核心功能**：
```python
@dataclass
class ExperimentConfig:
    # 模型配置
    model_type: str = 'SAM3'
    checkpoint_path: str
    bpe_path: str
    
    # 推理配置
    confidence_threshold: float = 0.3
    prob_thd: float = 0.1
    use_presence_score: bool = True
    
    # 滑动窗口配置
    slide_crop: int = 1024
    slide_stride: int = 768
    use_slide: bool = False
    
    # 数据集配置
    dataset_name: str = 'OpenEarthMap'
    dataset_root: str
    bg_idx: int = 0
    
    # 输出配置
    save_vis: bool = False
    output_dir: str = './output'
```

**预设配置**：
- `PRESETS`: 预定义配置字典
- 支持快速切换数据集
- 支持命令行覆盖参数

---

### 6. Utils（工具函数）- `utils.py`

**职责**：提供通用工具函数

**核心功能**：
```python
# 可视化
def visualize_prediction(image, pred, gt, output_path):
    # 可视化预测 vs 真值
    
# 文件 I/O
def save_mask(mask, output_path):
    # 保存掩码图像
    
def load_image(image_path):
    # 加载图像（支持多种格式）
    
# 滑动窗口辅助
def adjust_boundary(crop, stride, img_size):
    # 调整裁剪边界，避免边缘信息丢失
```

---

## 数据流

### 推理流程

```
输入图像 (大图)
    ↓
[滑动窗口分块] (可选)
    ↓
多个子图像
    ↓
[预处理] (Resize, Normalize)
    ↓
SAM3 模型推理
    ↓
┌──────────────────┐
│  Transformer     │ → masks_logits (实例级)
│  Decoder Head    │
├──────────────────┤
│  Semantic Head   │ → semantic_logits (像素级)
└──────────────────┘
    ↓
[双头融合] (MAX 融合)
    ↓
[Presence Score 过滤] (可选)
    ↓
[阈值后处理]
    ↓
拼接/融合 (滑动窗口)
    ↓
输出分割结果
```

### 实验流程

```
1. 配置加载
   ↓
2. 数据集初始化
   ├─ 加载图像列表
   ├─ 加载标注
   └─ 加载提示词映射
   ↓
3. 模型初始化
   ├─ 加载 SAM3 checkpoint
   └─ 初始化处理器
   ↓
4. 推理循环
   for each image:
       ├─ [滑动窗口] (可选)
       ├─ SAM3 推理
       ├─ 双头融合
       └─ 保存结果
   ↓
5. 评估
   ├─ 加载真值标注
   ├─ 计算混淆矩阵
   └─ 计算 mIoU, F1 等
   ↓
6. 结果输出
   ├─ metrics.json
   └─ 可视化图像 (可选)
```

---

## 关键算法

### 1. 双头融合（Dual-Head Fusion）

**原理**：结合 Transformer Decoder 的精确性和 Semantic Head 的完整性

```python
# segmentor.py:70-105
for query_idx, prompt in enumerate(prompts):
    # Instance head (Transformer decoder)
    masks_logits = inference_state['masks_logits']
    scores = inference_state['object_scores']
    
    # Weighted instance logits
    instance_logits = masks_logits * scores
    instance_logits = instance_logits.max(0)  # MAX 聚合
    
    # Semantic head
    semantic_logits = inference_state['semantic_mask_logits']
    
    # Fused result: MAX(instance, semantic)
    seg_logits[query_idx] = np.maximum(
        instance_logits * instance_score,
        semantic_logits
    )
```

**优势**：
- Instance head：精确目标定位
- Semantic head：全局覆盖保证
- MAX 融合：保留最佳预测

---

### 2. Presence Score 过滤

**原理**：利用 SAM3 的类别存在性分数抑制虚假检测

```python
# segmentor.py:107-110
if self.use_presence_score:
    presence_score = inference_state['presence_score']
    # 惩罚不存在的类别
    seg_logits[query_idx] *= presence_score
```

**场景**：遥感图像词汇量大、patch 处理易产生误检

---

### 3. 滑动窗口推理

**原理**：大图分块处理，避免信息丢失

```python
# segmentor.py:114-157
def _slide_inference(self, image, prompts, crop_size, stride):
    h, w = image.shape[:2]
    preds = np.zeros((len(prompts), h, w))
    count_mat = np.zeros((h, w))
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 边界调整
            y_end = min(y + crop_size, h)
            x_end = min(x + crop_size, w)
            
            # 裁剪
            crop = image[y:y_end, x:x_end]
            
            # 推理
            pred = self._single_inference(crop, prompts)
            
            # 累加结果
            preds[:, y:y_end, x:x_end] += pred
            count_mat[y:y_end, x:x_end] += 1
    
    # 平均融合
    preds = preds / count_mat
    return preds
```

**关键点**：
- 边界调整：避免裁剪超出图像
- 叠加融合：使用 stride < crop_size 减少边缘效应
- 归一化：除以覆盖次数

---

### 4. 多类别提示词处理

**原理**：同义词映射提升召回率

```python
# data.py:29-31
synonym_mapping = {
    'building': 8,
    'roof': 8,
    'house': 8,
    'construction': 8,
}
```

**处理流程**：
```python
# prompts.py
def get_prompt(self, class_id):
    synonyms = self.class_to_prompts[class_id]
    # 组合同义词: "building,roof,house"
    prompt = ','.join(synonyms)
    return prompt
```

**优势**：
- 覆盖更多语义表达
- 提升模型召回率
- 易于人工优化

---

## 扩展指南

### 添加新数据集

**步骤**：

1. **添加数据集配置**
```python
# data.py
DATASET_CONFIGS['MyDataset'] = {
    'img_dir': 'images',
    'mask_dir': 'masks',
    'img_ext': '.png',
    'mask_ext': '.png',
    'num_classes': 10,
    'class_names': ['背景', '建筑', '道路', ...]
}
```

2. **添加预设配置**
```python
# config.py
PRESETS['mydataset'] = ExperimentConfig(
    dataset_name='MyDataset',
    dataset_root='/path/to/mydataset',
    # ... 其他参数
)
```

3. **运行实验**
```bash
python -m sam3.rs.run_experiment --preset mydataset
```

---

### 添加新提示词策略

**步骤**：

1. **扩展 PromptManager**
```python
# prompts.py
class PromptManager:
    def optimize_for_context(self, base_prompt, region_type):
        """根据区域类型优化提示词"""
        if region_type == 'urban':
            return base_prompt + ',urban area'
        elif region_type == 'rural':
            return base_prompt + ',rural area'
        return base_prompt
```

2. **使用新策略**
```python
# run_experiment.py
prompt_mgr = PromptManager(config.prompt_file)
optimized_prompt = prompt_mgr.optimize_for_context(
    base_prompt='building',
    region_type='urban'
)
```

---

### 添加新融合策略

**步骤**：

1. **修改 segmentor.py**
```python
class SAM3RSSegmentor:
    def _fusion_strategy_max(self, instance_logits, semantic_logits):
        """MAX 融合（现有）"""
        return np.maximum(instance_logits, semantic_logits)
    
    def _fusion_strategy_weighted(self, instance_logits, semantic_logits, 
                                   alpha=0.7):
        """加权融合（新增）"""
        return alpha * instance_logits + (1 - alpha) * semantic_logits
    
    def _fusion_strategy_voting(self, instance_logits, semantic_logits):
        """投票融合（新增）"""
        # 实现投票逻辑
        pass
```

2. **配置策略**
```python
# config.py
@dataclass
class ExperimentConfig:
    fusion_strategy: str = 'max'  # 'max', 'weighted', 'voting'
```

---

### 添加新评估指标

**步骤**：

1. **扩展 SegmentationMetrics**
```python
# metrics.py
class SegmentationMetrics:
    def compute_dice(self):
        """计算 Dice 系数"""
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        return dice
    
    def compute(self):
        results = {
            'mIoU': self.compute_miou(),
            'F1': self.compute_f1(),
            'Dice': self.compute_dice(),  # 新增
            'PixelAcc': self.compute_pixel_acc(),
        }
        return results
```

---

## 性能优化建议

### 1. 批量推理

```python
# segmentor.py
def predict_batch(self, images, prompts, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = self._batch_inference(batch, prompts)
        results.extend(batch_results)
    return results
```

### 2. GPU 内存优化

```python
# 清理 GPU 缓存
torch.cuda.empty_cache()

# 使用混合精度
with torch.cuda.amp.autocast():
    output = model(input)
```

### 3. 多进程数据加载

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,
    pin_memory=True
)
```

---

## 调试技巧

### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看中间结果
print(f"Image shape: {image.shape}")
print(f"Masks logits shape: {masks_logits.shape}")
print(f"Presence score: {presence_score}")
```

### 2. 可视化中间步骤

```python
# utils.py
def debug_visualize(image, instance_logits, semantic_logits, fused_logits):
    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(instance_logits)
    axes[1].set_title('Instance Head')
    axes[2].imshow(semantic_logits)
    axes[2].set_title('Semantic Head')
    axes[3].imshow(fused_logits)
    axes[3].set_title('Fused')
    plt.savefig('debug_visualization.png')
```

### 3. 单元测试

```python
# test_segmentor.py
def test_dual_head_fusion():
    segmentor = SAM3RSSegmentor(config)
    instance_logits = np.random.rand(1, 256, 256)
    semantic_logits = np.random.rand(1, 256, 256)
    
    fused = segmentor._dual_head_fusion(instance_logits, semantic_logits)
    
    assert fused.shape == instance_logits.shape
    assert np.all(fused >= instance_logits) or np.all(fused >= semantic_logits)
```

---

## 常见问题

### Q1: 如何处理超大图像（> 10000x10000）？

**A**: 使用滑动窗口 + 分块保存
```python
# 分块推理并保存
for i, crop in enumerate(large_crops):
    pred = segmentor.predict(crop, prompts)
    save_mask(pred, f'output_{i}.png')
# 后处理合并
merge_masks('output_*.png', 'final.png')
```

### Q2: 如何提升推理速度？

**A**: 
1. 使用更大的 stride（减少重叠）
2. 使用更小的 crop_size
3. 减少 prompts 数量
4. 使用 GPU 并行

### Q3: 如何调优阈值参数？

**A**:
```python
# 网格搜索
for conf_thd in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for prob_thd in [0.05, 0.1, 0.15, 0.2]:
        results = run_experiment(conf_thd, prob_thd)
        print(f"conf_thd={conf_thd}, prob_thd={prob_thd}, mIoU={results['mIoU']}")
```

---

## 总结

SAM3-RS 框架的核心优势：

1. **轻量级**：无重型依赖，易于部署
2. **模块化**：清晰的模块划分，易于理解
3. **可扩展**：预留扩展接口，支持定制
4. **完整功能**：涵盖 SegEarthOV3 所有特性
5. **文档齐全**：API 文档、架构文档、示例代码

通过本文档，你应该能够：
- 理解框架的整体架构
- 掌握核心模块的使用方法
- 理解关键算法的实现原理
- 知道如何扩展框架功能

**开始你的遥感图像分割实验吧！** 🚀

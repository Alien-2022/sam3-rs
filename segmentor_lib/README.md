# SAM3-RS Segmentor Refactoring

## 重构目标

将 `segmentor.py` (1244 行) 重构为模块化、可维护的代码结构。

## 重构架构

```
workspace/core/sam3-rs/
├── segmentor.py              # 主入口 (~420 行，对外接口)
├── segmentorV1.py            # 原始备份 (1244 行)
└── segmentor_lib/           # 核心模块目录
    ├── __init__.py           # 模块入口，暴露核心接口
    ├── README.md             # 本文件
    ├── core.py               # InferenceEngine 核心推理引擎
    ├── analyzers.py          # 分析器/钩子 (PresenceScoreAnalyzer)
    ├── prompts.py            # 提示词加载和预计算
    ├── postprocess.py        # 后处理 (融合、resize、预测)
    ├── sliding_window.py     # 滑动窗口推理
    ├── debug.py              # 内存调试工具
    └── experimental/         # 实验性功能（可插拔）
        ├── __init__.py       # 实验模块入口
        └── semantic_enhancement.py  # 语义增强
```

## 模块说明

### 核心模块（稳定）

#### 1. core.py - InferenceEngine
核心推理引擎，包含单图和批量推理逻辑。

**主要方法：**
- `inference_single_view(image, detailed, image_name)` - 单图推理
- `inference_batch_view(images, detailed, image_names)` - 批量推理
- `get_analyzer(analyzer_class)` - 按类型获取 Analyzer

#### 2. prompts.py - 提示词管理
处理提示词加载和文本特征预计算。

**主要函数：**
- `load_prompts(prompts_file)` - 加载提示词配置
- `precompute_text_features(processor, prompts, device, num_prompts)` - 预计算文本特征

#### 3. postprocess.py - 后处理
推理结果的后处理操作。

**主要函数：**
- `fuse_prompts_to_classes(seg_logits, query_indices, num_classes, num_prompts)` - 同义词融合
- `logits_to_pred(logits, use_prompted_background, prob_threshold, bg_idx)` - logits 转预测
- `resize_logits(logits, target_shape)` - 调整尺寸

#### 4. analyzers.py - 分析器
基于钩子的分析器系统，支持扩展。

**主要类：**
- `BaseAnalyzer` - 抽象基类，定义钩子接口
- `PresenceScoreAnalyzer` - 收集和报告 presence score 统计
- `StatisticsAnalyzer` - 综合统计分析器

**钩子点：**
| 钩子点 | 调用时机 |
|--------|----------|
| `on_before_inference` | 每张/批图像推理前 |
| `on_after_prompt` | 每个prompt处理后 |
| `on_after_inference` | 每张/批图像推理后 |
| `on_after_eval` | 每张图像评估完成后 (含 GT 和预测) |

**StatisticsAnalyzer 功能：**
- 类别级分析: IoU, Precision, Recall, F1 分布
- 混淆矩阵: 识别类别混淆模式
- 尺寸分层分析: 小/中/大目标性能对比
- 错误模式分析: 漏检 vs 误检
- 类别共现分析: 多类别场景性能

#### 5. sliding_window.py - 滑动窗口
大图像的滑动窗口推理。

**主要类：**
- `SlidingWindowInference` - 封装滑动窗口逻辑

#### 6. debug.py - 调试工具
内存调试和日志记录。

**主要类：**
- `MemoryDebugger` - tensor 内存追踪、CUDA 内存监控

---

### 实验性模块（可能被移除）

> ⚠️ **注意**：这些模块是实验性功能，可能在未来的版本中被移除或大幅修改。

#### 1. experimental/semantic_enhancement.py - 语义增强
同义词语义增强策略。

**主要函数：**
- `apply_semantic_enhancement(processor, prompts, num_classes, device)` - 选择代表词模式
- `apply_semantic_enhancement_with_avg_embedding(...)` - 平均嵌入模式
- `get_semantic_enhancer(mode)` - 工厂函数

#### 2. experimental/unsupervised_threshold.py - 无监督阈值校准
基于置信度分布的无监督阈值校准，无需标注数据。

**主要类：**
- `UnsupervisedThresholdCalibration` - 无监督阈值校准器

**主要方法：**
- `collect_statistics(segmentor, image_paths, prompt_names, num_classes)` - 收集统计信息
- `estimate_confidence_threshold(percentile)` - 估计置信度阈值
- `estimate_global_prob_threshold(percentile)` - 估计全局概率阈值
- `estimate_prob_threshold_per_class()` - 估计每类概率阈值
- `calibrate(...)` - 一键完成所有校准

---

## 使用方式

### 1. 基础使用（通过 SAM3RSSegmentor）

```python
from segmentor import SAM3RSSegmentor, InferenceConfig

# 配置
config = InferenceConfig(
    checkpoint_path="path/to/checkpoint.pt",
    bpe_path="path/to/bpe.model",
    prompts_file="prompts.txt",
    device="cuda"
)

# 创建 segmentor
segmentor = SAM3RSSegmentor(config)

# 单图推理
result = segmentor.predict_single("image.jpg")

# 批量推理
results = segmentor.predict_batch(["img1.jpg", "img2.jpg"])
```

### 2. 启用分析器

```python
# 方式1: 配置文件中启用
config.analyze_presence_score = True
segmentor = SAM3RSSegmentor(config)

# 运行推理
segmentor.predict_single("image.jpg")

# 打印统计
from segmentor_lib.analyzers import PresenceScoreAnalyzer
analyzer = segmentor.engine.get_analyzer(PresenceScoreAnalyzer)
if analyzer:
    analyzer.report()
```

### 3. 启用实验性功能

```python
# 语义增强
config.semantic_enhancement_mode = "select_word"  # 或 "avg_embedding"
```

### 4. 无监督阈值校准

```python
from segmentor_lib.experimental.unsupervised_threshold import UnsupervisedThresholdCalibration

# 初始化校准器（收集 50 个样本的统计信息）
calibrator = UnsupervisedThresholdCalibration(num_samples=50)

# 收集统计信息（无需标注数据）
calibrator.collect_statistics(
    segmentor=segmentor,
    image_paths=image_paths,  # 任意测试图像
    prompt_names=segmentor.prompts['names'],
    num_classes=segmentor.num_classes
)

# 估计置信度阈值（使用 30% 百分位）
confidence_threshold = calibrator.estimate_confidence_threshold(percentile=30)
# 输出示例：
#   Confidence Score Statistics:
#     Min:       0.1234
#     Mean:      0.6543
#     Median:    0.6789
#     30%:       0.4567 ← threshold
#     Max:       0.9999
#     Final:     0.4567

# 估计全局概率阈值（使用 50% 百分位，即中位数）
prob_threshold = calibrator.estimate_global_prob_threshold(percentile=50)

# 或者估计每个类别的概率阈值
prob_thresholds_per_class = calibrator.estimate_prob_threshold_per_class()
# 输出示例：
# Class    Count     Min        P40        Median     P60        Max        Threshold
# 0        50        0.1234     0.4567     0.5432     0.6234     0.8901     0.5432
# 1        50        0.2345     0.5234     0.6123     0.7012     0.9234     0.6123

# 一键完成所有校准
confidence_threshold, prob_thresholds = calibrator.calibrate(
    segmentor=segmentor,
    image_paths=image_paths,
    prompt_names=segmentor.prompts['names'],
    num_classes=segmentor.num_classes,
    confidence_percentile=30,
    prob_percentile=50,
    use_per_class_prob=False  # 使用全局阈值而非每类阈值
)

# 更新配置
config.confidence_threshold = confidence_threshold
if prob_thresholds is None:
    # 使用全局 prob_threshold
    config.prob_threshold = 0.5876  # 从 calibrate() 输出获取
else:
    # 使用每类 prob_threshold（需要修改代码支持）
    pass
```

**使用命令行工具：**

```bash
# 运行校准脚本
python demo_calibration.py \
    --config eval/configs/loveda.yaml \
    --num_samples 50 \
    --confidence_percentile 30 \
    --prob_percentile 50 \
    --output assistant/calibration_results.txt

# 结果将保存到文件，包含推荐的阈值
```

### 5. 启用调试功能

```python
config.debug_memory = True
config.debug_log_file = "debug.log"

segmentor = SAM3RSSegmentor(config)
# 内存使用信息将写入 debug.log
```

### 5. 自定义分析器

```python
from segmentor_lib.analyzers import BaseAnalyzer

class TimingAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        import time
        self.start_time = None

    def on_before_inference(self, context):
        self.start_time = time.time()

    def on_after_inference(self, context):
        elapsed = time.time() - self.start_time
        print(f"Elapsed: {elapsed:.2f}s")

    def on_after_prompt(self, context):
        pass

# 注册分析器
timing_analyzer = TimingAnalyzer()
segmentor.engine.register_analyzer(timing_analyzer)
```

---

## 删除实验功能

如果实验功能验证失败需要移除，只需：

```bash
# 1. 删除实验功能目录
rm -rf segmentor_lib/experimental/

# 2. 从 InferenceConfig 中移除相关配置项
# 3. 从 segmentor.py 中移除实验功能的导入和调用
```

---

## 优势对比

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| 主文件行数 | 1244 | ~420 |
| 模块化程度 | 单文件 | 8个独立模块 |
| 核心代码可读性 | 调试代码混杂 | 纯净逻辑 |
| 添加新功能 | 需修改 segmentor.py | 添加新 Analyzer/模块 |
| 调试功能 | 分散在代码中 | 统一 MemoryDebugger |
| 单元测试 | 难以独立测试 | 每个模块独立测试 |
| API 稳定性 | - | 对外接口不变 |
| 实验功能隔离 | 混在核心代码中 | 独立 experimental 目录 |

---

## 外部接口变化

**保持不变：**
- `SAM3RSSegmentor.__init__(config)`
- `predict_single(image_path, detailed)`
- `predict_batch(image_paths, detailed)`

**已移除：**
- `print_presence_score_stats()` - 改用 `engine.get_analyzer(PresenceScoreAnalyzer).report()`

---

## 命令行使用

```bash
# 基础推理
python workspace/core/sam3-rs/eval/run_eval.py --config loveda.yaml

# 启用 presence score 分析
python workspace/core/sam3-rs/eval/run_eval.py --config loveda.yaml --analyze-presence-score

# 启用内存调试
python workspace/core/sam3-rs/eval/run_eval.py --config loveda.yaml --debug-memory
```

---

## 迁移历史

### 阶段 1: 核心推理 (已完成)
- 创建 `core.py` - InferenceEngine
- 创建 `analyzers.py` - 钩子系统

### 阶段 2: 功能模块化 (已完成)
- 创建 `prompts.py` - 提示词管理
- 创建 `postprocess.py` - 后处理
- 创建 `sliding_window.py` - 滑动窗口

### 阶段 3: 调试工具 (已完成)
- 创建 `debug.py` - MemoryDebugger
- 移除 `segmentor.py` 中的调试方法

### 阶段 4: 清理和优化 (已完成)
- 删除冗余代码
- 更新 `run_eval.py` 使用新接口
- 简化 `segmentor.py` 为对外接口

### 阶段 5: 实验功能隔离 (已完成)
- 创建 `experimental/` 目录
- 移动 `adaptive_threshold.py` 到 experimental
- 拆分 `prompts.py`，语义增强功能移入 experimental
- 添加 `__init__.py` 文件暴露接口

---

## 注意事项

1. **向后兼容**: `segmentor.py` 对外接口保持不变
2. **渐进式迁移**: 每个模块独立完成，不影响其他部分
3. **备份保留**: 原始代码备份在 `segmentorV1.py`
4. **实验功能可插拔**: experimental 目录可随时移除

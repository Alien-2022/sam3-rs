# SAM3-RS Segmentor Refactoring

## 重构目标

将 `segmentor.py` (1244 行) 重构为模块化、可维护的代码结构。

## 重构架构

```
workspace/core/sam3-rs/
├── segmentor.py              # 主入口 (~355 行，对外接口)
├── segmentorV1.py            # 原始备份 (1244 行)
└── segmentor_lib/           # 核心模块目录
    ├── __init__.py
    ├── README.md             # 本文件
    ├── core.py               # InferenceEngine 核心推理引擎
    ├── analyzers.py          # 分析器/钩子 (PresenceScoreAnalyzer)
    ├── prompts.py            # 提示词加载和语义增强
    ├── postprocess.py        # 后处理 (融合、resize、预测)
    ├── sliding_window.py     # 滑动窗口推理
    └── debug.py              # 内存调试工具
```

## 重构进度

### ✅ 已完成
- [x] `core.py` - InferenceEngine 核心推理逻辑 (432 行)
- [x] `analyzers.py` - BaseAnalyzer 和 PresenceScoreAnalyzer (133 行)
- [x] `prompts.py` - 提示词加载、预计算、语义增强 (196 行)
- [x] `postprocess.py` - 同义词融合、logits 转预测、resize (119 行)
- [x] `sliding_window.py` - 滑动窗口推理 (108 行)
- [x] `debug.py` - MemoryDebugger 内存调试 (71 行)
- [x] `segmentor.py` 重构 - 简化为对外接口 (~355 行)
- [x] `run_eval.py` 更新 - 使用新的 Analyzer 报告接口

### 📋 待完成
- [ ] 添加单元测试
- [ ] 性能基准测试
- [ ] 完善文档和注释

## 模块说明

### 1. core.py - InferenceEngine
核心推理引擎，包含单图和批量推理逻辑。

**主要方法：**
- `inference_single_view(image, detailed, image_name)` - 单图推理
- `inference_batch_view(images, detailed, image_names)` - 批量推理
- `get_analyzer(analyzer_class)` - 按类型获取 Analyzer

### 2. analyzers.py - 分析器
基于钩子的分析器系统，支持扩展。

**主要类：**
- `BaseAnalyzer` - 抽象基类，定义钩子接口
- `PresenceScoreAnalyzer` - 收集和报告 presence score 统计

**钩子点：**
| 钩子点 | 调用时机 |
|--------|----------|
| `on_before_inference` | 每张/批图像推理前 |
| `on_after_prompt` | 每个prompt处理后 |
| `on_after_inference` | 每张/批图像推理后 |

### 3. prompts.py - 提示词管理
处理提示词加载、文本特征预计算和语义增强。

**主要函数：**
- `load_prompts(prompts_file)` - 加载提示词配置
- `precompute_text_features(processor, prompts, device, num_prompts)` - 预计算文本特征
- `apply_semantic_enhancement(processor, prompts, num_classes, device)` - 语义增强（选择代表词）

### 4. postprocess.py - 后处理
推理结果的后处理操作。

**主要函数：**
- `fuse_prompts_to_classes(seg_logits, query_indices, num_classes, num_prompts)` - 同义词融合
- `logits_to_pred(logits, use_prompted_background, prob_threshold, bg_idx)` - logits 转预测
- `resize_logits(logits, target_shape)` - 调整尺寸

### 5. sliding_window.py - 滑动窗口
大图像的滑动窗口推理。

**主要类：**
- `SlidingWindowInference` - 封装滑动窗口逻辑

### 6. debug.py - 调试工具
内存调试和日志记录。

**主要类：**
- `MemoryDebugger` - tensor 内存追踪、CUDA 内存监控

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

### 3. 启用调试功能

```python
config.debug_memory = True
config.debug_log_file = "debug.log"

segmentor = SAM3RSSegmentor(config)
# 内存使用信息将写入 debug.log
```

### 4. 自定义分析器

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

## 优势对比

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| 主文件行数 | 1244 | ~355 |
| 模块化程度 | 单文件 | 6个独立模块 |
| 核心代码可读性 | 调试代码混杂 | 纯净逻辑 |
| 添加新功能 | 需修改 segmentor.py | 添加新 Analyzer/模块 |
| 调试功能 | 分散在代码中 | 统一 MemoryDebugger |
| 单元测试 | 难以独立测试 | 每个模块独立测试 |
| API 稳定性 | - | 对外接口不变 |

## 外部接口变化

**保持不变：**
- `SAM3RSSegmentor.__init__(config)`
- `predict_single(image_path, detailed)`
- `predict_batch(image_paths, detailed)`

**已移除：**
- `print_presence_score_stats()` - 改用 `engine.get_analyzer(PresenceScoreAnalyzer).report()`

## 命令行使用

```bash
# 基础推理
python workspace/core/sam3-rs/eval/run_eval.py --config loveda.yaml

# 启用 presence score 分析
python workspace/core/sam3-rs/eval/run_eval.py --config loveda.yaml --analyze-presence-score

# 启用内存调试
python workspace/core/sam3-rs/eval/run_eval.py --config loveda.yaml --debug-memory
```

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

## 注意事项

1. **向后兼容**: `segmentor.py` 对外接口保持不变
2. **渐进式迁移**: 每个模块独立完成，不影响其他部分
3. **备份保留**: 原始代码备份在 `segmentorV1.py`

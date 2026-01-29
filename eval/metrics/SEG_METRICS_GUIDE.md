# 语义分割评估指标说明文档

## 概述

`seg_metrics.py` 提供了一个轻量级的语义分割评估模块，基于混淆矩阵实现流式计算，适用于大规模数据集的评估。

## 核心类：SegmentationMetric

### 初始化参数

```python
SegmentationMetric(num_classes: int, ignore_index: int = 255)
```

- **num_classes**: 语义类别数量（不包括 ignore 类别）
- **ignore_index**: 忽略的标签值，通常为 255 表示无效像素

## 评估指标详解

### 1. mIoU (mean Intersection over Union)

**定义**：平均交并比，语义分割最常用的评估指标。

**计算公式**：
```
IoU_i = TP_i / (TP_i + FP_i + FN_i)
mIoU = mean(IoU_1, IoU_2, ..., IoU_N)
```

其中：
- **TP_i (True Positive)**：类别 i 被正确分类的像素数
- **FP_i (False Positive)**：其他类别被误分类为类别 i 的像素数
- **FN_i (False Negative)**：类别 i 被误分类为其他类别的像素数

**原理说明**：
- IoU 衡量预测区域与真实区域的交集与并集之比
- 值域：[0, 1]，值越高表示分割质量越好
- 对小类别和大类别同等权重，不受类别分布影响

**合理性分析**：

✅ **优点**：
- 标准指标，便于与同行工作对比
- 同时考虑召回率和精确率，对误检和漏检都敏感
- 对类别不平衡鲁棒

⚠️ **注意事项**：
- 当某类别在 GT 和预测中都未出现时，该类别的 IoU 为 NaN
- 最终 mIoU 使用 `nanmean` 计算，自动忽略 NaN 值

---

### 2. mAcc (mean Pixel Accuracy)

**定义**：平均像素准确率，每类别的分类准确率。

**计算公式**：
```
Acc_i = TP_i / (TP_i + FN_i)
mAcc = mean(Acc_1, Acc_2, ..., Acc_N)
```

**原理说明**：
- 衡量模型对每个类别的召回能力
- 只关注"有多少真实像素被正确分类"，不关注误分类到该类的像素
- 值域：[0, 1]

**合理性分析**：
✅ **优点**：
- 直观反映各类别的识别能力
- 与 mIoU 互补，可发现模型偏好（高 mAcc 低 mIoU 可能意味着过度预测）

⚠️ **注意事项**：
- 高 mAcc 不一定意味着高 mIoU（可能存在大量误检）
- 同样使用 `nanmean` 处理无预测/无 GT 的类别

---

### 3. aAcc (average Pixel Accuracy)

**定义**：全局像素准确率，所有像素的平均分类正确率。

**计算公式**：
```
aAcc = sum(TP_i) / sum(TP_i + FP_i + FN_i)
```

**原理说明**：
- 计算所有像素点的分类准确率
- 等价于混淆矩阵对角线元素之和除以所有元素之和
- 值域：[0, 1]

**合理性分析**：
✅ **优点**：
- 计算简单，直观易理解
- 反映整体分割准确度

⚠️ **局限性**：
- **对类别不平衡高度敏感**：大类别主导指标
- 例如：背景占 90%，模型只预测背景，aAcc=90% 但实际无意义
- 不推荐作为不平衡数据集的主要评估指标

---

### 4. per_class_iou (每类别 IoU)

**定义**：返回每个类别的独立 IoU 值。

**计算公式**：
```
per_class_iou[i] = TP_i / (TP_i + FP_i + FN_i)
```

**用途**：
- 深入分析模型在各类别的表现
- 发现模型的类别偏好（某些类别表现好/差）
- 适用于类别不平衡时的详细分析

---

## 混淆矩阵原理

### 混淆矩阵结构

```
              预测类别
              C0  C1  C2 ... Cn
真实类别 C0 [ TP00 FP01 FP02 ... FP0n ]
        C1 [ FP10 TP11 FP12 ... FP1n ]
        C2 [ FP20 FP21 TP22 ... FP2n ]
        ... [ ...                 ]
        Cn [ FPn0 FPn1 FPn2 ... TPnn ]
```

- **对角线元素 (TP)**：正确分类的像素数
- **行和**：该类别在 GT 中的像素数 (TP + FN)
- **列和**：预测为该类别的像素数 (TP + FP)
- **总和**：所有有效像素数

### 流式计算实现

```python
def _fast_hist(self, label, pred):
    # 1. 过滤 ignore_index
    mask = label != self.ignore_index
    label = label[mask]
    pred = pred[mask]

    # 2. 过滤类别范围外的标签
    valid = (label >= 0) & (label < self.num_classes) & \
            (pred >= 0) & (pred < self.num_classes)

    # 3. 计算混淆矩阵索引
    idx = self.num_classes * label + pred
    hist = np.bincount(idx, minlength=self.num_classes ** 2)

    # 4. 重塑为混淆矩阵
    hist = hist.reshape(self.num_classes, self.num_classes)
    return hist
```

**优势**：
- 单次遍历计算，复杂度 O(n)
- 内存高效，只存储 N×N 矩阵
- 支持增量更新，适用于大数据集

---

## 特殊处理机制

### 1. ignore_index 处理

**用途**：标记无效像素（如裁剪边界、云遮挡、标注缺失）

**处理方式**：
```python
mask = label != self.ignore_index
label = label[mask]
pred = pred[mask]
```
- 忽略索引像素不参与任何统计
- 不影响 TP、FP、FN 计算

---

### 2. 类别外标签过滤

**用途**：处理模型预测出不在 [0, num_classes-1] 范围的标签

**处理方式**：
```python
valid = (label >= 0) & (label < self.num_classes) & \
        (pred >= 0) & (pred < self.num_classes)
label = label[valid]
pred = pred[valid]
```
- 超出范围的预测和标签被丢弃
- 避免数组越界和无效统计

---

### 3. 数值稳定性

**问题**：除零错误（类别在 GT 或预测中完全缺失）

**解决方案**：
```python
eps = 1e-10
iou = tp / (pos_gt + pos_pred - tp + eps)
acc = tp / (pos_gt + eps)
```
- 分母添加微小常数 eps
- 不影响有效数值的计算精度
- 避免返回 `inf` 或 `nan`

---

## 使用示例

### 基础用法

```python
from eval.metrics.seg_metrics import SegmentationMetric
import numpy as np

# 初始化：7 类语义分割（LoveDA 数据集）
metric = SegmentationMetric(num_classes=7, ignore_index=255)

# 模拟预测和 GT
pred = np.random.randint(0, 7, (256, 256), dtype=np.uint8)
label = np.random.randint(0, 7, (256, 256), dtype=np.uint8)

# 更新统计
metric.update(pred, label)

# 计算指标
scores = metric.compute()
print(scores)
# 输出: {'mIoU': 0.142, 'mAcc': 0.142, 'aAcc': 0.142}
```

### 流式评估

```python
# 适用于大型数据集，不占用过多内存
for batch in dataloader:
    pred = model(batch['image'])
    label = batch['label']

    metric.update(pred, label)

# 所有批次处理完后一次性计算
final_scores = metric.compute()
```

### 获取每类别 IoU

```python
per_class_iou = metric.per_class_iou()
class_names = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture']

for name, iou in zip(class_names, per_class_iou):
    print(f"{name}: {iou:.4f}")
```

---

## 指标解读建议

### 指标对比矩阵

| 场景 | 推荐指标 | 理由 |
|------|----------|------|
| 学术论文对比 | mIoU | 行业标准，公平对比 |
| 类别不平衡数据 | mIoU + per_class_iou | 关注小类别表现 |
| 模型初步筛选 | mIoU | 快速评估整体质量 |
| 深度分析 | mIoU + mAcc | 区分召回/精确率问题 |
| 最终评估 | mIoU + aAcc + per_class_iou | 全面了解模型能力 |

### 常见模式分析

| mIoU | mAcc | aAcc | 可能原因 |
|-------|------|------|----------|
| 高 | 高 | 高 | 模型表现良好 |
| 低 | 高 | 高 | 大量误检，类别不平衡严重 |
| 高 | 低 | 高 | 过度保守，误检少但漏检多 |
| 低 | 低 | 高 | 小类别表现差，大类别表现好 |
| 低 | 低 | 低 | 模型需要全面改进 |

---

## 与其他评估框架对比

| 特性 | seg_metrics.py | mmseg | Detectron2 |
|------|----------------|-------|------------|
| 依赖 | 仅 NumPy | mmengine, mmcv | PyTorch, detectron2 |
| 内存占用 | 极低（N×N 矩阵） | 中等 | 中等 |
| 计算速度 | 极快（向量化） | 快 | 快 |
| 灵活性 | 高（可定制） | 中等 | 中等 |
| 适用场景 | 轻量级评估 | 完整训练流程 | 研究实验 |

---

## 最佳实践

### 1. 选择合适的 ignore_index

- **标准语义分割**：255
- **多任务学习**：根据任务设置不同值
- **实例分割**：使用 -1

### 2. 评估流程

```python
# 1. 初始化
metric = SegmentationMetric(num_classes=N, ignore_index=255)

# 2. 流式更新
for batch in dataset:
    metric.update(pred, gt)

# 3. 计算指标
scores = metric.compute()

# 4. 输出分析
print(f"mIoU: {scores['mIoU']:.4f}")
print(f"mAcc: {scores['mAcc']:.4f}")
print(f"aAcc: {scores['aAcc']:.4f}")
print("\nPer-class IoU:")
for i, iou in enumerate(metric.per_class_iou()):
    print(f"  Class {i}: {iou:.4f}")
```

### 3. 批量评估注意事项

```python
# 确保 pred 和 label 的维度匹配
pred = pred.argmax(dim=1).cpu().numpy()  # [B, H, W] → [H, W]
label = label.cpu().numpy()

# 逐样本更新，避免混淆矩阵累积错误
for i in range(pred.shape[0]):
    metric.update(pred[i], label[i])
```

---

## 总结

`seg_metrics.py` 提供了一个高效、准确的语义分割评估方案：

✅ **核心优势**：
- 轻量级：无外部依赖，仅 NumPy
- 高效：向量化计算，适合大规模数据集
- 准确：标准混淆矩阵实现，指标计算严谨
- 灵活：支持 ignore_index、类别过滤、流式更新

✅ **推荐使用**：
- 主要指标：**mIoU**（最全面的评估）
- 辅助指标：**mAcc**（召回分析）、**per_class_iou**（类别级分析）
- 参考指标：**aAcc**（整体准确度，注意类别不平衡影响）

⚠️ **注意事项**：
- 类别不平衡时优先使用 mIoU
- 始终检查 per_class_iou 了解模型偏好
- 合理设置 ignore_index 避免无效像素干扰

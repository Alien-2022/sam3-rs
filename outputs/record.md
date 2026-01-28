# 试验记录
## predict_single
```json
{
  "mIoU": 0.26468825981101757,
  "mAcc": 0.518471345033135,
  "aAcc": 0.45844266126520644,
  "per_class_iou": {
    "background": 0.36740050057002344,
    "building": 0.28229304470802824,
    "road": 0.2675856273057767,
    "water": 0.4387291106770057,
    "barren": 0.13344601282930132,
    "forest": 0.13826596231636618,
    "agriculture": 0.2250975602706215
  },
  "prob_threshold": 0.5,
  "confidence_threshold": 0.1
}
```

## predict_batch
```json
{
  "mIoU": 0.2519022966001659,
  "mAcc": 0.44912483963603717,
  "aAcc": 0.44364065346002507,
  "per_class_iou": {
    "background": 0.3809120405223756,
    "building": 0.25916911522917485,
    "road": 0.3305429218283146,
    "water": 0.398409672421499,
    "barren": 0.11199566340166989,
    "forest": 0.13186743318957006,
    "agriculture": 0.15041922960855714
  },
  "prob_threshold": 0.5,
  "confidence_threshold": 0.1
}
```
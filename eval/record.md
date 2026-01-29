# 试验记录
## predict_single
```json
# float32
{
  "mIoU": 0.2408727175458656,
  "mAcc": 0.5157849950191079,
  "aAcc": 0.4216112747248222,
  "per_class_iou": {
    "background": 0.3446556230578692,
    "building": 0.19616632413793283,
    "road": 0.2887614628660695,
    "water": 0.43376030090398615,
    "barren": 0.10005157882873011,
    "forest": 0.1763459375188405,
    "agriculture": 0.1463677955076307
  }
}
# bfloat16
{
  "mIoU": 0.24325518060664772,
  "mAcc": 0.5186455587219323,
  "aAcc": 0.4271626779713051,
  "per_class_iou": {
    "background": 0.35260114836500833,
    "building": 0.19348748369367247,
    "road": 0.28836922577993734,
    "water": 0.43383402922360365,
    "barren": 0.10192883030081783,
    "forest": 0.18648946095622862,
    "agriculture": 0.14607608592726576
  }
}
```

## predict_batch
### LoveDA
images: 1-120
```python
# bfloat16
{
  "mIoU": 0.25727913480159254,
  "mAcc": 0.4497233713426327,
  "aAcc": 0.4527964220551619,
  "per_class_iou": {
    "background": 0.3835701934289131,
    "building": 0.2824866511361364,
    "road": 0.32370001415027594,
    "water": 0.40106193201313867,
    "barren": 0.10622125711696341,
    "forest": 0.13572137261713496,
    "agriculture": 0.16819252314858554
  },
  "prob_threshold": 0.5,
  "confidence_threshold": 0.5,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": false,
  "prompts": {
    "names": [
      "background",
      "building",
      "road",
      "water",
      "barren",
      "forest",
      "agriculture"
    ]
  },
  "run_time": "0129_1358"
}

{
  "mIoU": 0.40349948609373726,
  "mAcc": 0.5343060929543869,
  "aAcc": 0.6310186466476768,
  "per_class_iou": {
    "background": 0.5103268074580884,
    "building,house": 0.5952755847821758,
    "road": 0.4495308826785018,
    "water": 0.3739966197148019,
    "barren,bareland,soil": 0.3486398900192539,
    "forest,tree": 0.07773053570461211,
    "agricultural": 0.4689960822987265
  },
  "prob_threshold": 0.5,
  "confidence_threshold": 0.5,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "building,house",
      "road",
      "water",
      "barren,bareland,soil",
      "forest,tree",
      "agricultural"
    ],
    "num_classes": 7,
    "num_prompts": 11
  },
  "run_time": "0129_1451"
}
```
whole images
```python
{
  "mIoU": 0.3656012947374089,
  "mAcc": 0.5463910054137766,
  "aAcc": 0.5123227849860149,
  "per_class_iou": {
    "background": 0.3523698816539767,
    "building,house": 0.5098341822550853,
    "road": 0.5102812032366598,
    "water": 0.38721716352578256,
    "barren,bareland,soil": 0.21897353682432644,
    "forest,tree": 0.2846337320979808,
    "agricultural": 0.29589936356805013
  },
  "prob_threshold": 0.5,
  "confidence_threshold": 0.5,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": false,
  "prompts": {
    "names": [
      "background",
      "building,house",
      "road",
      "water",
      "barren,bareland,soil",
      "forest,tree",
      "agricultural"
    ],
    "num_classes": 7,
    "num_prompts": 11
  },
  "run_time": "0129_1437"
}
```
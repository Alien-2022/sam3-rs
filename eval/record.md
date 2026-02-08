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
```python
{
  "mIoU": 0.47406092818297985,
  "mAcc": 0.6203538651369878,
  "aAcc": 0.6384518193902684,
  "per_class_iou": {
    "background": 0.4559237351821273,
    "building,house": 0.6380031739095836,
    "road": 0.5388812572177888,
    "water": 0.5145835977803307,
    "barren,bareland,soil": 0.35757179474610495,
    "forest,tree": 0.3384234930626181,
    "agricultural": 0.4750394453823056
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
  "run_time": "0203_2218"
}
```


### OpenEarthMap

```python
# prompts include background
{
  "mIoU": 0.42207676571524866,
  "mAcc": 0.6680068036914069,
  "aAcc": 0.625824923262487,
  "per_class_iou": {
    "background": 0.16577296288079713,
    "bareland,barren": 0.10627941847225456,
    "grass": 0.3740099646380354,
    "pavement": 0.23451839951109482,
    "road": 0.4174932485155203,
    "tree,forest": 0.6263201350710951,
    "water,river": 0.7264529685676762,
    "cropland": 0.42717494414777984,
    "building,roof,house": 0.7206688496329846
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.1,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "bareland,barren",
      "grass",
      "pavement",
      "road",
      "tree,forest",
      "water,river",
      "cropland",
      "building,roof,house"
    ],
    "num_classes": 9,
    "num_prompts": 14
  },
  "run_time": "0201_0346"
}
# prompts exclude background
{
  "mIoU": 0.4519051729578808,
  "mAcc": 0.6385198248762893,
  "aAcc": 0.6243053469640819,
  "per_class_iou": {
    "bareland,barren": 0.07098074493504967,
    "grass": 0.37407545163799905,
    "pavement": 0.23452805644729421,
    "road": 0.4175456174418445,
    "tree,forest": 0.626420474568423,
    "water,river": 0.7435139236892899,
    "cropland": 0.4274242886954835,
    "building,roof,house": 0.7207528262476625
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.1,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "bareland,barren",
      "grass",
      "pavement",
      "road",
      "tree,forest",
      "water,river",
      "cropland",
      "building,roof,house"
    ],
    "num_classes": 8,
    "num_prompts": 13
  },
  "run_time": "0201_0643"
}
```

### iSAID
```python
{
  "mIoU": 0.34784094607004246,
  "mAcc": 0.47576070141423943,
  "aAcc": 0.5229346515759904,
  "per_class_iou": {
    "large vehicle": 0.20717353191228197,
    "small vehicle": 0.5407116827040065,
    "harbor": 0.4468349692904411,
    "ship": 0.29554235618120367,
    "ground track field": 0.051747272195485756,
    "soccerball field": 0.22298031429469098,
    "baseball diamond": 0.04586497061243312,
    "swimming pool": 0.5537742693038646,
    "roundabout": 0.43958034660888284,
    "bridge": 0.2174050960475747,
    "tennis court": 0.33371366905787053,
    "basketball court": 0.10319594238776097,
    "plane": 0.9141951482443488,
    "helicopter": 0.23053010175495658,
    "storage tank": 0.614364520454835
  },
  "prob_threshold": 0.5,
  "confidence_threshold": 0.4,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "large vehicle",
      "small vehicle",
      "harbor",
      "ship",
      "ground track field",
      "soccerball field",
      "baseball diamond",
      "swimming pool",
      "roundabout",
      "bridge",
      "tennis court",
      "basketball court",
      "plane",
      "helicopter",
      "storage tank"
    ],
    "num_classes": 15,
    "num_prompts": 15
  },
  "run_time": "0201_2318"
}
```

### Potsdam
```python
{
  "mIoU": 0.5564132842864472,
  "mAcc": 0.7183652200208174,
  "aAcc": 0.7768676328659058,
  "per_class_iou": {
    "clutter": 0.19079181733580072,
    "road": 0.7035887028402845,
    "building": 0.8143184837664226,
    "grass": 0.6130281996497586,
    "tree": 0.4227684914227682,
    "car": 0.5939840107036481
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.2,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": false,
  "prompts": {
    "names": [
      "clutter",
      "road",
      "building",
      "grass",
      "tree",
      "car"
    ],
    "num_classes": 6,
    "num_prompts": 6
  },
  "run_time": "0204_0032"
}
```

### Vaihingen
```python
# 切片后
{
  "mIoU": 0.5672098130671359,
  "mAcc": 0.7558504351284846,
  "aAcc": 0.7804944283161781,
  "per_class_iou": {
    "clutter": 0.04156209976085197,
    "road": 0.697546180696842,
    "building": 0.8514050767921953,
    "grass": 0.4645403021602283,
    "tree": 0.69460122065504,
    "car": 0.6536039983376578
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.4,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "clutter",
      "road",
      "building",
      "grass",
      "tree",
      "car"
    ],
    "num_classes": 6,
    "num_prompts": 6
  },
  "run_time": "0204_1637"
}
# 切片前
{
  "mIoU": 0.553166798258943,
  "mAcc": 0.7814889993893548,
  "aAcc": 0.7871650920759639,
  "per_class_iou": {
    "clutter": 0.061584280995001336,
    "road": 0.6972314427866895,
    "building": 0.8611610683374367,
    "grass": 0.5010071052520115,
    "tree": 0.6837060839943884,
    "car": 0.5143108081881309
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.4,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "clutter",
      "road",
      "building",
      "grass",
      "tree",
      "car"
    ],
    "num_classes": 6,
    "num_prompts": 6
  },
  "run_time": "0206_2318"
}
```

### UAVid
```python
# 切片后，按原数据集类别
{
  "mIoU": 0.47622316188342917,
  "mAcc": 0.6080665330939028,
  "aAcc": 0.7838751475016276,
  "per_class_iou": {
    "background": 0.51205688774897,
    "building": 0.9132951771569179,
    "road": 0.6385649489913015,
    "tree": 0.5474290825713808,
    "vegetation": 0.5156252506880231,
    "moving car": 0.17053393749394366,
    "static car": 0.11963841161477543,
    "human": 0.39264159880212074
  },
  "prob_threshold": 0.3,
  "confidence_threshold": 0.3,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "building",
      "road",
      "tree",
      "vegetation",
      "moving car",
      "static car",
      "human"
    ],
    "num_classes": 8,
    "num_prompts": 8
  },
  "run_time": "0205_0007"
}
# 切片后，类别名优化后
{
  "mIoU": 0.5752007700439132,
  "mAcc": 0.705113530911873,
  "aAcc": 0.8256503211127387,
  "per_class_iou": {
    "background": 0.4702021985734755,
    "building": 0.9122912835793028,
    "road": 0.638466147049312,
    "tree": 0.7440699533316074,
    "grass": 0.5992778226876209,
    "driving vehicle": 0.3431517919380814,
    "parking vehicle": 0.5032377718989551,
    "human": 0.39090919129295104
  },
  "prob_threshold": 0.3,
  "confidence_threshold": 0.3,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "building",
      "road",
      "tree",
      "grass",
      "driving vehicle",
      "parking vehicle",
      "human"
    ],
    "num_classes": 8,
    "num_prompts": 8
  },
  "run_time": "0205_0025"
}
# 切片前，类别名优化后
{
  "mIoU": 0.5253546403574549,
  "mAcc": 0.650009483844239,
  "aAcc": 0.820916920519333,
  "per_class_iou": {
    "background": 0.4899124398894829,
    "building": 0.9096642051149775,
    "road": 0.6732242093720066,
    "tree": 0.7171094946009299,
    "grass": 0.5942192205514711,
    "driving vehicle": 0.15351011315340216,
    "parking vehicle": 0.4540278976187022,
    "human": 0.21116954255866643
  },
  "prob_threshold": 0.3,
  "confidence_threshold": 0.3,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "building",
      "road",
      "tree",
      "grass",
      "driving vehicle",
      "parking vehicle",
      "human"
    ],
    "num_classes": 8,
    "num_prompts": 8
  },
  "run_time": "0207_1751"
}
```

### UDD5
```python
# 先切片(1024x1024)再分割
{
  "mIoU": 0.6582121431400427,
  "mAcc": 0.8025991117733616,
  "aAcc": 0.8499296887430657,
  "per_class_iou": {
    "background": 0.32835122133198047,
    "vegetation": 0.8776514800911537,
    "building": 0.883124573509144,
    "road": 0.5988441321628741,
    "vehicle": 0.6030893086050612
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.5,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "vegetation",
      "building",
      "road",
      "vehicle"
    ],
    "num_classes": 5,
    "num_prompts": 5
  },
  "run_time": "0205_1551"
}
# 直接拿原图分割
{
  "mIoU": 0.7167113359558999,
  "mAcc": 0.8307877410504183,
  "aAcc": 0.8714044170914359,
  "per_class_iou": {
    "background": 0.45210385414816406,
    "vegetation": 0.8806075020234048,
    "building": 0.8969873794396269,
    "road": 0.640596896480756,
    "vehicle": 0.7132610476875473
  },
  "prob_threshold": 0.1,
  "confidence_threshold": 0.5,
  "use_semantic_head": true,
  "use_instance_head": true,
  "use_presence_score": true,
  "prompts": {
    "names": [
      "background",
      "vegetation",
      "building",
      "road",
      "vehicle"
    ],
    "num_classes": 5,
    "num_prompts": 5
  },
  "run_time": "0206_0112"
}
```
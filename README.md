# Hand Pose Regression

This project implements CNN-based models for 2D hand keypoint regression using datasets such as [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/).  
It supports both a simple CNN and a MobileNetV2-style inverted bottleneck model, with metrics like L1 loss and PCK (Percentage of Correct Keypoints).

## Features
- Preprocessing pipeline for FreiHAND (and similar hand datasets) into JSON format
- Train/validation split generator
- Two backbones:
  - Simple CNN (lightweight)
  - MobileLite with inverted bottleneck (MobileNetV2-inspired)
- Training loop with:
  - L1 regression loss
  - PCK evaluation metric
  - Cosine Annealing LR scheduler
- Progress bars via tqdm
- Model checkpoint saving (last.pth and best.pth)

## Setup

Clone the repository:
```bash
git clone https://github.com/DataMas/hand-pose-regression.git
cd hand-pose-regression
```

Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:
```
pip install -r requirements.txt
Dataset Preparation
```

Download the FreiHAND dataset and preprocess it:
```
python scripts/freihand_to_json.py
python scripts/split_dataset.py
```

This generates:

- train.json

- val.json

Each entry contains:
```
{
  "image_path": "data/rgb/00000000.jpg",
  "keypoints": [x1, y1, x2, y2, ...],
  "source": "FreiHAND"
}
```

Training
Run training with either backbone:
```
# Simple CNN
python src/train_keypoints.py --train_json data/processed/train.json --val_json data/processed/val.json --model simple

# MobileNetV2-style inverted bottleneck
python src/train_keypoints.py --train_json data/processed/train.json --val_json data/processed/val.json --model mobile

```

Example output:
```
Train 3/50: 100%|████████████████████████| 915/915 [00:12<00:00, 76.11it/s, loss=0.0421]
Eval  3/50: 100%|████████████████████████| 102/102 [00:01<00:00, 85.31it/s, loss=0.0385]
[003/050] train=0.0421 | val=0.0385 | PCK@0.05=72.3% | lr=2.99e-04
```

Results
Model checkpoints are saved in runs_keypoints/:

- best.pth

Evaluation reports L1 loss and PCK accuracy.

Roadmap
Add attention-augmented inverted bottleneck

Extend to other pose datasets (MPII, COCO, BodyHands)

Visualization utilities for predicted vs ground-truth keypoints

License
MIT License. See LICENSE for details.

Acknowledgments
- FreiHAND dataset
- MobileNetV2 authors for the inverted bottleneck design
- PyTorch team

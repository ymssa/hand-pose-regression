#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import your exported models
from models import SimpleCNN, MobileLiteRegressor


IM_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class KeypointDatasetEval(Dataset):
    """
    Expects a JSON list of:
      {"image_path": "...", "keypoints": [x1,y1,...,xK,yK], "source": "..."}
    - Resizes image to IM_SIZE x IM_SIZE
    - Converts keypoints from pixel coords to normalized [0,1] coords (same as training)
    """
    def __init__(self, json_path: str):
        super().__init__()
        self.items = json.load(open(json_path, "r"))
        self.root = Path(json_path).parent

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        p = Path(rec["image_path"])
        if not p.is_absolute():
            p = (self.root / p).resolve()

        img = Image.open(p).convert("RGB")
        W, H = img.size

        img = img.resize((IM_SIZE, IM_SIZE), resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        x = torch.from_numpy(arr.transpose(2, 0, 1))  # CHW

        kps = np.array(rec["keypoints"], dtype=np.float32).reshape(-1, 2)
        kps[:, 0] = kps[:, 0] * (IM_SIZE / W)
        kps[:, 1] = kps[:, 1] * (IM_SIZE / H)
        y = torch.from_numpy((kps / IM_SIZE).reshape(-1))  # normalized [0,1]

        return x, y


def l1_keypoint_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


@torch.no_grad()
def pck(pred, target, thr=0.05):
    """
    Percentage of Correct Keypoints in normalized coords.
    thr=0.05 corresponds to 5% of image size (~11.2 px at 224).
    """
    B, D = pred.shape
    K = D // 2
    pred = pred.view(B, K, 2)
    target = target.view(B, K, 2)
    dist = torch.norm(pred - target, dim=-1)  # [B,K] in normalized units
    return (dist < thr).float().mean().item()


def build_model(model_name: str, num_keypoints: int, device: torch.device):
    if model_name == "simple":
        model = SimpleCNN(num_keypoints=num_keypoints)
    elif model_name == "mobile":
        model = MobileLiteRegressor(num_keypoints=num_keypoints)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'simple' or 'mobile'.")
    return model.to(device)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a 2D keypoint regressor on a JSON dataset.")
    ap.add_argument("--val_json", type=str, default="val.json", help="Validation JSON path")
    ap.add_argument("--ckpt", type=str, default="runs_keypoints/best.pth", help="Checkpoint path")
    ap.add_argument("--model", type=str, default="simple", choices=["simple", "mobile"], help="Model architecture")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda', or 'auto'")
    ap.add_argument("--pck_thr", type=float, default=0.05)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    # Dataset and loader
    ds = KeypointDatasetEval(args.val_json)
    if len(ds) == 0:
        raise RuntimeError(f"No samples in {args.val_json}")
    sample_x, sample_y = ds[0]
    K = sample_y.numel() // 2

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # Build and load model
    model = build_model(args.model, K, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Evaluate
    n = 0
    loss_sum = 0.0
    pck_sum = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = l1_keypoint_loss(out, y)
            loss_sum += loss.item() * x.size(0)
            pck_sum += pck(out, y, thr=args.pck_thr) * x.size(0)
            n += x.size(0)

    print(f"Samples: {n}")
    print(f"Val L1 Loss: {loss_sum / n:.4f}")
    print(f"PCK@{args.pck_thr:.02f}: {pck_sum / n * 100:.2f}%")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")


if __name__ == "__main__":
    main()

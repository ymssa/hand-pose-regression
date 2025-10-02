#!/usr/bin/env python3
import os, json, math, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from tqdm import tqdm


# -------------------------
# Dataset
# -------------------------
class KeypointDataset(Dataset):
    """
    Expects a JSON list of:
      {"image_path": "...", "keypoints": [x1,y1,...,xK,yK], "source": "..."}
    - Resizes image to image_size x image_size
    - Converts keypoints from pixel coords to normalized [0,1] coords
    """
    def __init__(self, json_path: str, image_size: int = 224, is_train: bool = True, flip_prob: float = 0.5):
        super().__init__()
        self.items = json.load(open(json_path, "r"))
        self.root = Path(json_path).parent
        self.image_size = image_size
        self.is_train = is_train
        self.flip_prob = flip_prob if is_train else 0.0

        # imagenet-like normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        p = Path(rec["image_path"])
        if not p.is_absolute():
            p = (self.root / p).resolve()

        img = Image.open(p).convert("RGB")
        W, H = img.size

        # resize
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

        # keypoints to np [K,2] (pixels), then scale to new size and normalize to [0,1]
        kps = np.array(rec["keypoints"], dtype=np.float32).reshape(-1, 2)
        kps[:, 0] = kps[:, 0] * (self.image_size / W)
        kps[:, 1] = kps[:, 1] * (self.image_size / H)

        # optional flip (train only)
        if self.is_train and random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            kps[:, 0] = (self.image_size - 1) - kps[:, 0]
            # (If you need left/right semantic swap, do it here.)

        # normalize image
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = torch.from_numpy(arr.transpose(2, 0, 1))  # CHW

        # normalize coords to [0,1]
        kps_norm = (kps / self.image_size).reshape(-1)
        kps_norm = torch.from_numpy(kps_norm)

        return arr, kps_norm


# -------------------------
# Models
# -------------------------
class SimpleCNN(nn.Module):
    """Compact CNN for 2D keypoint regression."""
    def __init__(self, num_keypoints: int = 21):
        super().__init__()
        out_dim = num_keypoints * 2
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 224->112
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                         # 112->56
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                         # 56->28
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                         # 28->14
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


class InvertedBottleneck(nn.Module):
    """MobileNetV2-style block."""
    def __init__(self, in_ch, out_ch, expansion=4, stride=1):
        super().__init__()
        hid = in_ch * expansion
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True),
        ) if expansion != 1 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hid, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.project(y)
        return x + y if self.use_res else y


class MobileLiteRegressor(nn.Module):
    """A slightly stronger backbone using inverted bottlenecks; still light."""
    def __init__(self, num_keypoints: int = 21):
        super().__init__()
        out_dim = num_keypoints * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),  # 224->112
        )
        self.stage = nn.Sequential(
            InvertedBottleneck(32, 32, expansion=4, stride=1),
            InvertedBottleneck(32, 48, expansion=4, stride=2),  # 112->56
            InvertedBottleneck(48, 48, expansion=4, stride=1),
            InvertedBottleneck(48, 64, expansion=4, stride=2),  # 56->28
            InvertedBottleneck(64, 64, expansion=4, stride=1),
            InvertedBottleneck(64, 96, expansion=4, stride=2),  # 28->14
            InvertedBottleneck(96, 128, expansion=4, stride=2), # 14->7
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        return self.head(x)


# -------------------------
# Loss & Metric
# -------------------------
def l1_keypoint_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

@torch.no_grad()
def pck(pred, target, thr=0.05):
    """Percentage of Correct Keypoints in normalized coordinates."""
    B, D = pred.shape
    K = D // 2
    pred = pred.view(B, K, 2)
    target = target.view(B, K, 2)
    dist = torch.norm(pred - target, dim=-1)
    return (dist < thr).float().mean().item()


# -------------------------
# Training
# -------------------------
def train_one_epoch(model, loader, opt, device, epoch, epochs):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = l1_keypoint_loss(out, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, pck_thr, epoch, epochs):
    model.eval()
    loss_sum, pck_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"Eval {epoch}/{epochs}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = l1_keypoint_loss(out, y)
        loss_sum += loss.item() * x.size(0)
        pck_sum += pck(out, y, thr=pck_thr) * x.size(0)
        n += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / n, pck_sum / n



def main(
    train_json="train.json",
    val_json="val.json",
    model_type="simple",          # "simple" or "mobile"
    image_size=224,
    epochs=10,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-4,
    num_workers=4,
    pck_thr=0.05,
    outdir="runs_keypoints"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = KeypointDataset(train_json, image_size=image_size, is_train=True)
    val_ds   = KeypointDataset(val_json,   image_size=image_size, is_train=False, flip_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_kps = len(train_ds.items[0]["keypoints"]) // 2
    if model_type == "mobile":
        model = MobileLiteRegressor(num_keypoints=num_kps).to(device)
    else:
        model = SimpleCNN(num_keypoints=num_kps).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, ep, epochs)
        val_loss, val_pck = evaluate(model, val_loader, device, pck_thr=pck_thr, epoch=ep, epochs=epochs)
        sched.step()

        print(f"[{ep:03d}/{epochs}] train={tr_loss:.4f} | val={val_loss:.4f} | PCK@{pck_thr:.02f}={val_pck*100:.2f}% | lr={sched.get_last_lr()[0]:.2e}")

        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()}, outdir / "last.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()}, outdir / "best.pth")
            print("  ↳ saved best.pth")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train a CNN for 2D hand keypoint regression")
    ap.add_argument("--train_json", type=str, default="train.json")
    ap.add_argument("--val_json",   type=str, default="val.json")
    ap.add_argument("--model",      type=str, default="simple", choices=["simple","mobile"])
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs",     type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--pck_thr",    type=float, default=0.05)
    ap.add_argument("--outdir",     type=str, default="runs_keypoints")
    args = ap.parse_args()

    main(
        train_json=args.train_json,
        val_json=args.val_json,
        model_type=args.model,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pck_thr=args.pck_thr,
        outdir=args.outdir
    )

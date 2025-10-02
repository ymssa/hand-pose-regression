#!/usr/bin/env python3
import json
import random
from pathlib import Path
import argparse

def split_dataset(json_path, train_out, val_out, split_ratio=0.9, seed=0):
    random.seed(seed)

    with open(json_path, "r") as f:
        items = json.load(f)

    random.shuffle(items)
    split_idx = int(split_ratio * len(items))
    train, val = items[:split_idx], items[split_idx:]

    with open(train_out, "w") as f:
        json.dump(train, f, indent=2)
    with open(val_out, "w") as f:
        json.dump(val, f, indent=2)

    print(f"✓ Wrote {len(train)} samples to {train_out}")
    print(f"✓ Wrote {len(val)} samples to {val_out}")
    print(f"Split ratio: {split_ratio:.2f}, total: {len(items)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split dataset JSON into train/val")
    ap.add_argument("json_path", help="Input JSON file")
    ap.add_argument("--train-out", default="train.json", help="Output train JSON")
    ap.add_argument("--val-out", default="val.json", help="Output val JSON")
    ap.add_argument("--ratio", type=float, default=0.9, help="Train split ratio (default=0.9)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    args = ap.parse_args()

    split_dataset(args.json_path, args.train_out, args.val_out, args.ratio, args.seed)

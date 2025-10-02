#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

def project_points(XYZ: np.ndarray, K: np.ndarray) -> np.ndarray:
    # Project 3D joints (camera space) to 2D pixels using intrinsics K.
    # Inputs:
    #   XYZ: [21, 3]
    #   K:   [3, 3]
    # Returns:
    #   uv:  [21, 2]
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    Z = np.where(np.abs(Z) < 1e-8, 1e-8, Z)  # avoid div by zero
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def to_entry(img_path: Path, uv: np.ndarray, make_rel_to: Path = None) -> dict:
    if make_rel_to is not None:
        try:
            img_path = Path(img_path).resolve()
            rel = img_path.relative_to(make_rel_to.resolve())
            img_str = rel.as_posix()
        except Exception:
            img_str = str(img_path.as_posix())
    else:
        img_str = str(Path(img_path).as_posix())
    flat = uv.astype(float).reshape(-1).tolist()
    return {"image_path": img_str, "keypoints": flat, "source": "FreiHAND"}

def main():
    ap = argparse.ArgumentParser(description="Convert FreiHAND v2 training split to 2D keypoint JSON")
    ap.add_argument("--root", type=str, required=True, help="Path to FreiHAND_pub_v2")
    ap.add_argument("--out-json", type=str, required=True, help="Output JSON path")
    ap.add_argument("--relative-to", type=str, default=None, help="Store image paths relative to this directory")
    ap.add_argument("--skip-outside", action="store_true", help="Skip samples with any keypoint outside image bounds")
    ap.add_argument("--clamp-to-bounds", action="store_true", help="Clamp keypoints to image bounds instead of skipping")
    ap.add_argument("--take", type=int, default=None, help="Process only first N samples (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out_json)
    rel_base = Path(args.relative_to) if args.relative_to else None

    f_xyz = root / "data/training_xyz.json"
    f_K   = root / "data/training_K.json"
    rgb_dir = root / "data" / "rgb"

    if not f_xyz.exists() or not f_K.exists() or not rgb_dir.exists():
        raise FileNotFoundError("Expected files missing. Make sure --root has training_xyz.json, training_K.json and training/rgb/")

    xyz_list = load_json(f_xyz)   # list of [21,3]
    K_list   = load_json(f_K)     # list of [3,3]
    n = min(len(xyz_list), len(K_list))
    entries: List[dict] = []

    for i in range(n if args.take is None else min(n, args.take)):
        img_name = f"{i:08d}.png"
        img_path = rgb_dir / img_name
        if not img_path.exists():
            # fallback to .jpg if needed
            img_path = (rgb_dir / img_name).with_suffix(".jpg")
            if not img_path.exists():
                continue

        # image size
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue

        XYZ = np.array(xyz_list[i], dtype=np.float32).reshape(21, 3)
        K   = np.array(K_list[i], dtype=np.float32).reshape(3, 3)
        uv  = project_points(XYZ, K)  # [21,2]

        # bounds
        if args.skip_outside:
            if not np.all((uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)):
                continue
        if args.clamp_to_bounds:
            uv[:,0] = np.clip(uv[:,0], 0, W - 1)
            uv[:,1] = np.clip(uv[:,1], 0, H - 1)

        entries.append(to_entry(img_path, uv, make_rel_to=rel_base))

        if (i+1) % 2000 == 0:
            print(f"[{i+1}/{n}] processed...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"✓ Wrote {len(entries)} samples to {out_path}")

if __name__ == "__main__":
    main()

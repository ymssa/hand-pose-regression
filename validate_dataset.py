import json, random
from pathlib import Path
from typing import List
from pydantic import BaseModel, validator, ValidationError
from PIL import Image, ImageDraw

# --------------------------
# Pydantic schema
# --------------------------
class Sample(BaseModel):
    image_path: str
    keypoints: List[float]
    source: str

    @validator("image_path")
    def check_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Image file not found: {v}")
        return v

    @validator("keypoints")
    def check_keypoints(cls, v):
        if len(v) != 42:   # 21×2
            raise ValueError(f"Expected 42 values (21×2), got {len(v)}")
        return v

    @validator("source")
    def check_source(cls, v):
        if not v.strip():
            raise ValueError("Source string cannot be empty")
        return v


# --------------------------
# Visualization helper
# --------------------------
def show_sample(item, radius=2):
    img = Image.open(item["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    kps = item["keypoints"]
    for x, y in zip(kps[0::2], kps[1::2]):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     outline=(255,0,0), width=2)
    img.show()


# --------------------------
# Main validation
# --------------------------
def validate_json(path: str, max_samples: int = None, visualize: bool = False):
    with open(path, "r") as f:
        data = json.load(f)

    errors = 0
    valid_samples = []
    for i, item in enumerate(data):
        try:
            s = Sample(**item)
            valid_samples.append(item)
        except ValidationError as e:
            print(f"Error in sample {i}: {e}")
            errors += 1
        if max_samples and i >= max_samples:
            break

    if errors == 0:
        print(f"✓ All {len(valid_samples)} samples valid in {path}")
    else:
        print(f"✗ {errors} invalid samples in {path}")

    if visualize and valid_samples:
        print("Showing random valid samples with keypoints...")
        for _ in range(5):
            show_sample(random.choice(valid_samples))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Dataset JSON to validate")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--visualize", action="store_true", help="Show sample images with keypoints")
    args = ap.parse_args()

    validate_json(args.json_path, args.max_samples, args.visualize)

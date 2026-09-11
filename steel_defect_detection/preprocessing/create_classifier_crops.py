"""
Crop defect regions from ground-truth annotations for EfficientNet-B0 training.

Reads YOLO-format labels from data/yolo_format/ and crops the corresponding
regions from the images. Crops are saved in a folder-per-class layout for
torchvision.datasets.ImageFolder.

IMPORTANT: Uses GROUND-TRUTH annotations only — NOT model predictions.
This prevents data leakage between training and validation.

Output:
    data/classifier_format/
    ├── train/
    │   ├── crazing/
    │   ├── inclusion/
    │   ├── patches/
    │   ├── pitted_surface/
    │   ├── rolled-in_scale/
    │   └── scratches/
    └── validation/
        ├── crazing/ ...

Usage:
    python preprocessing/create_classifier_crops.py
    python preprocessing/create_classifier_crops.py --padding 0.15
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
YOLO_DIR = PROJECT_ROOT / "data" / "yolo_format"
OUT_DIR = PROJECT_ROOT / "data" / "classifier_format"

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

SPLITS = {
    "train": {"images": "images/train", "labels": "labels/train", "out": "train"},
    "val":   {"images": "images/val",   "labels": "labels/val",   "out": "validation"},
}


def crop_with_padding(image, x1, y1, x2, y2, padding_frac: float = 0.1):
    """Crop a region from an image with fractional padding, clamped to image bounds."""
    h, w = image.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)

    x1_pad = max(0, int(x1) - pad_x)
    y1_pad = max(0, int(y1) - pad_y)
    x2_pad = min(w, int(x2) + pad_x)
    y2_pad = min(h, int(y2) + pad_y)

    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
    return crop


def process_split(split_name: str, split_info: dict, padding: float) -> dict:
    """Process one split: read YOLO labels, crop from images, save crops."""
    images_dir = YOLO_DIR / split_info["images"]
    labels_dir = YOLO_DIR / split_info["labels"]
    out_name = split_info["out"]

    stats = defaultdict(int)

    if not images_dir.exists():
        print(f"  ERROR: {images_dir} does not exist")
        return dict(stats)
    if not labels_dir.exists():
        print(f"  ERROR: {labels_dir} does not exist")
        return dict(stats)

    # Create output class directories
    for cls_name in CLASSES:
        (OUT_DIR / out_name / cls_name).mkdir(parents=True, exist_ok=True)

    # Process each label file
    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"  Found {len(label_files)} label files")

    for lbl_file in label_files:
        stem = lbl_file.stem

        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            stats["images_missing"] += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["images_corrupt"] += 1
            continue

        h, w = image.shape[:2]

        # Read YOLO labels
        with open(lbl_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= len(CLASSES):
                stats["invalid_class"] += 1
                continue

            cls_name = CLASSES[cls_id]

            # YOLO format: cx, cy, bw, bh (normalized)
            cx = float(parts[1]) * w
            cy = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Crop with padding
            crop = crop_with_padding(image, x1, y1, x2, y2, padding)
            if crop.size == 0:
                stats["empty_crops"] += 1
                continue

            # Save crop
            crop_name = f"{stem}_obj{i}.jpg"
            save_path = OUT_DIR / out_name / cls_name / crop_name
            cv2.imwrite(str(save_path), crop)
            stats[cls_name] += 1
            stats["total"] += 1

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Create classifier crops from YOLO annotations")
    parser.add_argument("--padding", type=float, default=0.1,
                        help="Fractional padding around bounding box (default: 0.1 = 10%%)")
    args = parser.parse_args()

    print("Creating classifier crops from ground-truth annotations")
    print(f"Source: {YOLO_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Padding: {args.padding:.0%}")

    # Clean output
    if OUT_DIR.exists():
        print(f"\nRemoving existing output: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    all_stats = {}
    for split_name, split_info in SPLITS.items():
        print(f"\nProcessing {split_name}...")
        stats = process_split(split_name, split_info, args.padding)
        all_stats[split_name] = stats

    # Report
    print("\n" + "=" * 60)
    print("  CLASSIFIER CROP REPORT")
    print("=" * 60)

    for split_name, stats in all_stats.items():
        print(f"\n--- {split_name.upper()} ---")
        print(f"  Total crops: {stats.get('total', 0)}")
        for cls in CLASSES:
            print(f"    {cls}: {stats.get(cls, 0)}")
        if stats.get("images_missing", 0):
            print(f"  Images missing: {stats['images_missing']}")
        if stats.get("empty_crops", 0):
            print(f"  Empty crops: {stats['empty_crops']}")

    print("\nDone. Crops ready for EfficientNet-B0 training.")


if __name__ == "__main__":
    main()

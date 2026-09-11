"""
Dataset Inspection Script for NEU-DET Steel Defect Dataset.

Automatically inspects the actual dataset at:
    NEU-DET/train/   (images in class subdirs, annotations as flat XML)
    NEU-DET/validation/

Reports:
    - Number of images per split / class
    - Image dimensions, channels, formats
    - Annotation format, counts, class distribution
    - Bounding-box statistics (min/max/mean area, aspect ratio)
    - Missing / orphan annotations
    - Invalid bounding boxes
    - Unknown class names
    - Empty annotation files
    - Corrupt images
    - Duplicate filenames within and across splits
    - Cross-split duplicate images (by file hash)

This script does NOT modify any files.

Usage:
    python preprocessing/inspect_dataset.py
"""

import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths — resolve relative to the NEU-DET project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent                      # steel_defect_detection/
NEU_DET_ROOT = PROJECT_ROOT.parent                    # NEU-DET/

SPLITS = {
    "train": NEU_DET_ROOT / "train",
    "validation": NEU_DET_ROOT / "validation",
}

EXPECTED_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def file_md5(path: Path) -> str:
    """Return MD5 hex digest of a file (for duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_voc_xml(xml_path: Path) -> dict:
    """
    Parse a Pascal VOC XML annotation file.

    Returns dict with keys:
        filename, width, height, depth, objects (list of dicts with
        name, xmin, ymin, xmax, ymax, truncated, difficult)
    or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None

    root = tree.getroot()
    size_el = root.find("size")
    if size_el is None:
        return None

    w = int(size_el.findtext("width", "0"))
    h = int(size_el.findtext("height", "0"))
    d = int(size_el.findtext("depth", "0"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower().replace(" ", "_")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except ValueError:
            continue
        objects.append({
            "name": name,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "truncated": int(obj.findtext("truncated", "0")),
            "difficult": int(obj.findtext("difficult", "0")),
        })

    return {
        "filename": root.findtext("filename", ""),
        "width": w, "height": h, "depth": d,
        "objects": objects,
    }


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------
def inspect_split(split_name: str, split_dir: Path):
    """Inspect one split (train or validation) and return a report dict."""
    images_dir = split_dir / "images"
    annot_dir = split_dir / "annotations"

    report = {
        "split": split_name,
        "images_dir_exists": images_dir.exists(),
        "annot_dir_exists": annot_dir.exists(),
        "image_files": [],          # list of (relative_path, full_path)
        "annot_files": [],          # list of full_path
        "image_stems": set(),
        "annot_stems": set(),
        "class_image_counts": defaultdict(int),
        "class_annot_counts": defaultdict(int),  # from XML object names
        "dimensions": defaultdict(int),           # (w, h, d) -> count
        "formats": defaultdict(int),              # extension -> count
        "corrupt_images": [],
        "empty_annotations": [],
        "invalid_bboxes": [],
        "unknown_classes": set(),
        "bbox_areas": [],
        "bbox_aspect_ratios": [],
        "total_objects": 0,
        "image_hashes": {},          # stem -> md5
    }

    if not images_dir.exists():
        print(f"  WARNING: {images_dir} does not exist")
        return report

    # --- Collect images (in class subdirectories) ---
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = img_file.stem
            report["image_files"].append((f"{class_name}/{img_file.name}", img_file))
            report["image_stems"].add(stem)
            report["class_image_counts"][class_name] += 1
            report["formats"][img_file.suffix.lower()] += 1

            # Read image to check dimensions and corruption
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                report["corrupt_images"].append(str(img_file))
            else:
                if len(img.shape) == 2:
                    h, w = img.shape
                    d = 1
                else:
                    h, w, d = img.shape
                report["dimensions"][(w, h, d)] += 1

            # Hash for duplicate detection
            report["image_hashes"][stem] = file_md5(img_file)

    # --- Collect annotations (flat XML files) ---
    if annot_dir.exists():
        for xml_file in sorted(annot_dir.glob("*.xml")):
            report["annot_files"].append(xml_file)
            stem = xml_file.stem
            report["annot_stems"].add(stem)

            parsed = parse_voc_xml(xml_file)
            if parsed is None:
                report["empty_annotations"].append(str(xml_file))
                continue

            if len(parsed["objects"]) == 0:
                report["empty_annotations"].append(str(xml_file))

            for obj in parsed["objects"]:
                name = obj["name"]
                report["class_annot_counts"][name] += 1
                report["total_objects"] += 1

                if name not in EXPECTED_CLASSES:
                    # Try common variants
                    norm = name.replace("-", "_")
                    if norm not in EXPECTED_CLASSES:
                        report["unknown_classes"].add(name)

                # Validate bbox
                xmin, ymin = obj["xmin"], obj["ymin"]
                xmax, ymax = obj["xmax"], obj["ymax"]
                w_img = parsed["width"]
                h_img = parsed["height"]

                invalid_reasons = []
                if xmin >= xmax:
                    invalid_reasons.append("xmin >= xmax")
                if ymin >= ymax:
                    invalid_reasons.append("ymin >= ymax")
                if xmin < 0 or ymin < 0:
                    invalid_reasons.append("negative coordinate")
                if xmax > w_img or ymax > h_img:
                    invalid_reasons.append(f"bbox exceeds image ({w_img}x{h_img})")

                if invalid_reasons:
                    report["invalid_bboxes"].append({
                        "file": xml_file.name,
                        "bbox": [xmin, ymin, xmax, ymax],
                        "reasons": invalid_reasons,
                    })

                # Bbox stats
                area = max(0, xmax - xmin) * max(0, ymax - ymin)
                report["bbox_areas"].append(area)
                bw = max(0.01, xmax - xmin)
                bh = max(0.01, ymax - ymin)
                report["bbox_aspect_ratios"].append(bw / bh)

    return report


def print_report(reports: list):
    """Print a formatted dataset integrity report."""
    print("\n" + "=" * 70)
    print("  DATASET INTEGRITY REPORT")
    print("=" * 70)

    all_image_hashes = {}   # split -> {stem: hash}
    all_image_stems = {}    # split -> set of stems

    for r in reports:
        split = r["split"]
        n_images = len(r["image_files"])
        n_annots = len(r["annot_files"])
        all_image_hashes[split] = r["image_hashes"]
        all_image_stems[split] = r["image_stems"]

        print(f"\n--- {split.upper()} ---")
        print(f"  Images:      {n_images}")
        print(f"  Annotations: {n_annots}")
        print(f"  Total bbox objects: {r['total_objects']}")

        # Missing / orphan annotations
        missing = r["image_stems"] - r["annot_stems"]
        orphan = r["annot_stems"] - r["image_stems"]
        print(f"  Missing annotations (image without XML): {len(missing)}")
        if missing:
            for s in sorted(missing)[:10]:
                print(f"    - {s}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

        print(f"  Orphan annotations (XML without image):  {len(orphan)}")
        if orphan:
            for s in sorted(orphan)[:10]:
                print(f"    - {s}")
            if len(orphan) > 10:
                print(f"    ... and {len(orphan) - 10} more")

        # Image dimensions
        print(f"  Image dimensions (WxHxD -> count):")
        for (w, h, d), cnt in sorted(r["dimensions"].items()):
            print(f"    {w}x{h}x{d}: {cnt}")

        # Formats
        print(f"  Image formats:")
        for ext, cnt in sorted(r["formats"].items()):
            print(f"    {ext}: {cnt}")

        # Class distribution (from images)
        print(f"  Class distribution (by image folder):")
        for cls in EXPECTED_CLASSES:
            print(f"    {cls}: {r['class_image_counts'].get(cls, 0)}")

        # Class distribution (from annotations)
        print(f"  Class distribution (by annotation objects):")
        for cls in EXPECTED_CLASSES:
            print(f"    {cls}: {r['class_annot_counts'].get(cls, 0)}")

        # Unknown classes
        if r["unknown_classes"]:
            print(f"  ⚠ Unknown class names: {r['unknown_classes']}")

        # Corrupt images
        print(f"  Corrupt images: {len(r['corrupt_images'])}")
        for p in r["corrupt_images"][:5]:
            print(f"    - {p}")

        # Empty annotations
        print(f"  Empty annotation files: {len(r['empty_annotations'])}")
        for p in r["empty_annotations"][:5]:
            print(f"    - {p}")

        # Invalid bboxes
        print(f"  Invalid bounding boxes: {len(r['invalid_bboxes'])}")
        for inv in r["invalid_bboxes"][:5]:
            print(f"    - {inv['file']}: {inv['bbox']} ({', '.join(inv['reasons'])})")

        # Bbox statistics
        if r["bbox_areas"]:
            areas = np.array(r["bbox_areas"])
            ratios = np.array(r["bbox_aspect_ratios"])
            print(f"  Bounding box area stats:")
            print(f"    min: {areas.min():.0f}  max: {areas.max():.0f}  "
                  f"mean: {areas.mean():.1f}  median: {np.median(areas):.1f}")
            print(f"  Bounding box aspect ratio (w/h) stats:")
            print(f"    min: {ratios.min():.2f}  max: {ratios.max():.2f}  "
                  f"mean: {ratios.mean():.2f}  median: {np.median(ratios):.2f}")

        # Duplicate filenames within split
        stems = list(r["image_stems"])
        # Since stems are a set, no within-split duplicates by construction from stems
        # But let's check if same stem appears in multiple class folders
        stem_counts = defaultdict(int)
        for rel, full in r["image_files"]:
            stem_counts[full.stem] += 1
        within_dups = {s: c for s, c in stem_counts.items() if c > 1}
        print(f"  Within-split duplicate filenames: {len(within_dups)}")
        for s, c in list(within_dups.items())[:5]:
            print(f"    - {s} appears {c} times")

    # --- Cross-split checks ---
    print(f"\n--- CROSS-SPLIT CHECKS ---")

    if len(reports) >= 2:
        splits = list(all_image_stems.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                common_stems = all_image_stems[s1] & all_image_stems[s2]
                print(f"  Shared filenames ({s1} AND {s2}): {len(common_stems)}")
                if common_stems:
                    for s in sorted(common_stems)[:5]:
                        print(f"    - {s}")

                # Check by content hash
                hashes_1 = all_image_hashes[s1]
                hashes_2 = all_image_hashes[s2]
                hash_set_1 = set(hashes_1.values())
                hash_set_2 = set(hashes_2.values())
                common_hashes = hash_set_1 & hash_set_2
                print(f"  Exact duplicate images by content ({s1} AND {s2}): {len(common_hashes)}")

    print("\n" + "=" * 70)
    print("  INSPECTION COMPLETE")
    print("=" * 70 + "\n")


def main():
    print("NEU-DET Dataset Inspector")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"NEU-DET root: {NEU_DET_ROOT}")

    reports = []
    for split_name, split_dir in SPLITS.items():
        print(f"\nInspecting {split_name} at: {split_dir}")
        if not split_dir.exists():
            print(f"  ERROR: {split_dir} does not exist!")
            continue
        report = inspect_split(split_name, split_dir)
        reports.append(report)

    print_report(reports)


if __name__ == "__main__":
    main()

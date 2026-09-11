"""
Convert NEU-DET dataset from Pascal VOC XML annotations to YOLO format.

Reads from the ACTUAL dataset location:
    NEU-DET/train/images/<class>/*.jpg    + NEU-DET/train/annotations/*.xml
    NEU-DET/validation/images/<class>/*.jpg + NEU-DET/validation/annotations/*.xml

Outputs to:
    steel_defect_detection/data/yolo_format/images/{train,val}/*.jpg
    steel_defect_detection/data/yolo_format/labels/{train,val}/*.txt

Key design decisions:
    - PRESERVES the existing train/validation split (does NOT re-split)
    - Images are copied (flattened from class subdirs) — originals untouched
    - Bounding boxes are validated before writing
    - Class mapping verified against actual XML <name> tags
    - Generates a conversion report

Usage:
    python preprocessing/convert_neudet_to_yolo.py
"""

import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent                      # steel_defect_detection/
NEU_DET_ROOT = PROJECT_ROOT.parent                    # NEU-DET/

OUT_DIR = PROJECT_ROOT / "data" / "yolo_format"

SPLITS = {
    "train": {
        "images": NEU_DET_ROOT / "train" / "images",
        "annotations": NEU_DET_ROOT / "train" / "annotations",
        "out_name": "train",
    },
    "validation": {
        "images": NEU_DET_ROOT / "validation" / "images",
        "annotations": NEU_DET_ROOT / "validation" / "annotations",
        "out_name": "val",
    },
}

# Class mapping — verified against actual XML annotation <name> tags
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# VOC XML → YOLO conversion
# ---------------------------------------------------------------------------
def parse_voc_xml(xml_path: Path, img_w: int, img_h: int) -> list:
    """
    Parse a single VOC XML annotation and return YOLO-format label lines.

    Each line: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"  WARNING: could not parse {xml_path.name}: {e}")
        return []

    root = tree.getroot()
    lines = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower().replace(" ", "_")

        # Handle naming variants (e.g. "rolled-in scale" → "rolled-in_scale")
        if name not in CLASS_TO_ID:
            name_norm = name.replace("-", "_")
            match = [c for c in CLASSES if c.replace("-", "_") == name_norm]
            if not match:
                print(f"  WARNING: unknown class '{name}' in {xml_path.name}, skipping object")
                continue
            name = match[0]

        cls_id = CLASS_TO_ID[name]
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except ValueError:
            print(f"  WARNING: non-numeric bbox in {xml_path.name}, skipping object")
            continue

        # Validate bbox
        if xmin >= xmax or ymin >= ymax:
            print(f"  WARNING: invalid bbox {[xmin, ymin, xmax, ymax]} in {xml_path.name}, skipping")
            continue

        # Clamp to image boundaries
        xmin = max(0, min(xmin, img_w))
        ymin = max(0, min(ymin, img_h))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))

        # Convert to YOLO normalized center format
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Final sanity check
        if w <= 0 or h <= 0:
            continue

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return lines


def get_image_size_from_xml(xml_path: Path) -> tuple:
    """Extract (width, height) from VOC XML <size> element."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return None, None
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    return w, h


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def process_split(split_name: str, split_info: dict) -> dict:
    """
    Process one split: find images, match annotations, convert, copy.

    Returns a stats dict.
    """
    images_dir = split_info["images"]
    annot_dir = split_info["annotations"]
    out_name = split_info["out_name"]

    img_out = OUT_DIR / "images" / out_name
    lbl_out = OUT_DIR / "labels" / out_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split_name,
        "images_found": 0,
        "annotations_matched": 0,
        "annotations_missing": 0,
        "images_copied": 0,
        "labels_written": 0,
        "objects_total": 0,
        "class_counts": defaultdict(int),
        "skipped_no_annotation": [],
    }

    if not images_dir.exists():
        print(f"  ERROR: {images_dir} does not exist")
        return stats

    # Collect all images from class subdirectories
    image_files = {}  # stem -> full path
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                stem = img_file.stem
                if stem in image_files:
                    print(f"  WARNING: duplicate stem '{stem}' in {class_dir.name}, "
                          f"overwriting with {img_file}")
                image_files[stem] = img_file

    stats["images_found"] = len(image_files)
    print(f"  Found {len(image_files)} images across class subdirectories")

    # Collect all annotations
    annot_files = {}  # stem -> full path
    if annot_dir.exists():
        for xml_file in sorted(annot_dir.glob("*.xml")):
            annot_files[xml_file.stem] = xml_file

    print(f"  Found {len(annot_files)} annotation XML files")

    # Process each image
    for stem, img_path in sorted(image_files.items()):
        xml_path = annot_files.get(stem)

        if xml_path is None:
            stats["annotations_missing"] += 1
            stats["skipped_no_annotation"].append(stem)
            continue

        stats["annotations_matched"] += 1

        # Get image size from XML
        img_w, img_h = get_image_size_from_xml(xml_path)
        if img_w is None or img_w == 0 or img_h == 0:
            print(f"  WARNING: invalid size in {xml_path.name}, skipping")
            continue

        # Convert annotation
        yolo_lines = parse_voc_xml(xml_path, img_w, img_h)

        # Count classes
        for line in yolo_lines:
            cls_id = int(line.split()[0])
            stats["class_counts"][CLASSES[cls_id]] += 1
            stats["objects_total"] += 1

        # Copy image
        dest_img = img_out / img_path.name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
        stats["images_copied"] += 1

        # Write YOLO label
        label_file = lbl_out / f"{stem}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))
        stats["labels_written"] += 1

    return stats


def print_conversion_report(all_stats: list):
    """Print a formatted conversion report."""
    print("\n" + "=" * 60)
    print("  CONVERSION REPORT")
    print("=" * 60)

    for stats in all_stats:
        print(f"\n--- {stats['split'].upper()} -> {stats['split']} ---")
        print(f"  Images found:          {stats['images_found']}")
        print(f"  Annotations matched:   {stats['annotations_matched']}")
        print(f"  Annotations missing:   {stats['annotations_missing']}")
        print(f"  Images copied:         {stats['images_copied']}")
        print(f"  Labels written:        {stats['labels_written']}")
        print(f"  Total bbox objects:    {stats['objects_total']}")

        if stats["skipped_no_annotation"]:
            print(f"  Skipped (no annotation):")
            for s in stats["skipped_no_annotation"][:10]:
                print(f"    - {s}")

        print(f"  Class distribution (by annotation objects):")
        for cls in CLASSES:
            print(f"    {cls}: {stats['class_counts'].get(cls, 0)}")

    print(f"\nOutput directory: {OUT_DIR}")
    print("=" * 60 + "\n")


def main():
    print("NEU-DET -> YOLO Format Converter")
    print(f"Source:  {NEU_DET_ROOT}")
    print(f"Output:  {OUT_DIR}")

    # Clean output directory (start fresh)
    if OUT_DIR.exists():
        print(f"\nRemoving existing output: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    all_stats = []
    for split_name, split_info in SPLITS.items():
        print(f"\nProcessing {split_name}...")
        stats = process_split(split_name, split_info)
        all_stats.append(stats)

    print_conversion_report(all_stats)

    # Verify output
    for out_name in ["train", "val"]:
        img_dir = OUT_DIR / "images" / out_name
        lbl_dir = OUT_DIR / "labels" / out_name
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"Verification — {out_name}: {n_imgs} images, {n_lbls} labels")

    print("\nDone. Dataset is ready for YOLO training.")


if __name__ == "__main__":
    main()

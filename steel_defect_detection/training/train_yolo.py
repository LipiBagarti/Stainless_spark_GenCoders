"""
Fine-tune YOLOv8 (primary detector — Architecture A, see docs/case_study_analysis.md
Part 7) on the NEU-DET dataset converted to YOLO format.

Usage:
    python training/train_yolo.py --model yolov8s.pt --epochs 150

Uses transfer learning from COCO-pretrained weights (mandatory given the small
1,800-image dataset — see Part 9). Augmentation parameters below correspond to
the conservative, defect-aware ranges specified in Part 5 — NOT default aggressive
augmentation, since large distortions can misrepresent real defect appearance.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"


def train(model_name: str, epochs: int, batch: int, imgsz: int, patience: int):
    import yaml
    
    # Load data.yaml to resolve absolute path if needed
    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)
        
    model = YOLO(model_name)  # loads COCO-pretrained weights (transfer learning)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,          # early stopping (Part 9)
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        cos_lr=True,                # cosine LR decay
        # --- Conservative, defect-aware augmentation (Part 5) ---
        fliplr=0.5,                 # horizontal flip
        flipud=0.3,                 # vertical flip (verify against real camera geometry before prod use)
        degrees=12.0,               # small rotation range (±12°)
        translate=0.05,
        scale=0.15,                 # mild scale jitter
        shear=0.0,                  # avoid shear — unrealistic for planar strip imaging
        perspective=0.0,            # explicitly disabled — see Part 5 rationale
        hsv_h=0.0,                  # no hue jitter — grayscale/texture-based defects
        hsv_s=0.0,
        hsv_v=0.3,                  # brightness variation only (simulates lighting inconsistency)
        mosaic=0.3,                 # light mosaic — small dataset, keep conservative
        mixup=0.0,                  # disabled — can blend defect regions unrealistically
        project=str(PROJECT_ROOT / "logs"),
        name="yolov8_neu_det",
        exist_ok=True,
    )

    # Copy best weights to models/ for easy downstream use
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    if best_weights.exists():
        target = models_dir / "best.pt"
        target.write_bytes(best_weights.read_bytes())
        print(f"Best weights copied to {target}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET")
    parser.add_argument("--model", default="yolov8s.pt",
                         help="Base model: yolov8n.pt (fastest) / yolov8s.pt (balanced, default) / yolov8m.pt (max accuracy)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch, args.imgsz, args.patience)

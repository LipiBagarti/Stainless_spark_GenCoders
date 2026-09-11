"""
Training script for RT-DETR (Real-Time DEtection TRansformer) on NEU-DET.

Uses Ultralytics RT-DETR (e.g. rtdetr-l.pt or rtdetr-r18.pt) for state-of-the-art
transformer-based real-time object detection on steel surface defects.
"""

import os
import sys
from pathlib import Path
import yaml
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def train_rtdetr(
    config_path: str = "configs/config.yaml",
    data_yaml_path: str = "configs/data.yaml",
    model_name: str = "rtdetr-l.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 4,
):
    print(f"Loading RT-DETR model: {model_name}")
    model = RTDETR(model_name)

    data_file = ROOT / data_yaml_path
    if not data_file.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_file}")

    output_dir = ROOT / "models" / "weights" / "rtdetr_train"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting RT-DETR training on dataset: {data_file}")
    results = model.train(
        data=str(data_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name="rtdetr_run",
        exist_ok=True,
    )

    best_pt = output_dir / "rtdetr_run" / "weights" / "best.pt"
    dest_pt = ROOT / "models" / "weights" / "rtdetr_best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, dest_pt)
        print(f"Copied best model weights to {dest_pt}")

    print("RT-DETR training complete.")
    return results


if __name__ == "__main__":
    train_rtdetr()

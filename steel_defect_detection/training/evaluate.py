"""
Evaluate a trained YOLOv8 model on the validation set.

Reports:
    - Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
    - Per-class precision, recall, F1
    - Confusion matrix
    - False positive / false negative analysis
    - Visual predictions on sample images

Also runs a confidence threshold sweep to find optimal thresholds.

Usage:
    python training/evaluate.py --weights models/best.pt
    python training/evaluate.py --weights models/best.pt --sweep
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def evaluate_model(weights_path: str, imgsz: int = 640):
    """Run full evaluation on the validation set."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights_path)

    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Data config: {DATA_YAML}")
    print(f"Image size: {imgsz}")
    print()

    # Run validation
    results = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        verbose=True,
        save_json=True,
        plots=True,
        project=str(OUTPUT_DIR),
        name="val_results",
        exist_ok=True,
    )

    # Extract metrics
    class_names = model.names
    nc = len(class_names)

    print("\n" + "=" * 60)
    print("  OVERALL METRICS")
    print("=" * 60)

    # Access metrics from results
    metrics = results

    # Overall metrics
    map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0

    # Per-class metrics
    if hasattr(metrics.box, 'p') and hasattr(metrics.box, 'r'):
        precisions = metrics.box.p
        recalls = metrics.box.r
    else:
        precisions = np.zeros(nc)
        recalls = np.zeros(nc)

    # Compute F1 per class
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )

    overall_p = float(np.mean(precisions))
    overall_r = float(np.mean(recalls))
    overall_f1 = float(np.mean(f1_scores))

    print(f"\n  mAP@0.5:       {map50:.4f}")
    print(f"  mAP@0.5:0.95:  {map50_95:.4f}")
    print(f"  Precision:      {overall_p:.4f}")
    print(f"  Recall:         {overall_r:.4f}")
    print(f"  F1:             {overall_f1:.4f}")

    print("\n" + "-" * 60)
    print("  PER-CLASS METRICS")
    print("-" * 60)
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for i in range(nc):
        cls_name = class_names[i]
        p = float(precisions[i]) if i < len(precisions) else 0.0
        r = float(recalls[i]) if i < len(recalls) else 0.0
        f1 = float(f1_scores[i]) if i < len(f1_scores) else 0.0
        print(f"  {cls_name:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")

    # Save metrics to JSON
    metrics_dict = {
        "overall": {
            "mAP_0.5": map50,
            "mAP_0.5_0.95": map50_95,
            "precision": overall_p,
            "recall": overall_r,
            "f1": overall_f1,
        },
        "per_class": {
            class_names[i]: {
                "precision": float(precisions[i]) if i < len(precisions) else 0.0,
                "recall": float(recalls[i]) if i < len(recalls) else 0.0,
                "f1": float(f1_scores[i]) if i < len(f1_scores) else 0.0,
            }
            for i in range(nc)
        },
    }

    metrics_file = OUTPUT_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    # Generate visual predictions on sample images
    generate_visual_predictions(model, imgsz)

    return metrics_dict


def generate_visual_predictions(model, imgsz: int, max_samples: int = 20):
    """Generate annotated images on validation samples."""
    import yaml

    vis_dir = OUTPUT_DIR / "visual_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load data.yaml to find validation images
    with open(DATA_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Resolve val images path
    data_path = Path(data_cfg.get("path", ""))
    if not data_path.is_absolute():
        data_path = DATA_YAML.parent / data_path

    val_images_dir = data_path / data_cfg.get("val", "images/val")
    if not val_images_dir.exists():
        print(f"  WARNING: Validation images not found at {val_images_dir}")
        return

    image_files = sorted(val_images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    # Sample evenly
    step = max(1, len(image_files) // max_samples)
    samples = image_files[::step][:max_samples]

    print(f"\nGenerating {len(samples)} visual predictions...")

    for img_path in samples:
        results = model.predict(str(img_path), imgsz=imgsz, verbose=False)
        if results:
            annotated = results[0].plot()
            save_path = vis_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(save_path), annotated)

    print(f"Visual predictions saved to {vis_dir}")


def confidence_sweep(weights_path: str, imgsz: int = 640):
    """
    Sweep confidence thresholds and report precision/recall/F1 at each level.

    Helps select the optimal threshold per the industrial trade-off.
    """
    model = YOLO(weights_path)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    print("\n" + "=" * 60)
    print("  CONFIDENCE THRESHOLD SWEEP")
    print("=" * 60)
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP@0.5':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    sweep_results = []

    for conf in thresholds:
        results = model.val(
            data=str(DATA_YAML),
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            plots=False,
        )

        metrics = results
        p = float(np.mean(metrics.box.p)) if hasattr(metrics.box, 'p') else 0.0
        r = float(np.mean(metrics.box.r)) if hasattr(metrics.box, 'r') else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0

        print(f"  {conf:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {map50:>10.4f}")

        sweep_results.append({
            "threshold": conf,
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP_0.5": map50,
        })

    # Find best F1
    best = max(sweep_results, key=lambda x: x["f1"])
    print(f"\n  Best F1 threshold: {best['threshold']:.2f} "
          f"(P={best['precision']:.4f}, R={best['recall']:.4f}, F1={best['f1']:.4f})")

    # Save sweep results
    sweep_dir = OUTPUT_DIR / "confidence_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_file = sweep_dir / "sweep_results.json"
    with open(sweep_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Sweep results saved to {sweep_file}")

    # Industrial recommendation
    print("\n  Industrial threshold recommendation:")
    print("  - For SAFETY-CRITICAL (minimize false negatives): use lower threshold")
    print("    → Higher recall, more false alarms, but fewer missed defects")
    print("  - For FALSE-ALARM reduction (minimize false positives): use higher threshold")
    print("    → Higher precision, fewer false alarms, but may miss subtle defects")
    print("  - Best F1 balances both, but the final choice depends on the")
    print("    cost ratio of false positives vs false negatives in production.")

    return sweep_results


def benchmark_models(model_names: list, imgsz: int = 640):
    """
    Benchmark multiple YOLO variants for model selection.

    Compare: precision, recall, F1, mAP, inference latency, model size.
    """
    print("\n" + "=" * 60)
    print("  MODEL BENCHMARKING")
    print("=" * 60)

    results_all = []

    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        model = YOLO(model_name)

        # Get model size
        model_path = Path(model_name)
        model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0

        # Validate
        val_results = model.val(
            data=str(DATA_YAML),
            imgsz=imgsz,
            verbose=False,
            plots=False,
        )

        metrics = val_results
        p = float(np.mean(metrics.box.p)) if hasattr(metrics.box, 'p') else 0.0
        r = float(np.mean(metrics.box.r)) if hasattr(metrics.box, 'r') else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
        map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0

        # Inference speed (from val results)
        speed = val_results.speed if hasattr(val_results, 'speed') else {}

        results_all.append({
            "model": model_name,
            "precision": p,
            "recall": r,
            "f1": f1,
            "mAP_0.5": map50,
            "mAP_0.5_0.95": map50_95,
            "model_size_mb": model_size_mb,
            "speed": speed,
        })

    # Print comparison table
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    print(f"  {'Model':<15} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP50-95':>9} {'Size(MB)':>9}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*9}")

    for r in results_all:
        print(f"  {r['model']:<15} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['mAP_0.5']:>7.3f} {r['mAP_0.5_0.95']:>9.3f} "
              f"{r['model_size_mb']:>9.1f}")

    # Save benchmark results
    bench_file = OUTPUT_DIR / "benchmark_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(bench_file, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n  Benchmark results saved to {bench_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 on NEU-DET validation")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--sweep", action="store_true", help="Run confidence threshold sweep")
    parser.add_argument("--benchmark", nargs="+", help="Benchmark multiple model weights (e.g. --benchmark model_s.pt model_m.pt)")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_models(args.benchmark, args.imgsz)
    elif args.sweep:
        confidence_sweep(args.weights, args.imgsz)
    else:
        evaluate_model(args.weights, args.imgsz)

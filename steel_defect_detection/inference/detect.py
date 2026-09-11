"""
Run the trained YOLOv8 detector on a single image and pass raw detections through
the decision engine (postprocessing/decision_engine.py).

Usage:
    python inference/detect.py --image path/to/image.jpg --weights models/best.pt
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.preprocess import preprocess_image, load_config
from postprocessing.decision_engine import DecisionEngine

from ultralytics import YOLO


def run_inference(image_path: str, weights_path: str, save_path: str = None):
    config = load_config()
    model = YOLO(weights_path)
    engine = DecisionEngine(config)

    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    processed = preprocess_image(original.copy(), config)
    results = model.predict(processed, verbose=False)[0]

    class_names = model.names
    raw_detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        raw_detections.append({
            "class_name": class_names[cls_id],
            "confidence": conf,
            "box": xyxy,  # [x1, y1, x2, y2]
        })

    # Apply confidence / size / ROI filters + severity mapping
    # (temporal confirmation is applied by the caller across a frame sequence —
    #  see postprocessing/decision_engine.py DecisionEngine.confirm_temporal)
    final_detections = engine.process_single_frame(raw_detections, processed.shape)

    annotated = draw_detections(processed.copy(), final_detections)
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Annotated image saved to {save_path}")

    return final_detections, annotated


def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['class_name']} {det['confidence']:.2f} [{det['severity']}]"
        color = {"low": (0, 200, 0), "medium": (0, 165, 255), "high": (0, 0, 255), "critical": (0, 0, 139)}.get(
            det["severity"], (255, 255, 255)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defect detection on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default="models/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--save", default=None, help="Path to save annotated output")
    args = parser.parse_args()

    save_path = args.save
    if not save_path:
        out_dir = Path("outputs/inference")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_name = Path(args.image).name
        save_path = str(out_dir / f"pred_{img_name}")

    detections, _ = run_inference(args.image, args.weights, save_path)

    print(f"\nDetected {len(detections)} confirmed defect(s):")
    for d in detections:
        print(f"  - {d['class_name']}: confidence={d['confidence']*100:.1f}%, "
              f"severity={d['severity']}, action={d['action']}")

"""
Export the trained YOLOv8 model to ONNX for optimized deployment (Part 15).

Usage:
    python deployment/export_onnx.py --weights models/best.pt

TensorRT conversion (NVIDIA edge/GPU deployment) is a separate step done on the
TARGET device (Jetson or industrial GPU PC), since TensorRT engines are
hardware/driver-specific and not portable across machines:

    # On the target NVIDIA device, after installing TensorRT:
    trtexec --onnx=models/best.onnx --saveEngine=models/best.engine --fp16
    # Use --int8 instead of --fp16 only if you have calibration data and have
    # validated the accuracy drop is acceptable (see docs Part 15).

Required throughput (FPS) must be computed from real strip speed / camera FOV /
frame overlap BEFORE deciding which precision (fp16/int8) and model size
(n/s/m) to deploy — see docs/case_study_analysis.md Part 15 for the formula.
Do not assume a target FPS without that calculation and on-device benchmarking.
"""

import argparse

from ultralytics import YOLO


def export(weights_path: str, imgsz: int, half: bool):
    model = YOLO(weights_path)
    export_path = model.export(format="onnx", imgsz=imgsz, half=half, simplify=True)
    print(f"ONNX model exported to: {export_path}")
    print("Next step: convert to TensorRT on the TARGET deployment device (see script docstring).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained YOLO model to ONNX")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Export in FP16 precision")
    args = parser.parse_args()

    export(args.weights, args.imgsz, args.half)

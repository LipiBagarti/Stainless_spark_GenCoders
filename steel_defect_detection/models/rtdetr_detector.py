"""
RT-DETR detector wrapper — converts Ultralytics RT-DETR output to Prediction schema.

RT-DETR is available via the Ultralytics library, so it uses the same API
surface as YOLO. This allows it to train on the same data.yaml and produce
compatible output.

Usage:
    detector = RTDETRDetector("models/rtdetr_best.pt")
    predictions = detector.predict(image)
"""

import logging
from pathlib import Path

import numpy as np

from models.prediction_schema import Prediction, CLASSES

logger = logging.getLogger(__name__)


class RTDETRDetector:
    """Wraps Ultralytics RT-DETR to produce standardized Prediction objects."""

    MODEL_NAME = "rtdetr"

    def __init__(self, weights_path: str, device: str = "auto"):
        """
        Args:
            weights_path: Path to trained RT-DETR weights (.pt file)
            device: "cuda", "cpu", or "auto"
        """
        from ultralytics import RTDETR

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"RT-DETR weights not found: {self.weights_path}")

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = RTDETR(str(self.weights_path))
        logger.info(f"RT-DETR loaded from {self.weights_path} on {self.device}")

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25,
                iou_threshold: float = 0.45) -> list[Prediction]:
        """
        Run detection on a single image.

        Args:
            image: BGR image (np.ndarray)
            conf_threshold: Minimum confidence
            iou_threshold: NMS IoU threshold

        Returns:
            List of Prediction objects
        """
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        predictions = []
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()

                model_cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                predictions.append(Prediction(
                    bbox=xyxy,
                    class_id=cls_id,
                    class_name=model_cls_name,
                    confidence=conf,
                    model_name=self.MODEL_NAME,
                ))

        return predictions

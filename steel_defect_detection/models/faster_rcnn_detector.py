"""
Faster R-CNN detector wrapper — converts torchvision output to Prediction schema.

Usage:
    detector = FasterRCNNDetector("models/faster_rcnn_best.pt", num_classes=7)
    predictions = detector.predict(image)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.prediction_schema import Prediction, CLASSES

logger = logging.getLogger(__name__)


class FasterRCNNDetector:
    """Wraps torchvision Faster R-CNN to produce standardized Prediction objects."""

    MODEL_NAME = "faster_rcnn"

    def __init__(self, weights_path: str, num_classes: int = 7, device: str = "auto"):
        """
        Args:
            weights_path: Path to trained weights (.pt file with model_state_dict)
            num_classes: Number of classes (6 defects + 1 background = 7)
            device: "cuda", "cpu", or "auto"
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Faster R-CNN weights not found: {self.weights_path}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Build model architecture
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load trained weights
        checkpoint = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Faster R-CNN loaded from {self.weights_path} on {self.device}")

    @torch.no_grad()
    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[Prediction]:
        """
        Run detection on a single BGR image.

        Args:
            image: BGR image (np.ndarray, HxWxC)
            conf_threshold: Minimum confidence to keep

        Returns:
            List of Prediction objects
        """
        # Convert BGR -> RGB -> float tensor [0,1] -> CHW
        rgb = image[:, :, ::-1].copy()  # BGR to RGB
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        outputs = self.model([tensor])
        result = outputs[0]

        predictions = []
        boxes = result["boxes"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        scores = result["scores"].cpu().numpy()

        for bbox, label, score in zip(boxes, labels, scores):
            if score < conf_threshold:
                continue

            # Faster R-CNN label 0 = background, 1-6 = defect classes
            cls_id = int(label) - 1  # shift to 0-indexed
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue

            predictions.append(Prediction(
                bbox=bbox.tolist(),
                class_id=cls_id,
                class_name=CLASSES[cls_id],
                confidence=float(score),
                model_name=self.MODEL_NAME,
            ))

        return predictions

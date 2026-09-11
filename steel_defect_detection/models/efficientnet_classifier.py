"""
EfficientNet-B0 classifier wrapper — second-stage fine-grained classification.

Receives a cropped defect region (from an already-detected bounding box) and
returns a 6-class probability distribution.

Usage:
    classifier = EfficientNetClassifier("models/efficientnet_b0_classifier.pt")
    result = classifier.classify(crop_image)
    print(result.class_name, result.confidence)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0

from models.prediction_schema import ClassificationResult, CLASSES

logger = logging.getLogger(__name__)


class EfficientNetClassifier:
    """Wraps EfficientNet-B0 for defect crop classification."""

    def __init__(self, weights_path: str, input_size: int = 224, device: str = "auto"):
        """
        Args:
            weights_path: Path to trained classifier checkpoint
            input_size: Input image size (default 224 for EfficientNet-B0)
            device: "cuda", "cpu", or "auto"
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {self.weights_path}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)

        # Determine class list from checkpoint (or use default)
        self.classes = checkpoint.get("classes", CLASSES)
        num_classes = len(self.classes)

        # Build model
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Load weights
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform (must match training transforms)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"EfficientNet-B0 classifier loaded from {self.weights_path} on {self.device}")

    @torch.no_grad()
    def classify(self, crop_image: np.ndarray) -> ClassificationResult:
        """
        Classify a cropped defect region.

        Args:
            crop_image: BGR crop (np.ndarray, HxWxC)

        Returns:
            ClassificationResult with class probabilities
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Build result
        best_idx = int(np.argmax(probs))
        class_probabilities = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }

        return ClassificationResult(
            class_id=best_idx,
            class_name=self.classes[best_idx],
            confidence=float(probs[best_idx]),
            class_probabilities=class_probabilities,
        )

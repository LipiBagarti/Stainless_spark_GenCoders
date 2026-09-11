"""
Common prediction data structures shared by all detector and classifier wrappers.

Every detector (YOLO, Faster R-CNN, RT-DETR) must convert its raw output into
a list of Prediction objects. This ensures the ensemble and WBF modules receive
a uniform interface regardless of the underlying model.
"""

from dataclasses import dataclass, field


CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}


@dataclass
class Prediction:
    """A single detection from one model."""
    bbox: list          # [x1, y1, x2, y2] absolute pixel coordinates
    class_id: int
    class_name: str
    confidence: float
    model_name: str     # "yolo", "faster_rcnn", "rtdetr"

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }


@dataclass
class ClassificationResult:
    """Output from the EfficientNet-B0 second-stage classifier."""
    class_id: int
    class_name: str
    confidence: float
    class_probabilities: dict   # {class_name: probability}

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
        }


@dataclass
class FusedDetection:
    """A detection after Weighted Boxes Fusion + EfficientNet classification."""
    bbox: list                      # [x1, y1, x2, y2] absolute pixel coords
    class_id: int
    class_name: str
    detector_confidence: float      # from WBF-fused detector ensemble
    classifier_confidence: float    # from EfficientNet-B0
    final_confidence: float         # weighted combination of both
    severity: str = ""
    action: str = ""
    model_agreement: int = 0        # how many detectors contributed to this box
    contributing_models: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detector_confidence": round(self.detector_confidence, 4),
            "classifier_confidence": round(self.classifier_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "severity": self.severity,
            "action": self.action,
            "model_agreement": self.model_agreement,
            "contributing_models": self.contributing_models,
        }

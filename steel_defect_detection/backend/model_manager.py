"""
Singleton Model Manager for FastAPI backend.

Preloads and caches all models in GPU/CPU memory once on application startup.
Provides thread-safe access to YOLO, Faster R-CNN, RT-DETR, and EfficientNet-B0.
"""

import os
import logging
from pathlib import Path
import yaml
import torch

from models.yolo_detector import YOLODetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.rtdetr_detector import RTDETRDetector
from models.efficientnet_classifier import EfficientNetClassifier

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model cache holding all 4 model instances."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: dict = None):
        if self._initialized:
            return

        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo: YOLODetector = None
        self.faster_rcnn: FasterRCNNDetector = None
        self.rtdetr: RTDETRDetector = None
        self.classifier: EfficientNetClassifier = None
        self._initialized = True

    def load_all_models(self, config: dict = None):
        """Load all enabled models based on config."""
        if config:
            self.config = config

        model_cfg = self.config.get("model", {})
        device = model_cfg.get("device", self.device)
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        logger.info(f"Loading models onto device: {self.device}")

        # 1. YOLO
        yolo_path = model_cfg.get("weights_path", "models/weights/best.pt")
        try:
            self.yolo = YOLODetector(weights_path=yolo_path, device=self.device)
            logger.info("YOLODetector initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load YOLO model from {yolo_path}: {e}")
            self.yolo = None

        # 2. Faster R-CNN
        frcnn_cfg = self.config.get("faster_rcnn", {})
        frcnn_path = frcnn_cfg.get("weights_path", "models/weights/faster_rcnn_best.pth")
        try:
            self.faster_rcnn = FasterRCNNDetector(weights_path=frcnn_path, device=self.device)
            logger.info("FasterRCNNDetector initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load Faster R-CNN model from {frcnn_path}: {e}")
            self.faster_rcnn = None

        # 3. RT-DETR
        rtdetr_cfg = self.config.get("rtdetr", {})
        rtdetr_path = rtdetr_cfg.get("weights_path", "models/weights/rtdetr_best.pt")
        try:
            self.rtdetr = RTDETRDetector(weights_path=rtdetr_path, device=self.device)
            logger.info("RTDETRDetector initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load RT-DETR model from {rtdetr_path}: {e}")
            self.rtdetr = None

        # 4. EfficientNet-B0 Classifier
        cls_cfg = self.config.get("classifier", {})
        if cls_cfg.get("enabled", True):
            cls_path = cls_cfg.get("weights_path", "models/weights/efficientnet_b0_best.pth")
            try:
                self.classifier = EfficientNetClassifier(weights_path=cls_path, device=self.device)
                logger.info("EfficientNetClassifier initialized successfully")
            except Exception as e:
                logger.warning(f"Could not load EfficientNet classifier from {cls_path}: {e}")
                self.classifier = None

    def get_status(self) -> dict:
        """Return status of all loaded models."""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "models": {
                "yolo": self.yolo is not None,
                "faster_rcnn": self.faster_rcnn is not None,
                "rtdetr": self.rtdetr is not None,
                "classifier": self.classifier is not None,
            },
        }

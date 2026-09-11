"""
Ensemble detector orchestrator — runs all 3 detectors, applies WBF,
then runs EfficientNet-B0 classification on each fused crop.

This is the single entry point for the full inference pipeline.

Usage:
    ensemble = EnsembleDetector(config, model_manager)
    results = ensemble.run(image)
"""

import logging

import cv2
import numpy as np

from ensemble.wbf import weighted_boxes_fusion
from ensemble.confidence_fusion import fuse_confidences
from models.prediction_schema import FusedDetection, CLASSES

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Orchestrates the four-model ensemble pipeline:
        1. Run YOLO, Faster R-CNN, RT-DETR
        2. WBF fusion
        3. Crop defect regions
        4. EfficientNet-B0 classification
        5. Confidence fusion
    """

    def __init__(self, config: dict, model_manager):
        """
        Args:
            config: Parsed config.yaml dict
            model_manager: ModelManager instance with loaded models
        """
        self.config = config
        self.mm = model_manager

        # Ensemble config
        ens_cfg = config.get("ensemble", {})
        self.model_weights = [
            ens_cfg.get("yolo_weight", 1.0),
            ens_cfg.get("faster_rcnn_weight", 1.0),
            ens_cfg.get("rtdetr_weight", 1.0),
        ]
        self.wbf_iou = ens_cfg.get("iou_threshold", 0.55)
        self.wbf_skip = ens_cfg.get("skip_box_threshold", 0.01)

        # Confidence fusion config
        cf_cfg = config.get("confidence_fusion", {})
        self.det_weight = cf_cfg.get("detector_weight", 0.6)
        self.cls_weight = cf_cfg.get("classifier_weight", 0.4)

        # Classifier config
        cls_cfg = config.get("classifier", {})
        self.classifier_enabled = cls_cfg.get("enabled", True)
        self.crop_padding = cls_cfg.get("crop_padding_fraction", 0.1)

        # Model config
        model_cfg = config.get("model", {})
        self.conf_threshold = model_cfg.get("confidence_threshold", 0.35)
        self.iou_threshold = model_cfg.get("iou_threshold", 0.45)

        # Inference mode
        self.mode = config.get("inference", {}).get("mode", "ensemble")

    def run(self, image: np.ndarray) -> list[FusedDetection]:
        """
        Run the full ensemble pipeline on a single image.

        Args:
            image: BGR image (np.ndarray, HxWxC)

        Returns:
            List of FusedDetection objects with all confidence scores
        """
        h, w = image.shape[:2]

        if self.mode == "yolo_only":
            return self._run_yolo_only(image, w, h)

        return self._run_ensemble(image, w, h)

    def _run_ensemble(self, image: np.ndarray, w: int, h: int) -> list[FusedDetection]:
        """Full 4-model ensemble pipeline."""
        # Step 1: Run all available detectors
        predictions_per_model = []
        active_weights = []

        # YOLO
        if self.mm.yolo is not None:
            try:
                yolo_preds = self.mm.yolo.predict(image, self.conf_threshold, self.iou_threshold)
                predictions_per_model.append(yolo_preds)
                active_weights.append(self.model_weights[0])
                logger.debug(f"YOLO: {len(yolo_preds)} detections")
            except Exception as e:
                logger.error(f"YOLO inference failed: {e}")

        # Faster R-CNN
        if self.mm.faster_rcnn is not None:
            try:
                frcnn_preds = self.mm.faster_rcnn.predict(image, self.conf_threshold)
                predictions_per_model.append(frcnn_preds)
                active_weights.append(self.model_weights[1])
                logger.debug(f"Faster R-CNN: {len(frcnn_preds)} detections")
            except Exception as e:
                logger.error(f"Faster R-CNN inference failed: {e}")

        # RT-DETR
        if self.mm.rtdetr is not None:
            try:
                rtdetr_preds = self.mm.rtdetr.predict(image, self.conf_threshold, self.iou_threshold)
                predictions_per_model.append(rtdetr_preds)
                active_weights.append(self.model_weights[2])
                logger.debug(f"RT-DETR: {len(rtdetr_preds)} detections")
            except Exception as e:
                logger.error(f"RT-DETR inference failed: {e}")

        if not predictions_per_model:
            logger.warning("No detectors produced results")
            return []

        # Step 2: WBF fusion
        fused = weighted_boxes_fusion(
            predictions_per_model=predictions_per_model,
            model_weights=active_weights,
            iou_threshold=self.wbf_iou,
            skip_box_threshold=self.wbf_skip,
            image_size=(w, h),
        )

        if not fused:
            return []

        # Step 3: EfficientNet classification on each fused crop
        if self.classifier_enabled and self.mm.classifier is not None:
            for det in fused:
                crop = self._crop_region(image, det.bbox)
                if crop is not None and crop.size > 0:
                    try:
                        cls_result = self.mm.classifier.classify(crop)
                        det.classifier_confidence = cls_result.confidence
                        # Update class based on classifier if confident
                        det.class_id = cls_result.class_id
                        det.class_name = cls_result.class_name
                        # Fuse confidences
                        det.final_confidence = fuse_confidences(
                            det.detector_confidence,
                            cls_result.confidence,
                            self.det_weight,
                            self.cls_weight,
                        )
                    except Exception as e:
                        logger.error(f"Classifier failed for crop: {e}")
                        det.classifier_confidence = 0.0
                        det.final_confidence = det.detector_confidence
        else:
            for det in fused:
                det.final_confidence = det.detector_confidence

        return fused

    def _run_yolo_only(self, image: np.ndarray, w: int, h: int) -> list[FusedDetection]:
        """Fallback: YOLO-only mode (single model, no WBF)."""
        if self.mm.yolo is None:
            logger.error("YOLO model not loaded, cannot run yolo_only mode")
            return []

        preds = self.mm.yolo.predict(image, self.conf_threshold, self.iou_threshold)

        results = []
        for p in preds:
            det = FusedDetection(
                bbox=p.bbox,
                class_id=p.class_id,
                class_name=p.class_name,
                detector_confidence=p.confidence,
                classifier_confidence=0.0,
                final_confidence=p.confidence,
                model_agreement=1,
                contributing_models=[p.model_name],
            )

            # Run classifier if available
            if self.classifier_enabled and self.mm.classifier is not None:
                crop = self._crop_region(image, p.bbox)
                if crop is not None and crop.size > 0:
                    try:
                        cls_result = self.mm.classifier.classify(crop)
                        det.classifier_confidence = cls_result.confidence
                        det.class_id = cls_result.class_id
                        det.class_name = cls_result.class_name
                        det.final_confidence = fuse_confidences(
                            det.detector_confidence,
                            cls_result.confidence,
                            self.det_weight,
                            self.cls_weight,
                        )
                    except Exception as e:
                        logger.error(f"Classifier failed: {e}")

            results.append(det)

        return results

    def _crop_region(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """Crop a region from the image with configurable padding."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)

        x1_pad = max(0, int(x1) - pad_x)
        y1_pad = max(0, int(y1) - pad_y)
        x2_pad = min(w, int(x2) + pad_x)
        y2_pad = min(h, int(y2) + pad_y)

        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        return crop if crop.size > 0 else None

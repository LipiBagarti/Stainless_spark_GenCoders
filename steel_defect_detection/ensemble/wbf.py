"""
Weighted Boxes Fusion (WBF) for combining detections from multiple models.

Uses the ensemble_boxes library for the core WBF algorithm. Wraps it with
our Prediction schema and adds model agreement tracking.

Reference: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
"""

import logging
from collections import defaultdict

import numpy as np

from models.prediction_schema import Prediction, FusedDetection, CLASSES

logger = logging.getLogger(__name__)


def weighted_boxes_fusion(
    predictions_per_model: list[list[Prediction]],
    model_weights: list[float],
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.01,
    image_size: tuple[int, int] = (640, 640),
) -> list[FusedDetection]:
    """
    Apply Weighted Boxes Fusion to merge detections from multiple models.

    Args:
        predictions_per_model: List of prediction lists, one per model.
            predictions_per_model[0] = YOLO predictions
            predictions_per_model[1] = Faster R-CNN predictions
            predictions_per_model[2] = RT-DETR predictions
        model_weights: Weight for each model (same order as predictions_per_model)
        iou_threshold: IoU threshold for merging overlapping boxes
        skip_box_threshold: Minimum confidence to consider a box
        image_size: (width, height) for normalization

    Returns:
        List of FusedDetection objects with model agreement tracking
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion as _wbf
    except ImportError:
        logger.warning("ensemble_boxes not installed. Using fallback simple NMS merge.")
        return _fallback_merge(predictions_per_model, model_weights, iou_threshold)

    w, h = image_size
    n_models = len(predictions_per_model)

    # Convert to ensemble_boxes format:
    # boxes_list: list of np.array (N, 4) — normalized [0, 1]
    # scores_list: list of np.array (N,)
    # labels_list: list of np.array (N,) — integer class IDs
    boxes_list = []
    scores_list = []
    labels_list = []
    model_names_per_box = []  # track which model produced each box

    for model_idx, preds in enumerate(predictions_per_model):
        boxes = []
        scores = []
        labels = []
        names = []

        for p in preds:
            if p.confidence < skip_box_threshold:
                continue
            # Normalize to [0, 1]
            x1 = max(0.0, min(p.bbox[0] / w, 1.0))
            y1 = max(0.0, min(p.bbox[1] / h, 1.0))
            x2 = max(0.0, min(p.bbox[2] / w, 1.0))
            y2 = max(0.0, min(p.bbox[3] / h, 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(p.confidence)
            labels.append(p.class_id)
            names.append(p.model_name)

        boxes_list.append(np.array(boxes) if boxes else np.empty((0, 4)))
        scores_list.append(np.array(scores) if scores else np.empty(0))
        labels_list.append(np.array(labels, dtype=int) if labels else np.empty(0, dtype=int))
        model_names_per_box.append(names)

    # Run WBF
    if all(len(b) == 0 for b in boxes_list):
        return []

    fused_boxes, fused_scores, fused_labels = _wbf(
        boxes_list,
        scores_list,
        labels_list,
        weights=model_weights,
        iou_thr=iou_threshold,
        skip_box_thr=skip_box_threshold,
    )

    # Build FusedDetection objects and compute model agreement
    fused_detections = []
    for i in range(len(fused_boxes)):
        box_norm = fused_boxes[i]
        score = float(fused_scores[i])
        cls_id = int(fused_labels[i])

        # Denormalize back to pixel coordinates
        bbox = [
            float(box_norm[0] * w),
            float(box_norm[1] * h),
            float(box_norm[2] * w),
            float(box_norm[3] * h),
        ]

        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"

        # Determine model agreement: count how many models had a matching detection
        contributing = _find_contributing_models(
            bbox, cls_id, predictions_per_model, iou_threshold, image_size
        )

        fused_detections.append(FusedDetection(
            bbox=bbox,
            class_id=cls_id,
            class_name=cls_name,
            detector_confidence=score,
            classifier_confidence=0.0,  # filled later by ensemble_detector
            final_confidence=score,     # updated after classification
            model_agreement=len(contributing),
            contributing_models=contributing,
        ))

    return fused_detections


def _find_contributing_models(
    fused_bbox: list,
    fused_cls_id: int,
    predictions_per_model: list[list[Prediction]],
    iou_threshold: float,
    image_size: tuple,
) -> list[str]:
    """Find which models contributed detections that overlap with a fused box."""
    contributing = []

    for preds in predictions_per_model:
        for p in preds:
            if p.class_id != fused_cls_id:
                continue
            iou = _compute_iou(fused_bbox, p.bbox)
            if iou >= iou_threshold * 0.5:  # use relaxed threshold for attribution
                if p.model_name not in contributing:
                    contributing.append(p.model_name)
                break

    return contributing


def _compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _fallback_merge(
    predictions_per_model: list[list[Prediction]],
    model_weights: list[float],
    iou_threshold: float,
) -> list[FusedDetection]:
    """
    Fallback when ensemble_boxes is not installed.
    Simple confidence-weighted merge with NMS-like dedup.
    """
    all_preds = []
    for preds in predictions_per_model:
        all_preds.extend(preds)

    # Sort by confidence descending
    all_preds.sort(key=lambda p: p.confidence, reverse=True)

    kept = []
    for pred in all_preds:
        # Check if this overlaps with an already-kept detection of the same class
        is_dup = False
        for k in kept:
            if k.class_id == pred.class_id:
                iou = _compute_iou(k.bbox, pred.bbox)
                if iou >= iou_threshold:
                    is_dup = True
                    # Track additional model contribution
                    if pred.model_name not in k.contributing_models:
                        k.contributing_models.append(pred.model_name)
                        k.model_agreement += 1
                    break

        if not is_dup:
            kept.append(FusedDetection(
                bbox=pred.bbox,
                class_id=pred.class_id,
                class_name=pred.class_name,
                detector_confidence=pred.confidence,
                classifier_confidence=0.0,
                final_confidence=pred.confidence,
                model_agreement=1,
                contributing_models=[pred.model_name],
            ))

    return kept

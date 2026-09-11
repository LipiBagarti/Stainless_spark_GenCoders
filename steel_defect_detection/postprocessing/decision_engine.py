"""
Decision engine implementing the false-alarm-reduction staged pipeline:

    Ensemble detection -> confidence filter -> size filter -> ROI filter
    -> temporal consistency -> severity classification -> final alert

Compatible with both FusedDetection objects and legacy detection dicts.
"""

import time
import uuid
from dataclasses import dataclass, field
from models.prediction_schema import FusedDetection


@dataclass
class TrackedDefect:
    """A defect tracked across multiple frames for temporal confirmation."""
    track_id: str
    class_name: str
    box: list
    confidence: float
    frame_count: int = 1
    confirmed: bool = False


class DecisionEngine:
    def __init__(self, config: dict):
        self.config = config
        self.de_cfg = config.get("decision_engine", {})
        self.thresholds = config.get("confidence_thresholds", {})
        self.severity_cfg = config.get("severity_rules", {})
        self.plc_cfg = config.get("plc_integration", {})

        # Active tracks for temporal confirmation across a frame stream
        self._active_tracks: list[TrackedDefect] = []

    def _normalize_detection(self, det) -> dict:
        """Convert FusedDetection object or dict to standard dict."""
        if isinstance(det, FusedDetection):
            return det.to_dict()
        elif hasattr(det, "to_dict"):
            return det.to_dict()
        elif isinstance(det, dict):
            # Normalize key aliases
            box = det.get("bbox", det.get("box", [0, 0, 0, 0]))
            conf = det.get("final_confidence", det.get("confidence", det.get("detector_confidence", 0.0)))
            cls_name = det.get("class_name", det.get("defect_type", "unknown"))
            return {
                **det,
                "bbox": box,
                "box": box,
                "confidence": conf,
                "final_confidence": conf,
                "class_name": cls_name,
            }
        return dict(det)

    # ---------- Stage 1: confidence filter ----------
    def _passes_confidence(self, det: dict) -> bool:
        cls_name = det.get("class_name", "default")
        threshold = self.thresholds.get(cls_name, self.thresholds.get("default", 0.35))
        conf = det.get("final_confidence", det.get("confidence", 0.0))
        return conf >= threshold

    # ---------- Stage 2: size filter ----------
    def _passes_size(self, det: dict) -> bool:
        box = det.get("bbox", det.get("box", [0, 0, 0, 0]))
        x1, y1, x2, y2 = box
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        return area >= self.de_cfg.get("min_defect_area_px", 100)

    # ---------- Stage 3: ROI filter ----------
    def _passes_roi(self, det: dict, frame_shape) -> bool:
        roi = self.de_cfg.get("roi", {"enabled": False, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0})
        if not roi.get("enabled", False):
            return True

        h, w = frame_shape[0], frame_shape[1]
        box = det.get("bbox", det.get("box", [0, 0, 0, 0]))
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (roi.get("x_min", 0.0) * w <= cx <= roi.get("x_max", 1.0) * w) and (
            roi.get("y_min", 0.0) * h <= cy <= roi.get("y_max", 1.0) * h
        )

    # ---------- Stage 4: severity classification ----------
    def _classify_severity(self, det: dict) -> str:
        box = det.get("bbox", det.get("box", [0, 0, 0, 0]))
        x1, y1, x2, y2 = box
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        th = self.severity_cfg.get("area_thresholds_px", {"low": 100, "medium": 2500, "high": 10000})

        if area >= th.get("high", 10000):
            level = "high"
        elif area >= th.get("medium", 2500):
            level = "medium"
        else:
            level = "low"

        # Escalate based on class-level business criticality
        if det.get("class_name") in self.severity_cfg.get("high_severity_classes", []):
            if level == "low":
                level = "medium"
            elif level == "medium":
                level = "high"
            elif level == "high":
                level = "critical"

        return level

    # ---------- Single-frame processing ----------
    def process_single_frame(self, raw_detections: list, frame_shape) -> list:
        """
        Apply confidence -> size -> ROI filters and severity mapping to detections
        from a single frame.
        """
        final = []
        action_map = self.severity_cfg.get("action_map", {
            "low": "log",
            "medium": "operator_alert",
            "high": "mark_reject",
            "critical": "line_stop",
        })

        for raw_det in raw_detections:
            det = self._normalize_detection(raw_det)
            if not self._passes_confidence(det):
                continue
            if not self._passes_size(det):
                continue
            if not self._passes_roi(det, frame_shape):
                continue

            severity = self._classify_severity(det)
            action = action_map.get(severity, "log")

            final.append({
                **det,
                "severity": severity,
                "action": action,
            })
        return final

    # ---------- Stage 5: temporal confirmation (multi-frame stream) ----------
    def confirm_temporal(self, frame_detections: list, frame_shape) -> list:
        """
        Call once per incoming frame in a live/streaming pipeline. Matches new
        detections against active tracks using IoU.
        """
        temporal_cfg = self.de_cfg.get("temporal_confirmation", {})
        if not temporal_cfg.get("enabled", True):
            return self.process_single_frame(frame_detections, frame_shape)

        filtered = self.process_single_frame(frame_detections, frame_shape)
        iou_thresh = temporal_cfg.get("iou_match_threshold", 0.3)
        min_frames = temporal_cfg.get("min_consecutive_frames", 2)

        matched_track_ids = set()
        confirmed_output = []

        for det in filtered:
            box = det.get("bbox", det.get("box", [0, 0, 0, 0]))
            conf = det.get("final_confidence", det.get("confidence", 0.0))
            cls_name = det.get("class_name", "unknown")

            best_match, best_iou = None, 0.0
            for track in self._active_tracks:
                if track.class_name != cls_name:
                    continue
                iou = self._iou(track.box, box)
                if iou > best_iou:
                    best_match, best_iou = track, iou

            if best_match is not None and best_iou >= iou_thresh:
                best_match.frame_count += 1
                best_match.box = box
                best_match.confidence = max(best_match.confidence, conf)
                matched_track_ids.add(best_match.track_id)

                if best_match.frame_count >= min_frames:
                    best_match.confirmed = True
                    confirmed_output.append({
                        **det,
                        "confidence": best_match.confidence,
                        "track_id": best_match.track_id,
                        "frame_count": best_match.frame_count,
                    })
            else:
                new_track = TrackedDefect(
                    track_id=str(uuid.uuid4())[:8],
                    class_name=cls_name,
                    box=box,
                    confidence=conf,
                )
                self._active_tracks.append(new_track)
                matched_track_ids.add(new_track.track_id)

        # Drop tracks not seen in this frame
        self._active_tracks = [t for t in self._active_tracks if t.track_id in matched_track_ids]

        return confirmed_output

    @staticmethod
    def _iou(box_a, box_b) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    # ---------- PLC / database payload ----------
    def build_plc_payload(self, detection: dict, strip_position: str = None) -> dict:
        box = detection.get("bbox", detection.get("box", [0, 0, 0, 0]))
        return {
            "defect_type": detection.get("class_name", detection.get("defect_type", "unknown")),
            "detector_confidence": round(detection.get("detector_confidence", detection.get("confidence", 0.0)), 3),
            "classifier_confidence": round(detection.get("classifier_confidence", 0.0), 3),
            "final_confidence": round(detection.get("final_confidence", detection.get("confidence", 0.0)), 3),
            "severity": detection.get("severity", "low"),
            "model_agreement": detection.get("model_agreement", 1),
            "strip_position": strip_position,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "action": detection.get("action", "log"),
            "box": box,
        }

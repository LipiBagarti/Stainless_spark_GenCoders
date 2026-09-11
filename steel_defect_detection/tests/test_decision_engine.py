"""
Unit tests for the decision engine's filtering, severity, and temporal logic.
Run: python -m pytest tests/test_decision_engine.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from postprocessing.decision_engine import DecisionEngine

SAMPLE_CONFIG = {
    "confidence_thresholds": {"scratches": 0.4, "default": 0.35},
    "decision_engine": {
        "min_defect_area_px": 100,
        "roi": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "temporal_confirmation": {
            "enabled": True,
            "min_consecutive_frames": 2,
            "iou_match_threshold": 0.3,
        },
    },
    "severity_rules": {
        "area_thresholds_px": {"low": 500, "medium": 2000, "high": 5000},
        "high_severity_classes": ["inclusion"],
        "action_map": {"low": "log", "medium": "operator_alert", "high": "mark_reject", "critical": "escalate"},
    },
    "plc_integration": {"protocol": "opc-ua", "payload_fields": []},
}

FRAME_SHAPE = (640, 640, 3)


def test_confidence_filter_rejects_low_confidence():
    engine = DecisionEngine(SAMPLE_CONFIG)
    dets = [{"class_name": "scratches", "confidence": 0.2, "box": [10, 10, 100, 100]}]
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert len(result) == 0


def test_confidence_filter_accepts_above_threshold():
    engine = DecisionEngine(SAMPLE_CONFIG)
    dets = [{"class_name": "scratches", "confidence": 0.9, "box": [10, 10, 100, 100]}]
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert len(result) == 1


def test_size_filter_rejects_tiny_box():
    engine = DecisionEngine(SAMPLE_CONFIG)
    dets = [{"class_name": "scratches", "confidence": 0.9, "box": [10, 10, 12, 12]}]  # area=4
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert len(result) == 0


def test_severity_escalates_for_high_severity_class():
    engine = DecisionEngine(SAMPLE_CONFIG)
    # small box (low area) but class is in high_severity_classes -> escalated to medium
    dets = [{"class_name": "inclusion", "confidence": 0.9, "box": [0, 0, 10, 10]}]
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert result[0]["severity"] == "medium"


def test_severity_large_area_is_high():
    engine = DecisionEngine(SAMPLE_CONFIG)
    dets = [{"class_name": "scratches", "confidence": 0.9, "box": [0, 0, 100, 100]}]  # area=10000
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert result[0]["severity"] == "high"


def test_temporal_confirmation_requires_min_frames():
    engine = DecisionEngine(SAMPLE_CONFIG)
    det = [{"class_name": "scratches", "confidence": 0.9, "box": [10, 10, 100, 100]}]

    # Frame 1: new track created, not yet confirmed
    out1 = engine.confirm_temporal(det, FRAME_SHAPE)
    assert len(out1) == 0

    # Frame 2: same box seen again -> should now be confirmed (min_consecutive_frames=2)
    out2 = engine.confirm_temporal(det, FRAME_SHAPE)
    assert len(out2) == 1


def test_plc_payload_structure():
    engine = DecisionEngine(SAMPLE_CONFIG)
    detection = {
        "class_name": "scratches", "confidence": 0.87, "box": [0, 0, 50, 50],
        "severity": "medium", "action": "operator_alert",
    }
    payload = engine.build_plc_payload(detection, strip_position="12.5m")
    assert payload["defect_type"] == "scratches"
    assert payload["severity"] == "medium"
    assert payload["strip_position"] == "12.5m"
    assert "timestamp" in payload

def test_roi_filter_rejects_outside_box():
    engine = DecisionEngine(SAMPLE_CONFIG)
    # ROI is x: 0-0.5, y: 0-0.5
    engine.de_cfg["roi"] = {"enabled": True, "x_min": 0.0, "x_max": 0.5, "y_min": 0.0, "y_max": 0.5}
    # center is at (400, 400) -> outside ROI for a 640x640 frame
    dets = [{"class_name": "scratches", "confidence": 0.9, "box": [350, 350, 450, 450]}]
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert len(result) == 0


def test_iou_calculation():
    engine = DecisionEngine(SAMPLE_CONFIG)
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    # intersection area is 5*5=25, union is 100+100-25=175 -> IoU = 25/175 ≈ 0.1428
    iou = engine._iou(box1, box2)
    assert round(iou, 4) == 0.1428


def test_clean_image_decision():
    engine = DecisionEngine(SAMPLE_CONFIG)
    result = engine.process_single_frame([], FRAME_SHAPE)
    assert len(result) == 0


def test_defective_image_decision():
    engine = DecisionEngine(SAMPLE_CONFIG)
    dets = [{"class_name": "scratches", "confidence": 0.9, "box": [0, 0, 100, 100]}]
    result = engine.process_single_frame(dets, FRAME_SHAPE)
    assert len(result) == 1
    assert result[0]["severity"] == "high"
    assert result[0]["action"] == "mark_reject"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

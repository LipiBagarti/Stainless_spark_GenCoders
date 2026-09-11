"""
PLC-ready JSON payload builder for industrial integration.

Builds structured payloads suitable for:
- OPC-UA transmission to PLCs
- Database logging
- Dashboard display
- SCADA system integration

Actual PLC communication is DISABLED by default (config: plc_integration.enabled).
"""

import time
from models.prediction_schema import FusedDetection


def build_plc_payload(
    detection: dict,
    strip_position: str = None,
    timestamp: str = None,
) -> dict:
    """
    Build a production-ready JSON payload for a single detection.

    Args:
        detection: Detection dict (from FusedDetection.to_dict() or decision engine output)
        strip_position: Physical position on the strip (runtime value, null if unavailable)
        timestamp: ISO 8601 timestamp (auto-generated if None)

    Returns:
        Structured payload dict
    """
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "defect_type": detection.get("class_name", detection.get("defect_type", "unknown")),
        "detector_confidence": round(detection.get("detector_confidence", 0.0), 4),
        "classifier_confidence": round(detection.get("classifier_confidence", 0.0), 4),
        "final_confidence": round(detection.get("final_confidence", detection.get("confidence", 0.0)), 4),
        "severity": detection.get("severity", "unknown"),
        "model_agreement": detection.get("model_agreement", 0),
        "strip_position": strip_position,
        "timestamp": timestamp,
        "action": detection.get("action", "log"),
    }


def build_inspection_payload(
    detections: list[dict],
    final_decision: str,
    action: str,
    strip_position: str = None,
) -> dict:
    """
    Build a complete inspection payload covering all detections.

    Args:
        detections: List of detection dicts
        final_decision: "CLEAN" or "DEFECTIVE"
        action: Overall action (log, operator_alert, mark_reject, escalate)
        strip_position: Physical strip position

    Returns:
        Complete inspection payload
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "timestamp": timestamp,
        "final_decision": final_decision,
        "action": action,
        "strip_position": strip_position,
        "num_defects": len(detections),
        "detections": [
            build_plc_payload(d, strip_position, timestamp)
            for d in detections
        ],
    }

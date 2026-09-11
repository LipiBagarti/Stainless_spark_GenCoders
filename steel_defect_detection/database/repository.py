"""
Data access layer for inspection results.

Provides CRUD operations over the inspections and detections tables.
"""

import uuid
import time
import logging

from database.database import get_connection

logger = logging.getLogger(__name__)


def save_inspection(
    image_path: str,
    final_decision: str,
    action: str,
    detections: list[dict],
) -> str:
    """
    Save a complete inspection (with all detections) to the database.

    Args:
        image_path: Path or name of the inspected image
        final_decision: "CLEAN" or "DEFECTIVE"
        action: Overall action taken
        detections: List of detection dicts

    Returns:
        inspection_id (UUID string)
    """
    inspection_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO inspections (id, timestamp, image_path, final_decision, action, num_defects) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (inspection_id, timestamp, image_path, final_decision, action, len(detections)),
        )

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cursor.execute(
                "INSERT INTO detections "
                "(inspection_id, defect_type, x1, y1, x2, y2, "
                "detector_confidence, classifier_confidence, final_confidence, "
                "severity, model_agreement, action) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    inspection_id,
                    det.get("class_name", det.get("defect_type", "unknown")),
                    bbox[0] if len(bbox) > 0 else 0,
                    bbox[1] if len(bbox) > 1 else 0,
                    bbox[2] if len(bbox) > 2 else 0,
                    bbox[3] if len(bbox) > 3 else 0,
                    det.get("detector_confidence", 0.0),
                    det.get("classifier_confidence", 0.0),
                    det.get("final_confidence", det.get("confidence", 0.0)),
                    det.get("severity", "unknown"),
                    det.get("model_agreement", 0),
                    det.get("action", "log"),
                ),
            )

        conn.commit()
        logger.info(f"Saved inspection {inspection_id} with {len(detections)} detections")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save inspection: {e}")
        raise
    finally:
        conn.close()

    return inspection_id


def get_inspections(limit: int = 50, offset: int = 0) -> list[dict]:
    """Retrieve recent inspections (paginated)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM inspections ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_inspection_by_id(inspection_id: str) -> dict:
    """Retrieve a single inspection with its detections."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
    inspection = cursor.fetchone()
    if inspection is None:
        conn.close()
        return None

    inspection = dict(inspection)

    cursor.execute("SELECT * FROM detections WHERE inspection_id = ?", (inspection_id,))
    inspection["detections"] = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return inspection


def get_statistics() -> dict:
    """Get aggregate statistics from the database."""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    # Total inspections
    cursor.execute("SELECT COUNT(*) as count FROM inspections")
    stats["total_inspections"] = cursor.fetchone()["count"]

    # By decision
    cursor.execute(
        "SELECT final_decision, COUNT(*) as count FROM inspections GROUP BY final_decision"
    )
    stats["by_decision"] = {row["final_decision"]: row["count"] for row in cursor.fetchall()}

    # By defect type
    cursor.execute(
        "SELECT defect_type, COUNT(*) as count FROM detections GROUP BY defect_type ORDER BY count DESC"
    )
    stats["by_defect_type"] = {row["defect_type"]: row["count"] for row in cursor.fetchall()}

    # By severity
    cursor.execute(
        "SELECT severity, COUNT(*) as count FROM detections GROUP BY severity ORDER BY count DESC"
    )
    stats["by_severity"] = {row["severity"]: row["count"] for row in cursor.fetchall()}

    # Total detections
    cursor.execute("SELECT COUNT(*) as count FROM detections")
    stats["total_detections"] = cursor.fetchone()["count"]

    conn.close()
    return stats

"""
SQLite database connection manager.

For PoC, uses SQLite. Production migration path: swap the connection string
to PostgreSQL without changing the repository layer.
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = None
_SCHEMA_INITIALIZED = False


def init_db(db_path: str):
    """Initialize the database and create tables if they don't exist."""
    global _DB_PATH, _SCHEMA_INITIALIZED

    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            final_decision TEXT NOT NULL,
            action TEXT NOT NULL,
            num_defects INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL,
            detector_confidence REAL,
            classifier_confidence REAL,
            final_confidence REAL,
            severity TEXT,
            model_agreement INTEGER DEFAULT 0,
            action TEXT,
            FOREIGN KEY (inspection_id) REFERENCES inspections(id)
        )
    """)

    conn.commit()
    conn.close()
    _SCHEMA_INITIALIZED = True
    logger.info(f"Database initialized at {_DB_PATH}")


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection. Thread-safe by default."""
    if _DB_PATH is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

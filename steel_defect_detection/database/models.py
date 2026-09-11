"""
Database ORM-like models — plain dicts for SQLite simplicity.

These define the schema shape. For production with PostgreSQL,
replace with SQLAlchemy ORM models.
"""

# Schema documentation — these are the column definitions for reference.
# Actual table creation is in database.py init_db().

INSPECTION_COLUMNS = [
    "id",               # TEXT (UUID) primary key
    "timestamp",        # TEXT (ISO 8601)
    "image_path",       # TEXT
    "final_decision",   # TEXT ("CLEAN" / "DEFECTIVE")
    "action",           # TEXT
    "num_defects",      # INTEGER
]

DETECTION_COLUMNS = [
    "id",                       # INTEGER autoincrement
    "inspection_id",            # TEXT (FK -> inspections.id)
    "defect_type",              # TEXT
    "x1", "y1", "x2", "y2",    # REAL (bounding box)
    "detector_confidence",      # REAL
    "classifier_confidence",    # REAL
    "final_confidence",         # REAL
    "severity",                 # TEXT
    "model_agreement",          # INTEGER
    "action",                   # TEXT
]

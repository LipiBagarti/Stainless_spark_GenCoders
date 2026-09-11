"""
Analytics & History API routes.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from database.repository import get_inspections, get_inspection_by_id, get_statistics

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get("/statistics")
async def get_summary_statistics():
    """Retrieve historical defect inspection stats, counts by defect type, and severity distribution."""
    try:
        return get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inspections")
async def list_inspections(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List paginated inspection records from the historical database."""
    try:
        return get_inspections(limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inspections/{inspection_id}")
async def retrieve_inspection_detail(inspection_id: str):
    """Retrieve detailed record and bounding box detections for a specific inspection ID."""
    try:
        record = get_inspection_by_id(inspection_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Inspection not found.")
        return record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

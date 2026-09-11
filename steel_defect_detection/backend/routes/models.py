"""
Model management API routes.
"""

from fastapi import APIRouter
from backend.model_manager import ModelManager

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/status")
async def get_model_status():
    """Get active status and GPU/CPU device allocations for all 4 models."""
    mm = ModelManager()
    return mm.get_status()


@router.post("/reload")
async def reload_models():
    """Trigger hot reloading of weights from disk."""
    mm = ModelManager()
    mm.load_all_models(mm.config)
    return {"status": "reloaded", "details": mm.get_status()}

"""
Inspection API routes.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

from backend.model_manager import ModelManager
from backend.inference_service import InferenceService

router = APIRouter(prefix="/api/v1/inspect", tags=["Inspection"])

_service: Optional[InferenceService] = None


def get_inference_service(config: dict) -> InferenceService:
    global _service
    if _service is None:
        mm = ModelManager()
        _service = InferenceService(config, mm)
    return _service


@router.post("")
async def inspect_image(
    file: UploadFile = File(...),
    strip_position: Optional[str] = Form(None),
    save_to_db: bool = Form(True),
):
    """
    Submit a steel strip image for real-time defect inspection.
    Returns defect detections, confidence breakdown, severity, PLC payload, and visual annotations.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        service = get_inference_service(ModelManager().config)
        result = service.process_image(
            image_bytes=contents,
            filename=file.filename or "upload.jpg",
            strip_position=strip_position,
            save_to_db=save_to_db,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

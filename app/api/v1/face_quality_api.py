from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.image_utils import load_image_from_bytes
from app.utils.face_quality import (
    is_blurry,
    is_too_dark_or_bright,
    validate_face_size
)
import cv2
import numpy as np

router = APIRouter()

@router.post("/quality-check")
async def quality_check(image: UploadFile = File(...)):
    try:
        bytes_data = await image.read()
        img = load_image_from_bytes(bytes_data)

        # Run checks but do NOT raise — return messages for UI
        errors = []
        warnings = []

        # 1. Blurriness
        blurry, score = is_blurry(img)
        if blurry:
            errors.append(f"Image is too blurry (score={score:.2f}, need >80)")

        # 2. Lighting
        bad_light, brightness = is_too_dark_or_bright(img)
        if bad_light:
            errors.append(f"Bad lighting (brightness={brightness:.2f}, expected 60–200)")

        # 3. Face size
        try:
            validate_face_size(img)
        except HTTPException as e:
            errors.append(e.detail)

        return {
            "status": "failed" if errors else "ok",
            "errors": errors,
            "brightness": brightness,
            "blurriness_score": score
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")
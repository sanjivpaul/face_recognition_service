import cv2
import numpy as np
import face_recognition
from fastapi import HTTPException


def is_blurry(image, threshold=80):
    """Detect blurriness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def is_too_dark_or_bright(image, min_brightness=60, max_brightness=200):
    """Checks average brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return not (min_brightness < mean_val < max_brightness), mean_val


def validate_face_size(frame, min_face_ratio=0.15):
    """Checks that face bounding box is large enough."""
    face_locations = face_recognition.face_locations(frame)

    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected")

    top, right, bottom, left = face_locations[0]
    h, w = frame.shape[:2]

    face_h = bottom - top
    face_ratio = face_h / h

    if face_ratio < min_face_ratio:
        raise HTTPException(
            status_code=400,
            detail=f"Face too small in frame. Minimum required {min_face_ratio*100:.1f}% of image height"
        )

    return True

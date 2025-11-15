# import numpy as np
# import cv2
# from fastapi import UploadFile, HTTPException

# def load_image(image_file: UploadFile):
#     try:
#         image_bytes = image_file.file.read()
#         if not image_bytes:
#             raise ValueError("Empty image file")

#         image = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(image, cv2.IMREAD_COLOR)

#         if img is None:
#             raise ValueError("Invalid image format")

#         return img
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


# ========================================
#  working v2
# ========================================


# import numpy as np
# import cv2
# from fastapi import UploadFile, HTTPException

# MIN_WIDTH = 200
# MIN_HEIGHT = 200
# MAX_WIDTH = 3000
# MAX_HEIGHT = 3000

# def load_image(image_file: UploadFile):
#     try:
#         # Read image bytes
#         image_bytes = image_file.file.read()
#         if not image_bytes:
#             raise ValueError("Empty image file")

#         # Decode image
#         image = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(image, cv2.IMREAD_COLOR)

#         if img is None:
#             raise ValueError("Invalid or corrupted image format")

#         # Get dimensions
#         h, w, _ = img.shape

#         # Validate minimum resolution
#         if w < MIN_WIDTH or h < MIN_HEIGHT:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Image too small: {w}x{h}. Minimum required is {MIN_WIDTH}x{MIN_HEIGHT}px"
#             )

#         # (Optional) Validate maximum resolution to avoid 8K uploads
#         if w > MAX_WIDTH or h > MAX_HEIGHT:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Image too large: {w}x{h}. Max allowed is {MAX_WIDTH}x{MAX_HEIGHT}px"
#             )

#         return img

#     except HTTPException:
#         # Let FastAPI handle this directly
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")



# ========================================
#  working v3
# ========================================



# from app.utils.face_quality import (
#     is_blurry,
#     is_too_dark_or_bright,
#     validate_face_size
# )

# import numpy as np
# import cv2
# from fastapi import UploadFile, HTTPException

# MIN_WIDTH = 200
# MIN_HEIGHT = 200
# MAX_WIDTH = 3000
# MAX_HEIGHT = 3000

# def load_image(image_file: UploadFile):
#     try:
#         image_bytes = image_file.file.read()
#         if not image_bytes:
#             raise ValueError("Empty image file")

#         image = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(image, cv2.IMREAD_COLOR)

#         if img is None:
#             raise ValueError("Invalid or corrupted image format")

#         # Resolution checks (existing)
#         h, w, _ = img.shape
#         if w < MIN_WIDTH or h < MIN_HEIGHT:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Image too small: {w}x{h}. Minimum required is {MIN_WIDTH}x{MIN_HEIGHT}px"
#             )

#         # -----------------------------
#         # ➕ NEW: FACE QUALITY CHECKS
#         # -----------------------------

#         # 1. Reject blurry images
#         blurry, score = is_blurry(img)
#         if blurry:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Image too blurry. Sharpness score={score:.2f}, required > 80"
#             )

#         # 2. Reject dark or over-bright images
#         bad_light, brightness = is_too_dark_or_bright(img)
#         if bad_light:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Bad lighting. Brightness={brightness:.2f}. Expected 60-200"
#             )

#         # 3. Ensure face size is big enough
#         validate_face_size(img)

#         return img

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")








# ========================================
#  working v4
# ========================================


# app/utils/image_utils.py

import cv2
import numpy as np
import requests
from fastapi import UploadFile, HTTPException

from app.utils.face_quality import (
    is_blurry,
    is_too_dark_or_bright,
    validate_face_size
)

# Resolution limits
MIN_WIDTH = 200
MIN_HEIGHT = 200
MAX_WIDTH = 3000
MAX_HEIGHT = 3000


# ------------------------------------------------------------
# Core image loading utilities
# ------------------------------------------------------------

def load_image_from_bytes(bytes_data: bytes):
    """Decode image bytes into an OpenCV array."""
    arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return img


def download_image(url: str):
    """Download and decode image from a URL."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return load_image_from_bytes(r.content)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to download or decode image from URL")


# ------------------------------------------------------------
# Main image loader used everywhere
# ------------------------------------------------------------

def load_image(image_file: UploadFile):
    """Load image from UploadFile AND perform quality checks."""

    try:
        bytes_data = image_file.file.read()
        if not bytes_data:
            raise ValueError("Empty image file")

        img = load_image_from_bytes(bytes_data)

        # Resolution validation
        h, w, _ = img.shape
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {w}x{h}. Minimum: {MIN_WIDTH}x{MIN_HEIGHT}"
            )

        # --------------------------------------------------------
        # FACE QUALITY CHECKS
        # --------------------------------------------------------

        blurry, score = is_blurry(img)
        if blurry:
            raise HTTPException(
                status_code=400,
                detail=f"Image too blurry. Score={score:.2f}, need >80"
            )

        bad_light, brightness = is_too_dark_or_bright(img)
        if bad_light:
            raise HTTPException(
                status_code=400,
                detail=f"Lighting issue. Brightness={brightness:.2f}, expected 60–200"
            )

        validate_face_size(img)

        return img

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

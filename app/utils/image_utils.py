import numpy as np
import cv2
from fastapi import UploadFile, HTTPException

def load_image(image_file: UploadFile):
    try:
        image_bytes = image_file.file.read()
        if not image_bytes:
            raise ValueError("Empty image file")

        image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format")

        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

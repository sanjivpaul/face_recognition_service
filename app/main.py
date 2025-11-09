# FastAPI Facial Recognition Microservice (Starter Template)
# ----------------------------------------------------------
# This is a minimal, production-ready starter for your facial recognition API.
# It provides two endpoints: /embed (extract embedding) and /verify (compare faces).
# Later, you can containerize it and connect it with your main backend (Node.js or Spring Boot).

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import face_recognition
import io

import os, site, pathlib

# --- Fix for Python 3.14 ---
for p in site.getsitepackages() + [site.getusersitepackages()]:
    models_dir = pathlib.Path(p) / "face_recognition_models"
    if models_dir.exists():
        os.environ["FACE_RECOGNITION_MODELS_PATH"] = str(models_dir)
        break
# ----------------------------
app = FastAPI(title="Facial Recognition Service", version="1.0.0")

# ----------------------------
# Models & Schemas
# ----------------------------
class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_version: str = "face_recognition_v1"
    timestamp: str

class VerificationResponse(BaseModel):
    match: bool
    distance: float
    confidence: float

# ----------------------------
# Utility functions
# ----------------------------
# def load_image(image_file: UploadFile):
#     try:
#         image_bytes = image_file.file.read()
#         image = np.array(bytearray(image_bytes), dtype=np.uint8)
#         img = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         return img
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image file")

def load_image(image_file: UploadFile):
    try:
        image_bytes = image_file.file.read()
        if not image_bytes:
            raise ValueError("Empty image file")

        image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid or unreadable image format")

        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
# ----------------------------
# Endpoints
# ----------------------------
@app.post("/api/v1/embed", response_model=EmbeddingResponse)
def extract_embedding(image: UploadFile = File(...)):
    import datetime
    img = load_image(image)

    # Convert BGR to RGB for face_recognition
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if len(encodings) == 0:
        raise HTTPException(status_code=404, detail="No face detected")

    embedding = encodings[0].tolist()
    ts = datetime.datetime.utcnow().isoformat()

    return {
        "embedding": embedding,
        "model_version": "face_recognition_v1",
        "timestamp": ts
    }

# @app.post("/api/v1/verify", response_model=VerificationResponse)
# def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
#     img1 = load_image(image1)
#     img2 = load_image(image2)

#     rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#     enc1 = face_recognition.face_encodings(rgb1)
#     enc2 = face_recognition.face_encodings(rgb2)

#     if len(enc1) == 0 or len(enc2) == 0:
#         raise HTTPException(status_code=404, detail="Face not found in one or both images")

#     face_distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
#     threshold = 0.6  # configurable later
#     match = face_distance < threshold
#     confidence = float(1 - face_distance)

#     return {
#         "match": match,
#         "distance": float(face_distance),
#         "confidence": confidence
#     }

@app.post("/api/v1/verify", response_model=VerificationResponse)
def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = load_image(image1)
    img2 = load_image(image2)

    # Double-check that both images are loaded correctly
    if img1 is None or img2 is None:
        raise HTTPException(status_code=400, detail="One or both images could not be read")

    # Convert BGR to RGB for face_recognition
    try:
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        raise HTTPException(status_code=400, detail=f"Color conversion failed: {str(e)}")

    enc1 = face_recognition.face_encodings(rgb1)
    enc2 = face_recognition.face_encodings(rgb2)

    if len(enc1) == 0 or len(enc2) == 0:
        raise HTTPException(status_code=404, detail="Face not found in one or both images")

    face_distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
    threshold = 0.6
    match = face_distance < threshold
    confidence = float(1 - face_distance)

    return {
        "match": match,
        "distance": float(face_distance),
        "confidence": confidence
    }
# ----------------------------
# Run locally Version v1.0.1
# ----------------------------
# Use: uvicorn app.main:app --reload
# Then test:
# curl -X POST "http://127.0.0.1:8000/api/v1/embed" -F "image=@face.jpg"
# curl -X POST "http://127.0.0.1:8000/api/v1/verify" -F "image1=@face1.jpg" -F "image2=@face2.jpg"

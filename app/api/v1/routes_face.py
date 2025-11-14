from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import EmbeddingResponse, VerificationResponse
from app.utils.image_utils import load_image
from app.services.face_service import extract_embedding_from_image, verify_faces
from app.core.config import settings
import datetime

router = APIRouter()

@router.post("/embed", response_model=EmbeddingResponse)
def extract_embedding(image: UploadFile = File(...)):
    img = load_image(image)
    embedding = extract_embedding_from_image(img)

    if embedding is None:
        raise HTTPException(status_code=404, detail="No face detected")

    ts = datetime.datetime.utcnow().isoformat()
    return {
        "embedding": embedding,
        "model_version": settings.MODEL_VERSION,
        "embedding_version": "v1",  # manually track changes
        "timestamp": ts
    }

@router.post("/verify", response_model=VerificationResponse)
def verify_faces_route(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = load_image(image1)
    img2 = load_image(image2)
    result = verify_faces(img1, img2)

    if result is None:
        raise HTTPException(status_code=404, detail="Face not found in one or both images")

    match, distance, confidence = result
    return {
        "match": match,
        "distance": distance,
        "confidence": confidence
    }

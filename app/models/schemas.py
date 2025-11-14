from pydantic import BaseModel
from typing import List

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model_version: str
    embedding_version:str
    timestamp: str

class VerificationResponse(BaseModel):
    match: bool
    distance: float
    confidence: float

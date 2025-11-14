import face_recognition
import cv2
import numpy as np
import datetime
from app.core.config import settings

def extract_embedding_from_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None
    # return encodings[0].tolist()
    
    embedding = encodings[0]
    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def verify_faces(img1, img2):
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    enc1 = face_recognition.face_encodings(rgb1)
    enc2 = face_recognition.face_encodings(rgb2)
    if not enc1 or not enc2:
        return None

    face_distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
    match = face_distance < settings.FACE_THRESHOLD
    confidence = float(1 - face_distance)
    return match, face_distance, confidence

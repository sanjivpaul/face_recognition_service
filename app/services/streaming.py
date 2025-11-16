# import asyncio
# import base64
# import json
# import cv2
# import numpy as np
# from fastapi import WebSocket
# from app.services.face_service import extract_embedding_from_image
# from app.services.kafka_producer import send_embedding_result
# # from app.services.embedding_worker import load_all_user_embeddings  # optional cache
# from app.services.embeddings_cache import embeddings_cache  # singleton cache
# import logging

# logger = logging.getLogger("face-stream")
# logging.basicConfig(level=logging.INFO)

# # Load known user embeddings from DB (to compare quickly)
# # USER_EMBEDDINGS = load_all_user_embeddings()  # {userId: embedding_list}

# # Access the embeddings dynamically from cache
# def get_user_embeddings():
#     return embeddings_cache.get_embeddings()  # {userId: np.array([...])}

# # def decode_base64_image(data_url:str):
# #     """Decode base64 image from client"""
# #     header, encode = data_url.split(",", 1)
# #     img_bytes = base64.b64decode(encode)
# #     nparr = np.frombuffer(img_bytes, np.unit8)
# #     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# #     return img

# def decode_base64_image(data_url: str):
#     """Decode base64 image from client"""
#     header, encode = data_url.split(",", 1)
#     img_bytes = base64.b64decode(encode)
#     nparr = np.frombuffer(img_bytes, np.uint8)  # correct dtype
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img


# def find_best_match(live_embedding, user_embeddings, threshold=0.6):
#     """
#     Compare live embedding with all user embeddings.
#     live_embedding: np.array
#     user_embeddings: dict {userId: np.array}
#     """
#     from scipy.spatial.distance import cosine

#     best_match = None
#     best_score = 1.0  # lower is better

#     for user_id, db_embedding in user_embeddings.items():
#         score = cosine(live_embedding, db_embedding)
#         if score < best_score and score < threshold:
#             best_score = score
#             best_match = user_id

#     return best_match, best_score


# # async def face_stream(websocket: WebSocket):
# #     await websocket.accept()
# #     await websocket.send_text(json.dumps({"status":"connected"}))

# #     try:
# #         while True:
# #             data = await websocket.receive_text()
# #             frame = decode_base64_image(data)

# #             # Extract embedding
# #             embedding = extract_embedding_from_image(frame)
# #             if embedding is None:
# #                 await websocket.send_text(json.dumps({"status":"no_face"}))
# #                 continue

# #                         # Fetch embeddings dynamically from cache
# #             USER_EMBEDDINGS = embeddings_cache.get_embeddings()

# #             # Find match
# #             user_id, score = find_best_match(embedding, USER_EMBEDDINGS)
# #             if user_id:
# #                 # Send attendance event to kafka
# #                 send_embedding_result({
# #                     "userId":user_id,
# #                     "status":"completed",
# #                     "embedding":embedding,
# #                     "biometricId": user_id  # or map real biometricId
# #                 })

# #                 await websocket.send_text(json.dumps({"status": "matched", "userId": user_id}))
# #             else:
# #                 await websocket.send_text(json.dumps({"status": "unmatched"}))

            
# #             await asyncio.sleep(0.05)  # small delay to not overwhelm
# #     except Exception as e:
# #         await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
# #         await websocket.close()


# async def face_stream(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("WebSocket connected")

#     try:
#         while True:
#             data = await websocket.receive_text()
#             frame = decode_base64_image(data)

#             # Extract embedding
#             embedding = extract_embedding_from_image(frame)
#             if embedding is None:
#                 await websocket.send_text(json.dumps({"status":"no_face"}))
#                 logger.info("No face detected in frame")
#                 continue

#             # Get latest embeddings from cache
#             USER_EMBEDDINGS = embeddings_cache.get_embeddings()

#             # Find best match
#             user_id, score = find_best_match(embedding, USER_EMBEDDINGS)

#             if user_id:
#                 # Convert embedding to list before sending
#                 send_embedding_result({
#                     "userId": user_id,
#                     "status": "completed",
#                     "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
#                     "biometricId": user_id
#                 })

#                 await websocket.send_text(json.dumps({
#                     "status": "matched",
#                     "userId": user_id,
#                     "score": score,  # optional confidence
#                     "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
#                 }))
#                 logger.info(f"Face matched: userId={user_id}, score={score:.4f}")
#             else:
#                 await websocket.send_text(json.dumps({"status": "unmatched"}))
#                 logger.info("No match found")

#             await asyncio.sleep(0.05)

#     except Exception as e:
#         await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
#         logger.error(f"WebSocket error: {e}")
#         await websocket.close()





# ===========================
# v2
# ===========================



# import asyncio
# import base64
# import json
# import cv2
# import numpy as np
# from fastapi import WebSocket
# import face_recognition
# from app.services.face_service import extract_embedding_from_image
# from app.services.kafka_producer import send_embedding_result
# from app.services.embeddings_cache import embeddings_cache
# import logging
# from scipy.spatial.distance import cosine
# from numpy.linalg import norm


# logger = logging.getLogger("face-stream")
# logging.basicConfig(level=logging.INFO)


# def normalize_embedding(embedding):
#     return embedding / norm(embedding)

# def decode_base64_image(data_url: str):
#     """Decode base64 image from client"""
#     header, encode = data_url.split(",", 1)
#     img_bytes = base64.b64decode(encode)
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

# def find_best_match(live_embedding, user_embeddings, threshold=0.45):
#     """Compare live embedding with all user embeddings."""
#     best_match = None
#     best_score = 1.0
#     for user_id, db_embedding in user_embeddings.items():
#         if db_embedding is None:
#             continue
#         # score = cosine(live_embedding, db_embedding)
#         # if score < best_score and score < threshold:
#         score_cos = cosine(live_embedding, db_embedding)
#         score_euc = np.linalg.norm(live_embedding - db_embedding)
#         if score_cos < 0.45 and score_euc < 0.6:
#             best_score = score_cos
#             best_match = user_id
#     return best_match, best_score

# async def face_stream(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("WebSocket connected")

#     try:
#         while True:
#             data = await websocket.receive_text()
#             frame = decode_base64_image(data)
#             # user_embeddings = embeddings_cache.get_embeddings()  # dynamic cache
#             user_embeddings = {uid: normalize_embedding(e) for uid, e in embeddings_cache.get_embeddings().items()}


#             # Detect faces
#             face_locations = face_recognition.face_locations(frame)

#             if not face_locations:
#                 await websocket.send_text(json.dumps({"status": "no_face"}))
#                 logger.info("No face detected in frame")
#                 continue

#             annotated_frame = frame.copy()

#             # # Process each face
#             # for face_location in face_locations:
#             #     top, right, bottom, left = face_location
#             #     face_img = frame[top:bottom, left:right]

#             #     # Extract embedding
#             #     embedding = normalize_embedding(extract_embedding_from_image(face_img))
#             #     if embedding is None:
#             #         continue

#             #     # Find best match
#             #     user_id, score = find_best_match(embedding, user_embeddings)

#             for face_location in face_locations:
#                 top, right, bottom, left = face_location
#                 face_img = frame[top:bottom, left:right]

#                 # Extract embedding
#                 raw_embedding = extract_embedding_from_image(face_img)

#                 if raw_embedding is None:
#                     logger.warning("Embedding extraction returned None")
#                     continue

#                 embedding = normalize_embedding(raw_embedding)

#                 # Find best match
#                 user_id, score = find_best_match(embedding, user_embeddings)

#                 # Draw rectangle and name
#                 color = (0, 255, 0) if user_id else (0, 0, 255)
#                 cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
#                 name = f"User {user_id}" if user_id else "Unknown"
#                 cv2.putText(annotated_frame, name, (left, top - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#                 # Send attendance if matched
#                 if user_id:
#                     send_embedding_result({
#                         "userId": user_id,
#                         "status": "completed",
#                         "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
#                         "biometricId": user_id
#                     })
#                     logger.info(f"Face matched: userId={user_id}, score={score:.4f}")

#             # Encode annotated frame to base64
#             _, buffer = cv2.imencode(".jpg", annotated_frame)
#             jpg_as_text = base64.b64encode(buffer).decode("utf-8")

#             await websocket.send_text(json.dumps({
#                 "status": "frame",
#                 "image": jpg_as_text
#             }))

#             await asyncio.sleep(0.03)  # small delay to yield control

#     except Exception as e:
#         await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
#         logger.error(f"WebSocket error: {e}")
#         await websocket.close()






# import asyncio
# import base64
# import json
# import cv2
# import numpy as np
# from fastapi import WebSocket
# from mtcnn import MTCNN
# from insightface.app import FaceAnalysis  # ArcFace backend
# from app.services.kafka_producer import send_embedding_result
# from app.services.embeddings_cache import embeddings_cache
# import logging
# from scipy.spatial.distance import cosine
# from numpy.linalg import norm

# logger = logging.getLogger("face-stream")
# logging.basicConfig(level=logging.INFO)

# # Initialize MTCNN for face detection
# detector = MTCNN()

# # Initialize ArcFace model
# face_app = FaceAnalysis(name='antelope', providers=['CPUExecutionProvider'])
# face_app.prepare(ctx_id=0, nms=0.4)

# # Helper functions
# def normalize_embedding(embedding):
#     """L2-normalize embedding for cosine comparison."""
#     return embedding / norm(embedding)

# def decode_base64_image(data_url: str):
#     header, encode = data_url.split(",", 1)
#     img_bytes = base64.b64decode(encode)
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

# def find_best_match(live_embedding, user_embeddings, threshold_cos=0.45, threshold_euc=0.6):
#     """Find best match for a live embedding."""
#     best_match = None
#     best_score = 1.0
#     for user_id, db_embedding in user_embeddings.items():
#         score_cos = cosine(live_embedding, db_embedding)
#         score_euc = np.linalg.norm(live_embedding - db_embedding)
#         if score_cos < threshold_cos and score_euc < threshold_euc:
#             best_score = score_cos
#             best_match = user_id
#     return best_match, best_score

# # Keep track of consecutive matches to reduce false positives
# MATCH_HISTORY = {}  # {user_id: count}

# async def face_stream(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("WebSocket connected")

#     try:
#         while True:
#             data = await websocket.receive_text()
#             frame = decode_base64_image(data)

#             # Get embeddings cache (normalized)
#             user_embeddings = {
#                 uid: normalize_embedding(e) for uid, e in embeddings_cache.get_embeddings().items()
#             }

#             # Detect faces using MTCNN
#             detections = detector.detect_faces(frame)

#             if not detections:
#                 await websocket.send_text(json.dumps({"status": "no_face"}))
#                 logger.info("No face detected in frame")
#                 continue

#             annotated_frame = frame.copy()

#             for det in detections:
#                 x, y, width, height = det['box']
#                 x, y = max(0, x), max(0, y)
#                 face_img = frame[y:y+height, x:x+width]

#                 # Extract embedding using ArcFace
#                 faces = face_app.get(np.array(face_img))
#                 if not faces:
#                     continue
#                 embedding = normalize_embedding(faces[0].embedding)

#                 # Find best match
#                 user_id, score = find_best_match(embedding, user_embeddings)

#                 # Multi-frame verification
#                 if user_id:
#                     MATCH_HISTORY[user_id] = MATCH_HISTORY.get(user_id, 0) + 1
#                     if MATCH_HISTORY[user_id] < 3:  # Require 3 consecutive frames
#                         user_id = None  # Temporarily ignore
#                 else:
#                     # Reset history for unmatched users
#                     MATCH_HISTORY = {k: v for k, v in MATCH_HISTORY.items() if k != user_id}

#                 # Draw rectangle and name
#                 color = (0, 255, 0) if user_id else (0, 0, 255)
#                 name = f"User {user_id}" if user_id else "Unknown"
#                 cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), color, 2)
#                 cv2.putText(annotated_frame, name, (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#                 # Send attendance if verified
#                 if user_id:
#                     send_embedding_result({
#                         "userId": user_id,
#                         "status": "completed",
#                         "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
#                         "biometricId": user_id
#                     })
#                     logger.info(f"Face matched: userId={user_id}, score={score:.4f}")

#             # Encode annotated frame and send to client
#             _, buffer = cv2.imencode(".jpg", annotated_frame)
#             jpg_as_text = base64.b64encode(buffer).decode("utf-8")

#             await websocket.send_text(json.dumps({
#                 "status": "frame",
#                 "image": jpg_as_text
#             }))

#             await asyncio.sleep(0.03)

#     except Exception as e:
#         await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
#         logger.error(f"WebSocket error: {e}")
#         await websocket.close()










import asyncio
import base64
import json
import cv2
import numpy as np
from fastapi import WebSocket
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from numpy.linalg import norm
from app.services.embeddings_cache import embeddings_cache
from app.services.kafka_producer import send_embedding_result
import logging

logger = logging.getLogger("face-stream")
logging.basicConfig(level=logging.INFO)

# Initialize ArcFace + RetinaFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

MATCH_HISTORY = {}   # {user_id: count}
REQUIRED_CONSECUTIVE_MATCHES = 3


def normalize_embedding(e):
    return e / norm(e)


def decode_base64_image(data_url):
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def find_best_match(live, db_embeddings, threshold=0.45):
    best_id = None
    best_score = 1.0

    for user_id, db in db_embeddings.items():
        score = cosine(live, db)
        if score < best_score and score < threshold:
            best_score = score
            best_id = user_id

    return best_id, best_score


async def face_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_base64_image(data)

            # Load DB embeddings
            user_embeddings = {
                uid: normalize_embedding(e)
                for uid, e in embeddings_cache.get_embeddings().items()
            }

            # Detect + embed faces using InsightFace
            faces = face_app.get(frame)

            if not faces:
                await websocket.send_text(json.dumps({"status": "no_face"}))
                continue

            annotated = frame.copy()

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                emb = normalize_embedding(face.embedding.astype("float32"))

                # Verify identity
                user_id, score = find_best_match(emb, user_embeddings)

                # Multi-frame check
                if user_id:
                    MATCH_HISTORY[user_id] = MATCH_HISTORY.get(user_id, 0) + 1

                    if MATCH_HISTORY[user_id] < REQUIRED_CONSECUTIVE_MATCHES:
                        user_id = None  # Not stable yet
                else:
                    MATCH_HISTORY.clear()  # Reset if mismatch

                color = (0, 255, 0) if user_id else (0, 0, 255)
                name = f"User {user_id}" if user_id else "Unknown"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Send identification event
                if user_id:
                    send_embedding_result({
                        "userId": user_id,
                        "status": "completed",
                        "embedding": emb.tolist(),
                        "biometricId": user_id
                    })
                    logger.info(f"MATCH: {user_id} (score={score:.4f})")

            # Send annotated frame back
            _, buf = cv2.imencode(".jpg", annotated)
            jpg = base64.b64encode(buf).decode("utf-8")

            await websocket.send_text(json.dumps({
                "status": "frame",
                "image": jpg
            }))

            await asyncio.sleep(0.03)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
        await websocket.close()

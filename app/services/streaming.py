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









import asyncio
import base64
import json
import cv2
import numpy as np
from fastapi import WebSocket
import face_recognition
from app.services.face_service import extract_embedding_from_image
from app.services.kafka_producer import send_embedding_result
from app.services.embeddings_cache import embeddings_cache
import logging
from scipy.spatial.distance import cosine
from numpy.linalg import norm


logger = logging.getLogger("face-stream")
logging.basicConfig(level=logging.INFO)


def normalize_embedding(embedding):
    return embedding / norm(embedding)

def decode_base64_image(data_url: str):
    """Decode base64 image from client"""
    header, encode = data_url.split(",", 1)
    img_bytes = base64.b64decode(encode)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def find_best_match(live_embedding, user_embeddings, threshold=0.45):
    """Compare live embedding with all user embeddings."""
    best_match = None
    best_score = 1.0
    for user_id, db_embedding in user_embeddings.items():
        # score = cosine(live_embedding, db_embedding)
        # if score < best_score and score < threshold:
        score_cos = cosine(live_embedding, db_embedding)
        score_euc = np.linalg.norm(live_embedding - db_embedding)
        if score_cos < 0.45 and score_euc < 0.6:
            best_score = score_cos
            best_match = user_id
    return best_match, best_score

async def face_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_base64_image(data)
            # user_embeddings = embeddings_cache.get_embeddings()  # dynamic cache
            user_embeddings = {uid: normalize_embedding(e) for uid, e in embeddings_cache.get_embeddings().items()}


            # Detect faces
            face_locations = face_recognition.face_locations(frame)

            if not face_locations:
                await websocket.send_text(json.dumps({"status": "no_face"}))
                logger.info("No face detected in frame")
                continue

            annotated_frame = frame.copy()

            # Process each face
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_img = frame[top:bottom, left:right]

                # Extract embedding
                embedding = normalize_embedding(extract_embedding_from_image(face_img))
                if embedding is None:
                    continue

                # Find best match
                user_id, score = find_best_match(embedding, user_embeddings)

                # Draw rectangle and name
                color = (0, 255, 0) if user_id else (0, 0, 255)
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                name = f"User {user_id}" if user_id else "Unknown"
                cv2.putText(annotated_frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Send attendance if matched
                if user_id:
                    send_embedding_result({
                        "userId": user_id,
                        "status": "completed",
                        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                        "biometricId": user_id
                    })
                    logger.info(f"Face matched: userId={user_id}, score={score:.4f}")

            # Encode annotated frame to base64
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text(json.dumps({
                "status": "frame",
                "image": jpg_as_text
            }))

            await asyncio.sleep(0.03)  # small delay to yield control

    except Exception as e:
        await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

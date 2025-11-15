Perfect! Here’s a full rewrite of your `face_stream` WebSocket function that:

- Detects faces in the frame
- Extracts embeddings
- Matches against cached embeddings
- Draws rectangles and matched user names
- Sends annotated frames back to the client as base64

```python
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

logger = logging.getLogger("face-stream")
logging.basicConfig(level=logging.INFO)

def decode_base64_image(data_url: str):
    """Decode base64 image from client"""
    header, encode = data_url.split(",", 1)
    img_bytes = base64.b64decode(encode)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def find_best_match(live_embedding, user_embeddings, threshold=0.6):
    """Compare live embedding with all user embeddings."""
    best_match = None
    best_score = 1.0
    for user_id, db_embedding in user_embeddings.items():
        score = cosine(live_embedding, db_embedding)
        if score < best_score and score < threshold:
            best_score = score
            best_match = user_id
    return best_match, best_score

async def face_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_base64_image(data)
            user_embeddings = embeddings_cache.get_embeddings()  # dynamic cache

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
                embedding = extract_embedding_from_image(face_img)
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
```

---

### ✅ Features added:

1. Handles multiple faces in a frame.
2. Draws **green rectangle** for matched faces, **red for unknown**.
3. Displays **user name** above the rectangle.
4. Sends **annotated frame** as Base64 to frontend.
5. Sends **attendance events** to Kafka for matched users.

---

On the frontend, you can decode and display the `image` field from JSON in a `<img>` tag using:

```js
img.src = "data:image/jpeg;base64," + data.image;
```

---

If you want, I can also **optimize it for real-time performance** using **smaller frame resizing, processing every 2-3 frames**, so it won’t lag the webcam stream.

Do you want me to do that optimization as well?

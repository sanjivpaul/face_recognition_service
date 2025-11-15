Perfect! Let's build a **real-time face recognition WebSocket API** in your Face Service that streams frames from a browser, matches faces, and sends Kafka events for attendance marking.

This will integrate directly with your current Kafka + embeddings system.

---

### 1️⃣ Install dependencies

Make sure you have `fastapi`, `uvicorn`, `opencv-python`, `numpy`, `face_recognition`, and `websockets`:

```bash
pip install fastapi uvicorn opencv-python numpy face_recognition python-multipart websockets
```

---

### 2️⃣ Create `app/services/streaming.py`

```python
import asyncio
import base64
import json
import cv2
import numpy as np
from fastapi import WebSocket
from app.services.face_service import extract_embedding_from_image
from app.services.kafka_producer import send_embedding_result
from app.services.embedding_worker import load_all_user_embeddings  # optional cache

# Load known user embeddings from DB (to compare quickly)
USER_EMBEDDINGS = load_all_user_embeddings()  # {userId: embedding_list}


def decode_base64_image(data_url: str):
    """Decode base64 image from client."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def find_best_match(live_embedding, threshold=0.6):
    """Compare live embedding with all user embeddings."""
    from scipy.spatial.distance import cosine

    best_match = None
    best_score = 1.0  # cosine distance, lower is better

    for user_id, db_embedding in USER_EMBEDDINGS.items():
        score = cosine(live_embedding, db_embedding)
        if score < best_score and score < threshold:
            best_score = score
            best_match = user_id

    return best_match, best_score


async def face_stream(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({"status": "connected"}))

    try:
        while True:
            data = await websocket.receive_text()
            frame = decode_base64_image(data)

            # Extract embedding
            embedding = extract_embedding_from_image(frame)
            if embedding is None:
                await websocket.send_text(json.dumps({"status": "no_face"}))
                continue

            # Find match
            user_id, score = find_best_match(embedding)
            if user_id:
                # Send attendance event to Kafka
                send_embedding_result({
                    "userId": user_id,
                    "status": "completed",
                    "embedding": embedding,
                    "biometricId": user_id  # or map real biometricId
                })
                await websocket.send_text(json.dumps({"status": "matched", "userId": user_id}))
            else:
                await websocket.send_text(json.dumps({"status": "unmatched"}))

            await asyncio.sleep(0.05)  # small delay to not overwhelm
    except Exception as e:
        await websocket.send_text(json.dumps({"status": "error", "error": str(e)}))
        await websocket.close()
```

---

### 3️⃣ Create WebSocket route in `app/api/v1/routes_face.py`

```python
from fastapi import APIRouter, WebSocket
from app.services.streaming import face_stream

router = APIRouter()

@router.websocket("/ws/attendance")
async def websocket_attendance(ws: WebSocket):
    await face_stream(ws)
```

---

### 4️⃣ Browser-side example

Send webcam frames to FastAPI WebSocket as base64:

```html
<video id="video" autoplay></video>
<script>
  const ws = new WebSocket("ws://127.0.0.1:8000/api/v1/ws/attendance");

  ws.onmessage = (event) => {
    console.log("Server:", event.data);
  };

  navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    const video = document.getElementById("video");
    video.srcObject = stream;

    const canvas = document.createElement("canvas");
    setInterval(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL("image/jpeg");
      ws.send(dataUrl);
    }, 200); // send frame every 200ms
  });
</script>
```

---

### 5️⃣ Key points

- **Face embeddings cache:** Load all user embeddings into memory for faster matching.
- **Thresholds:** Adjust `cosine` distance threshold for matches.
- **Kafka decoupling:** Attendance events are sent to AMS, which writes to DB.
- **WebSocket:** Browser streams frames; server responds with `matched/unmatched/no_face`.

---

✅ With this setup:

- Face Service handles **real-time face recognition**.
- AMS consumes **Kafka attendance events**.
- UI gets **immediate feedback** on matches.

---

If you want, I can also **update your existing embedding worker** to include **loading embeddings into memory** for faster real-time streaming so the WebSocket will never query DB on every frame.

Do you want me to do that?

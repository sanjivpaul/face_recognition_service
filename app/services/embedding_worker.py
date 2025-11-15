import time
import json
import logging
from datetime import datetime

from app.services.kafka_consumer import read_embedding_job
from app.services.kafka_producer import send_embedding_result
from app.utils.image_utils import download_image
from app.services.face_service import extract_embedding_from_image


# ---------------------------------------------------
# LOGGING CONFIG
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("embedding-worker")


def log_event(event: str, data: dict = None, level="info"):
    """
    Unified structured + console log
    """
    log_obj = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "data": data or {},
    }

    formatted = json.dumps(log_obj)

    if level == "error":
        logger.error(formatted)
    elif level == "warning":
        logger.warning(formatted)
    else:
        logger.info(formatted)


# ---------------------------------------------------
# WORKER LOOP
# ---------------------------------------------------
def run_worker():
    log_event("worker_started")

    while True:
        job = read_embedding_job()
        if not job:
            time.sleep(0.2)
            continue

        user_id = job.get("userId")
        image_url = job.get("imageUrl")
        biometric_id = job.get("biometricId")

        log_event("job_received", {"userId": user_id, "biometricId": biometric_id})

        # ---------------------
        # Download image
        try:
            img = download_image(image_url)
            log_event("image_download_success", {"userId": user_id})
        except Exception as e:
            log_event(
                "image_download_failed",
                {"userId": user_id, "error": str(e)},
                level="error",
            )
            send_embedding_result({
                "userId": user_id,
                "status": "failed",
                "biometricId": biometric_id
            })
            continue

        # ---------------------
        # Extract Embedding
        try:
            # embedding = extract_embedding_from_image(img)
            embedding_np = extract_embedding_from_image(img)

            if embedding_np is not None:
                 # Convert NumPy → list (IMPORTANT)
                embedding = embedding_np.tolist()

                log_event(
                    "embedding_extraction_success",
                    {"userId": user_id, "vector_length": len(embedding)},
                )
            else:
                embedding = None
                log_event(
                    "embedding_extraction_failed",
                    {"userId": user_id, "reason": "no-face-or-low-quality"},
                    level="warning",
                )
        except Exception as e:
            log_event(
                "embedding_exception",
                {"userId": user_id, "error": str(e)},
                level="error",
            )
            embedding = None

        # ---------------------
        # Send result back to Kafka
        status = "completed" if embedding else "failed"
        payload = {
            "userId": user_id,
            "status": status,
            "embedding": embedding,
            "biometricId": biometric_id,
        }

        try:
            send_embedding_result(payload)
            log_event("kafka_send_success", {"userId": user_id, "status": status})
        except Exception as e:
            log_event(
                "kafka_send_failed",
                {"userId": user_id, "error": str(e)},
                level="error",
            )


if __name__ == "__main__":
    run_worker()

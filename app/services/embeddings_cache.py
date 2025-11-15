# app/services/embeddings_cache.py
import requests
import numpy as np
import threading
import time
from typing import Dict

EMBEDDINGS_API_URL = "http://localhost:8888/api/v1/users/embeddings"  # your AMS service API

class EmbeddingsCache:
    def __init__(self, refresh_interval=300):
        """
        refresh_interval: seconds to refresh embeddings cache from AMS service
        """
        self.refresh_interval = refresh_interval
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.lock = threading.Lock()
        self._start_refresh_thread()

    def _start_refresh_thread(self):
        thread = threading.Thread(target=self._refresh_loop, daemon=True)
        thread.start()

    def _refresh_loop(self):
        while True:
            try:
                self.refresh_cache()
            except Exception as e:
                print(f"[EmbeddingsCache] Error refreshing embeddings: {e}")
            time.sleep(self.refresh_interval)

    def refresh_cache(self):
        response = requests.get(EMBEDDINGS_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()  # expect list of {"userId":1, "embedding":[...]}

        new_cache = {}
        for row in data:
            if row.get("embedding"):
                new_cache[row["userId"]] = np.array(row["embedding"], dtype=np.float32)

        with self.lock:
            self.user_embeddings = new_cache
        print(f"[EmbeddingsCache] Loaded {len(new_cache)} embeddings")

    def get_embeddings(self):
        with self.lock:
            return dict(self.user_embeddings)


# singleton instance
embeddings_cache = EmbeddingsCache()

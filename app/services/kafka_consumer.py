# from confluent_kafka import Consumer
# import json

# consumer = Consumer({
#     'bootstrap.servers': 'localhost:9092',
#     'group.id': 'face-service-group',
#     'auto.offset.reset': 'earliest'
# })

# consumer.subscribe(["embedding_jobs"])

# def read_embedding_jobs():
#     msg = consumer.poll(1.0)
#     if msg is None or msg.error():
#         return None

#     return json.loads(msg.value())




# # app/services/kafka_consumer.py
# from confluent_kafka import Consumer
# import json
# from typing import Optional

# consumer = Consumer({
#     'bootstrap.servers': 'localhost:9092',
#     'group.id': 'face-worker-group',
#     'auto.offset.reset': 'earliest'
# })

# consumer.subscribe(['embedding_jobs'])

# def read_embedding_job(timeout=1.0) -> Optional[dict]:
#     msg = consumer.poll(timeout)
#     if msg is None:
#         return None
#     if msg.error():
#         print("Kafka consumer error:", msg.error())
#         return None
#     return json.loads(msg.value().decode('utf-8'))




# app/services/kafka_consumer.py
from confluent_kafka import Consumer
import json
from typing import Optional

def create_consumer():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'face-worker-group',
        'auto.offset.reset': 'earliest'
    })

    consumer.subscribe(['embedding_jobs'])

    # -------- CONNECTION TEST ----------
    try:
        metadata = consumer.list_topics(timeout=3)
        if metadata.brokers:
            print("✅ Kafka Consumer Connected → localhost:9092")
            print("📌 Subscribed to topic: embedding_jobs")
    except Exception as e:
        print("❌ Kafka Consumer Failed to connect:", e)

    return consumer


consumer = create_consumer()


def read_embedding_job(timeout=1.0) -> Optional[dict]:
    msg = consumer.poll(timeout)
    if msg is None:
        return None
    if msg.error():
        print("Kafka consumer error:", msg.error())
        return None
    return json.loads(msg.value().decode('utf-8'))

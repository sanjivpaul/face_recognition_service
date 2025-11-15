# from confluent_kafka import Producer
# import json

# producer = Producer({'bootstrap.servers':'localhost:9092'})

# def send_embedding_result(user_id, status):
#     data = json.dumps({"userId": user_id, "status": status})
#     producer.produce("embedding_results", key=str(user_id), value=data)
#     producer.flush()



# app/services/kafka_producer.py
from confluent_kafka import Producer
import json

producer = Producer({'bootstrap.servers': 'localhost:9092'})

def send_embedding_result(payload: dict):
    # payload example: {"userId": 1, "status": "completed", "embedding": [...], "biometricId": 123}
    producer.produce(
        "embedding_results", 
        key=str(payload.get('userId')), 
        value=json.dumps(payload).encode('utf-8'))
    producer.flush()

"""Consumes the transaction stream: updates the Redis sliding-window feature store, scores
each transaction against the live FastAPI service, and logs the result.

The sliding-window aggregate is a genuine, working Redis feature store — but it isn't yet an
input to the trained model (that would mean retraining on a feature that doesn't exist in the
static offline dataset). It's logged alongside each prediction to demonstrate the mechanism
end-to-end; wiring it into the model is exactly the kind of thing the "what I'd do with more
data" section of the README gestures at.
"""

import json
import os

import httpx
from confluent_kafka import Consumer

from src.streaming.redis_features import get_redis_client, record_transaction, sliding_window_stats

TOPIC = "transactions"


def main() -> None:
    broker = os.environ.get("KAFKA_BROKER", "localhost:9092")
    api_url = os.environ.get("API_URL", "http://localhost:8000")

    consumer = Consumer(
        {
            "bootstrap.servers": broker,
            "group.id": "fraud-consumer",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([TOPIC])
    redis_client = get_redis_client()

    print(f"consuming '{TOPIC}' from {broker}, scoring against {api_url}")
    with httpx.Client(timeout=10.0) as http_client:
        try:
            while True:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"consumer error: {msg.error()}")
                    continue

                transaction = json.loads(msg.value().decode("utf-8"))
                record_transaction(redis_client, amount=transaction["Amount"])
                window = sliding_window_stats(redis_client)

                response = http_client.post(f"{api_url}/predict", json=transaction)
                prediction = response.json()

                print(
                    f"amount=${transaction['Amount']:.2f} "
                    f"fraud_probability={prediction['fraud_probability']:.4f} "
                    f"is_fraud={prediction['is_fraud']} "
                    f"window[{window['window_seconds']}s]: "
                    f"count={window['transaction_count']} sum=${window['amount_sum']:.2f}"
                )
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()


if __name__ == "__main__":
    main()

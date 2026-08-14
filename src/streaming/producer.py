"""Replays test-set transactions onto a Kafka topic with a small delay between messages,
simulating a live transaction stream instead of a static batch replay."""

import json
import os
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

TOPIC = "transactions"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def main() -> None:
    broker = os.environ.get("KAFKA_BROKER", "localhost:9092")
    delay_seconds = float(os.environ.get("PRODUCER_DELAY_SECONDS", "0.2"))
    n = int(os.environ.get("PRODUCER_N", "500"))

    producer = Producer({"bootstrap.servers": broker})

    df = pd.read_parquet(PROCESSED_DIR / "test.parquet").head(n)
    feature_columns = [c for c in df.columns if c != "Class"]

    print(f"producing {len(df)} transactions to '{TOPIC}' on {broker}")
    for _, row in df.iterrows():
        payload = {col: float(row[col]) for col in feature_columns}
        producer.produce(TOPIC, json.dumps(payload).encode("utf-8"))
        producer.poll(0)
        time.sleep(delay_seconds)

    producer.flush()
    print("done")


if __name__ == "__main__":
    main()

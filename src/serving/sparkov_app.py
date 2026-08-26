"""FastAPI inference service for the Sparkov model — the standing-service counterpart to the
replay-only streaming demo. Unlike the demo, every request computes its velocity feature live
from Redis and feeds it straight into the model, the same mechanism verified in
run_sparkov_streaming_demo.py, now behind a real endpoint rather than a batch replay script.
"""

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.data.ingest_sparkov import FEATURE_COLUMNS, raw_row_to_static_features
from src.serving.schemas import PredictionResponse
from src.serving.sparkov_model_loader import load_sparkov_production_pipeline
from src.streaming.redis_features_sparkov import card_velocity_features, get_redis_client

state: dict = {"pipeline": None, "metadata": None, "redis": None}


class SparkovTransactionRequest(BaseModel):
    cc_num: str
    trans_date_trans_time: str
    dob: str
    amt: float
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int
    gender: str
    category: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"], state["metadata"] = load_sparkov_production_pipeline()
    state["redis"] = get_redis_client()
    yield


app = FastAPI(title="Sparkov Fraud Detection Inference Service", lifespan=lifespan)


@app.get("/health")
def health():
    metadata = state["metadata"]
    return {"status": "ok", "threshold": metadata["optimal_threshold"] if metadata else None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: SparkovTransactionRequest):
    start = time.perf_counter()

    static = raw_row_to_static_features(transaction.model_dump())
    now_ms = int(pd.to_datetime(transaction.trans_date_trans_time).timestamp() * 1000)
    velocity = card_velocity_features(state["redis"], transaction.cc_num, transaction.amt, now_ms=now_ms)

    row = pd.DataFrame([{**static, **velocity}])[FEATURE_COLUMNS]
    proba = float(state["pipeline"].predict_proba(row)[0, 1])
    threshold = state["metadata"]["optimal_threshold"]
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= threshold,
        threshold=threshold,
        latency_ms=latency_ms,
    )

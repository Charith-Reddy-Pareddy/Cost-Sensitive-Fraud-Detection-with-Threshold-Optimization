"""FastAPI inference service.

`/predict` scores a single transaction on demand. `/replay` replays a batch of real test-set
rows with a small delay between each — simulating production traffic — scoring every one and
recording its latency. `/latency` reports p50/p95 over everything scored so far.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from src.features.pipeline import RAW_FEATURE_COLUMNS
from src.serving.latency_tracker import LatencyTracker
from src.serving.model_loader import load_production_pipeline
from src.serving.schemas import PredictionResponse, TransactionRequest

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

latency_tracker = LatencyTracker()
state: dict = {"pipeline": None, "metadata": None, "test_df": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"], state["metadata"] = load_production_pipeline()
    test_path = PROCESSED_DIR / "test.parquet"
    if test_path.exists():
        state["test_df"] = pd.read_parquet(test_path)
    yield


app = FastAPI(title="Fraud Detection Inference Service", lifespan=lifespan)


def _row_to_frame(values: dict) -> pd.DataFrame:
    """Build the model input by feature name, in the training-time column order — never by
    positional/insertion order, which is what makes this immune to field-ordering drift."""
    return pd.DataFrame([{col: values[col] for col in RAW_FEATURE_COLUMNS}])


def _score(row: pd.DataFrame) -> tuple[float, float]:
    start = time.perf_counter()
    proba = float(state["pipeline"].predict_proba(row[RAW_FEATURE_COLUMNS])[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    latency_tracker.record(latency_ms)
    return proba, latency_ms


@app.get("/health")
def health():
    metadata = state["metadata"]
    return {"status": "ok", "threshold": metadata["optimal_threshold"] if metadata else None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    row = _row_to_frame(transaction.model_dump())
    proba, latency_ms = _score(row)
    threshold = state["metadata"]["optimal_threshold"]
    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=proba >= threshold,
        threshold=threshold,
        latency_ms=latency_ms,
    )


@app.post("/replay")
def replay(n: int = 100, delay_ms: float = 5.0, seed: int | None = None):
    test_df = state["test_df"]
    if test_df is None:
        return {"error": "test set not available"}

    sample = test_df.sample(n=min(n, len(test_df)), random_state=seed)
    threshold = state["metadata"]["optimal_threshold"]
    scored = 0
    for _, row in sample.iterrows():
        row_frame = pd.DataFrame([row[RAW_FEATURE_COLUMNS]])
        proba, _ = _score(row_frame)
        scored += 1
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    return {"n_scored": scored, "threshold": threshold}


@app.get("/latency")
def latency_stats():
    return latency_tracker.percentiles()


@app.get("/latency/raw")
def latency_raw():
    return {"samples_ms": latency_tracker.snapshot()}

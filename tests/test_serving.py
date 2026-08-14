import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.pipeline import RAW_FEATURE_COLUMNS
from src.serving.app import app
from src.serving.model_loader import ARTIFACTS_DIR

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "production_pipeline.joblib").exists(),
    reason="requires `python -m src.models.train_production_model` to have been run first",
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_matches_offline_batch_scoring(client):
    """The exact test that would catch a feature-order mismatch between training and
    serving: the API's prediction for a real row must match the pipeline scoring that same
    row directly, offline."""
    from src.serving.model_loader import load_production_pipeline

    pipeline, metadata = load_production_pipeline()

    test_df = pd.read_parquet(ARTIFACTS_DIR.parent.parent / "data" / "processed" / "test.parquet")
    row = test_df.iloc[0]

    payload = {col: float(row[col]) for col in RAW_FEATURE_COLUMNS}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    served_proba = resp.json()["fraud_probability"]
    offline_proba = float(pipeline.predict_proba(row[RAW_FEATURE_COLUMNS].to_frame().T)[0, 1])

    assert served_proba == pytest.approx(offline_proba, abs=1e-6)


def test_predict_applies_threshold_consistently(client):
    resp = client.post("/predict", json={col: 0.0 for col in RAW_FEATURE_COLUMNS})
    body = resp.json()
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])


def test_replay_scores_requested_count(client):
    resp = client.post("/replay", params={"n": 5, "delay_ms": 0})
    assert resp.status_code == 200
    assert resp.json()["n_scored"] == 5


def test_latency_reports_percentiles_after_predictions(client):
    client.post("/predict", json={col: 0.0 for col in RAW_FEATURE_COLUMNS})
    resp = client.get("/latency")
    body = resp.json()
    assert body["count"] >= 1
    assert body["p50_ms"] is not None
    assert body["p95_ms"] is not None

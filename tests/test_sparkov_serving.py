import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.ingest_sparkov import FEATURE_COLUMNS, raw_row_to_static_features
from src.serving.sparkov_app import app
from src.serving.sparkov_model_loader import ARTIFACTS_DIR, load_sparkov_production_pipeline
from src.streaming.redis_features_sparkov import card_velocity_features, get_redis_client


def _redis_available() -> bool:
    try:
        get_redis_client().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "sparkov_production_pipeline.joblib").exists() or not _redis_available(),
    reason="requires `python -m src.models.train_sparkov_production_model` and a running redis",
)

SAMPLE_TRANSACTION = {
    "cc_num": "TEST_CARD_SERVING_1",
    "trans_date_trans_time": "2020-06-15 14:30:00",
    "dob": "1985-03-20",
    "amt": 123.45,
    "lat": 40.0,
    "long": -74.0,
    "merch_lat": 40.1,
    "merch_long": -74.2,
    "city_pop": 50000,
    "gender": "F",
    "category": "grocery_pos",
}


@pytest.fixture()
def client():
    # flush this test card's window so the test is deterministic regardless of prior runs
    get_redis_client().delete(f"card_window:{SAMPLE_TRANSACTION['cc_num']}")
    with TestClient(app) as c:
        yield c
    get_redis_client().delete(f"card_window:{SAMPLE_TRANSACTION['cc_num']}")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_matches_offline_scoring_for_first_transaction(client):
    """A fresh card's first transaction has no prior Redis state, so live velocity should be
    exactly zero — matching what an offline row with zero velocity would score. This is the
    Sparkov-serving version of the exact test that catches a feature mismatch between training
    and serving on the primary dataset."""
    pipeline, _ = load_sparkov_production_pipeline()

    resp = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert resp.status_code == 200
    served_proba = resp.json()["fraud_probability"]

    static = raw_row_to_static_features(SAMPLE_TRANSACTION)
    offline_row = pd.DataFrame([{**static, "card_txn_count_24h": 0, "card_amt_sum_24h": 0.0}])[FEATURE_COLUMNS]
    offline_proba = float(pipeline.predict_proba(offline_row)[0, 1])

    assert served_proba == pytest.approx(offline_proba, abs=1e-6)


def test_predict_applies_threshold_consistently(client):
    resp = client.post("/predict", json=SAMPLE_TRANSACTION)
    body = resp.json()
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])


def test_repeated_transactions_increase_live_velocity(client):
    """Sending the same card twice should make the next lookup reflect the first — the whole
    point of wiring Redis into serving instead of just logging it alongside a prediction."""
    client.post("/predict", json=SAMPLE_TRANSACTION)

    now_ms = int(pd.to_datetime(SAMPLE_TRANSACTION["trans_date_trans_time"]).timestamp() * 1000) + 1000
    velocity = card_velocity_features(
        get_redis_client(), SAMPLE_TRANSACTION["cc_num"], SAMPLE_TRANSACTION["amt"], now_ms=now_ms
    )
    assert velocity["card_txn_count_24h"] == 1

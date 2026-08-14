"""Loads the persisted production pipeline and its threshold/feature-column metadata —
produced by `python -m src.models.train_production_model`."""

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"


def load_production_pipeline() -> tuple[Pipeline, dict]:
    pipeline = joblib.load(ARTIFACTS_DIR / "production_pipeline.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "production_metadata.json").read_text())
    return pipeline, metadata

import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS
from src.models.train_baseline import build_logistic_regression_pipeline, evaluate


def _synthetic_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in RAW_FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    y = (df["V1"] + rng.normal(scale=0.1, size=n) > 0.5).astype(int)
    return df, y


def test_logistic_regression_pipeline_fits_and_predicts():
    X, y = _synthetic_frame()
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_expected_metrics():
    X, y = _synthetic_frame()
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X, y)
    metrics = evaluate(pipeline, X, y)
    assert set(metrics) == {"pr_auc", "roc_auc", "f1_at_0.5"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

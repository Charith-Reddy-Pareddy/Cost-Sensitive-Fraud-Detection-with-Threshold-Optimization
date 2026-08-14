import numpy as np
import pandas as pd

from src.features.pipeline import PIPELINE_OUTPUT_COLUMNS, RAW_FEATURE_COLUMNS
from src.models.imbalance_comparison import build_pipeline
from src.models.interpretability import compute_shap_values, mean_abs_shap_by_feature


def _synthetic_frame(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in RAW_FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    # V5 is the only feature that actually drives the label -> SHAP should surface it as
    # the dominant feature
    y = (df["V5"] > 0.0).astype(int)
    return df, y


def test_shap_surfaces_the_informative_feature():
    X, y = _synthetic_frame()
    pipeline = build_pipeline("none")
    pipeline.fit(X, y)

    shap_values, X_transformed = compute_shap_values(pipeline, X)
    assert X_transformed.shape == (len(X), len(PIPELINE_OUTPUT_COLUMNS))

    importance = mean_abs_shap_by_feature(shap_values, PIPELINE_OUTPUT_COLUMNS)
    assert list(importance.index)[0] == "V5"
    # sorted descending
    assert (importance.values[:-1] >= importance.values[1:]).all()

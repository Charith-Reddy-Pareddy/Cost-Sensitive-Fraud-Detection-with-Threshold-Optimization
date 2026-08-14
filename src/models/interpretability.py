"""Model interpretability pipeline: SHAP values on top of the anonymized PCA components.

Even though `V1`..`V28` carry no semantic meaning on their own (they're PCA components, not
named business features), SHAP still tells us which of those components the model actually
relies on — which is worth relating to published analyses of this dataset, where a handful of
components (commonly V14, V17) are consistently identified as the strongest fraud signal.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.features.pipeline import PIPELINE_OUTPUT_COLUMNS


def compute_shap_values(pipeline: Pipeline, X_raw_sample: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (shap_values, transformed_features) — SHAP needs the classifier and the
    already-preprocessed matrix in the same column order the classifier was trained on."""
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = pd.DataFrame(
        preprocessor.transform(X_raw_sample), columns=PIPELINE_OUTPUT_COLUMNS, index=X_raw_sample.index
    )

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    return shap_values, X_transformed


def mean_abs_shap_by_feature(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    return pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)

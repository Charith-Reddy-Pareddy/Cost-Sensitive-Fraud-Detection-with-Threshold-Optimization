"""Feature preprocessing shared by every model and by the serving path.

`Amount` and `Time` are the only raw-scale columns in this dataset — `V1`..`V28` are already
PCA components (roughly zero-centered). Only `Amount`/`Time` get scaled; the fitted scaler's
statistics come from `fit()` on training data only, so this must live *inside* a pipeline that's
fit per training fold, never fit once on the full dataset ahead of time.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

RAW_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET_COLUMN = "Class"

SCALED_COLUMNS = ["Amount", "Time"]

# ColumnTransformer emits the transformed columns first, then passthrough columns in their
# original relative order. This is the authoritative column order downstream code (models,
# SHAP, the serving service) must agree on.
PIPELINE_OUTPUT_COLUMNS = SCALED_COLUMNS + [f"V{i}" for i in range(1, 29)]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), SCALED_COLUMNS)],
        remainder="passthrough",
    )

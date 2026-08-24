"""Train and persist the model the serving layer actually loads.

Per the Day 3/4 conclusions, that's the class-weighted XGBoost pipeline, decided at its
cost-optimized threshold rather than 0.5. The threshold is selected on the validation split,
never the test split — test is reserved for a single, final, untouched evaluation (see
run_bootstrap_analysis.py). Both the fitted pipeline and the threshold/feature contract are
saved to disk so the serving service has a single, versioned source of truth instead of
re-deriving them.
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_utils import use_local_tracking_store
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, optimize_threshold
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-production")

    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")

    pipeline = build_pipeline("class_weight")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)

    with mlflow.start_run(run_name="production_model"):
        pipeline.fit(X_train, y_train)

        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        sweep = optimize_threshold(
            y_val.to_numpy(), y_val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
        )

        mlflow.log_param("cost_fn", DEFAULT_COST_FALSE_NEGATIVE)
        mlflow.log_param("cost_fp", DEFAULT_COST_FALSE_POSITIVE)
        mlflow.log_metric("optimal_threshold", sweep.optimal_threshold)
        mlflow.log_metric("optimal_cost_on_val", sweep.optimal_cost)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="cloudpickle")

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ARTIFACTS_DIR / "production_pipeline.joblib")

        metadata = {
            "feature_columns": RAW_FEATURE_COLUMNS,
            "optimal_threshold": sweep.optimal_threshold,
            "cost_fn": DEFAULT_COST_FALSE_NEGATIVE,
            "cost_fp": DEFAULT_COST_FALSE_POSITIVE,
        }
        (ARTIFACTS_DIR / "production_metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"saved production pipeline, threshold={sweep.optimal_threshold:.2f}")


if __name__ == "__main__":
    main()

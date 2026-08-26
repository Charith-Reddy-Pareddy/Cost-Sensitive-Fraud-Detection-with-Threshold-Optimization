"""Train and persist the Sparkov model the serving layer actually loads — the standing-service
counterpart to the primary dataset's train_production_model.py. Threshold selected on val, never
on test, matching the protocol used throughout this project.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, optimize_threshold
from src.models.run_sparkov_bootstrap_analysis import PROCESSED_DIR

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "models" / "artifacts"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def _build_pipeline(scale_pos_weight: float) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline = _build_pipeline(scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    sweep = optimize_threshold(
        y_val.to_numpy(), val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "sparkov_production_pipeline.joblib")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "optimal_threshold": sweep.optimal_threshold,
        "cost_fn": DEFAULT_COST_FALSE_NEGATIVE,
        "cost_fp": DEFAULT_COST_FALSE_POSITIVE,
    }
    (ARTIFACTS_DIR / "sparkov_production_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"saved sparkov production pipeline, threshold={sweep.optimal_threshold:.2f}")


if __name__ == "__main__":
    main()

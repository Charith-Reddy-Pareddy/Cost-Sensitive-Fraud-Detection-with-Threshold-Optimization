"""Generates tiny synthetic data and trained model artifacts so CI can build and smoke-test the
Docker serving images without the real Kaggle datasets or credentials.

Not used for anything except CI verification — every real number in the README and
RESEARCH_REPORT.md comes from the actual training scripts run against the actual datasets. This
script exists purely so a broken Dockerfile, a broken serving contract, or a broken
train/serve feature mismatch gets caught automatically on every push instead of only when
someone happens to test it by hand.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import TARGET_COLUMN as SPARKOV_TARGET_COLUMN
from src.data.ingest_sparkov import engineer_features
from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, optimize_threshold
from src.models.imbalance_comparison import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
N_ROWS = 600


def _make_primary_fixtures() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=N_ROWS) for c in RAW_FEATURE_COLUMNS})
    df[TARGET_COLUMN] = (df["V1"] + rng.normal(scale=0.1, size=N_ROWS) > 1.0).astype(int)

    n_train = int(N_ROWS * 0.7)
    train_df, test_df = df.iloc[:n_train], df.iloc[n_train:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    pipeline = build_pipeline("class_weight")
    n_pos = max(int(train_df[TARGET_COLUMN].sum()), 1)
    n_neg = max(len(train_df) - n_pos, 1)
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline.fit(train_df[RAW_FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    sweep = optimize_threshold(
        test_df[TARGET_COLUMN].to_numpy(),
        pipeline.predict_proba(test_df[RAW_FEATURE_COLUMNS])[:, 1],
        cost_fn=DEFAULT_COST_FALSE_NEGATIVE,
        cost_fp=DEFAULT_COST_FALSE_POSITIVE,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "production_pipeline.joblib")
    metadata = {
        "feature_columns": RAW_FEATURE_COLUMNS,
        "optimal_threshold": sweep.optimal_threshold,
        "cost_fn": DEFAULT_COST_FALSE_NEGATIVE,
        "cost_fp": DEFAULT_COST_FALSE_POSITIVE,
    }
    (ARTIFACTS_DIR / "production_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"primary fixtures: {len(test_df)} test rows, threshold={sweep.optimal_threshold:.2f}")


def _synthetic_sparkov_raw(n: int = N_ROWS, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, size=n)
    dates = (
        pd.Timestamp("2020-01-01")
        + pd.to_timedelta(rng.integers(0, 60, size=n), unit="D")
        + pd.to_timedelta(hours, unit="h")
    )
    dobs = pd.Timestamp("1970-01-01") + pd.to_timedelta(rng.integers(0, 365 * 60, size=n), unit="D")
    amt = rng.uniform(1, 500, size=n)
    return pd.DataFrame(
        {
            "trans_date_trans_time": dates.astype(str),
            # small pool of card numbers so the velocity feature actually gets exercised
            "cc_num": rng.integers(1000, 1020, size=n).astype(str),
            "category": rng.choice(["grocery_pos", "shopping_net", "gas_transport"], size=n),
            "amt": amt,
            "gender": rng.choice(["M", "F"], size=n),
            "lat": rng.uniform(30, 45, size=n),
            "long": rng.uniform(-120, -70, size=n),
            "city_pop": rng.integers(1000, 100000, size=n),
            "dob": dobs.astype(str),
            "unix_time": (dates.astype("int64") // 10**9).astype(int),
            "merch_lat": rng.uniform(30, 45, size=n),
            "merch_long": rng.uniform(-120, -70, size=n),
            "is_fraud": (amt > 400).astype(int),
        }
    )


def _make_sparkov_fixtures() -> None:
    features = engineer_features(_synthetic_sparkov_raw())
    feature_cols = [c for c in features.columns if c not in ("unix_time", SPARKOV_TARGET_COLUMN)]

    n_train = int(len(features) * 0.7)
    train_df, test_df = features.iloc[:n_train], features.iloc[n_train:]

    classifier = XGBClassifier(n_estimators=20, max_depth=3, n_jobs=-1)
    pipeline = Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])
    pipeline.fit(train_df[feature_cols], train_df[SPARKOV_TARGET_COLUMN])

    sweep = optimize_threshold(
        test_df[SPARKOV_TARGET_COLUMN].to_numpy(),
        pipeline.predict_proba(test_df[feature_cols])[:, 1],
        cost_fn=DEFAULT_COST_FALSE_NEGATIVE,
        cost_fp=DEFAULT_COST_FALSE_POSITIVE,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "sparkov_production_pipeline.joblib")
    metadata = {
        "feature_columns": feature_cols,
        "optimal_threshold": sweep.optimal_threshold,
        "cost_fn": DEFAULT_COST_FALSE_NEGATIVE,
        "cost_fp": DEFAULT_COST_FALSE_POSITIVE,
    }
    (ARTIFACTS_DIR / "sparkov_production_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"sparkov fixtures: {len(test_df)} test rows, threshold={sweep.optimal_threshold:.2f}")


def main() -> None:
    _make_primary_fixtures()
    _make_sparkov_fixtures()


if __name__ == "__main__":
    main()

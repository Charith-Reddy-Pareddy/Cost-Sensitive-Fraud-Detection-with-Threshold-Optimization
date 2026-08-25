"""Bootstrap CIs on Sparkov — the primary dataset's "no effect" threshold-optimization finding
came from a single split; this checks whether that null result is itself stable, or just as
fragile as the primary dataset's original (pre-fix) point estimate was.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import TARGET_COLUMN
from src.models.bootstrap_evaluation import bootstrap_ci
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "sparkov"
N_BOOTSTRAP = 1000


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ("Time", TARGET_COLUMN)]
    return df[feature_cols], df[TARGET_COLUMN]


def _build_pipeline(scale_pos_weight: float) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def _precision_at(threshold):
    return lambda y_true, y_proba: precision_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)


def _recall_at(threshold):
    return lambda y_true, y_proba: recall_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)


def _cost_at(threshold):
    return lambda y_true, y_proba: expected_cost(
        y_true, y_proba, threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )


def _cost_reduction_pct(threshold):
    def fn(y_true, y_proba):
        default = expected_cost(y_true, y_proba, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        optimal = expected_cost(y_true, y_proba, threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        return 0.0 if default == 0 else 100 * (default - optimal) / default

    return fn


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline = _build_pipeline(scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    sweep = optimize_threshold(y_val.to_numpy(), val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    threshold = sweep.optimal_threshold

    y_test_arr = y_test.to_numpy()
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "pr_auc": average_precision_score,
        f"precision_at_{threshold:.2f}": _precision_at(threshold),
        f"recall_at_{threshold:.2f}": _recall_at(threshold),
        f"expected_cost_at_{threshold:.2f}": _cost_at(threshold),
        "cost_reduction_pct_vs_default": _cost_reduction_pct(threshold),
    }

    print(f"threshold selected on val: {threshold:.2f}")
    print(f"{'metric':32} {'estimate':>12} {'95% CI':>24}")
    for name, fn in metrics.items():
        result = bootstrap_ci(y_test_arr, test_proba, fn, n_bootstrap=N_BOOTSTRAP)
        print(f"{name:32} {result.point_estimate:12.4f} [{result.lower:10.4f}, {result.upper:10.4f}]")


if __name__ == "__main__":
    main()

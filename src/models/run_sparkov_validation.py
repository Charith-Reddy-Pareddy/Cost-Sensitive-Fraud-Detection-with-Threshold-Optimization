"""External validation: do this project's two core conclusions replicate on a second,
structurally different dataset?

1. Does class-weighting beat the unweighted baseline?
2. Does a validation-selected cost-optimal threshold beat the default 0.5 on held-out test?

Same train/val/test protocol, same illustrative $500/$5 costs, but a dataset with real
merchant/category/geolocation fields and engineered features (distance, age, hour) instead of
anonymized PCA components — this is the generalization check the primary dataset alone can't
provide.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "sparkov"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ("Time", TARGET_COLUMN)]
    return df[feature_cols], df[TARGET_COLUMN]


def _build_pipeline(scale_pos_weight: float | None = None) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    print(f"Sparkov: train={len(X_train)} ({y_train.sum()} fraud), "
          f"val={len(X_val)} ({y_val.sum()} fraud), test={len(X_test)} ({y_test.sum()} fraud)")
    print(f"features: {list(X_train.columns)}\n")

    # 1: does class-weighting beat unweighted, on this dataset?
    pipeline_none = _build_pipeline()
    pipeline_none.fit(X_train, y_train)
    proba_none_test = pipeline_none.predict_proba(X_test)[:, 1]

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline_cw = _build_pipeline(scale_pos_weight=n_neg / n_pos)
    pipeline_cw.fit(X_train, y_train)
    proba_cw_val = pipeline_cw.predict_proba(X_val)[:, 1]
    proba_cw_test = pipeline_cw.predict_proba(X_test)[:, 1]

    print("## Imbalance handling\n")
    print("| Strategy | PR-AUC | ROC-AUC |")
    print("|---|---|---|")
    print(f"| None | {average_precision_score(y_test_arr, proba_none_test):.3f} | {roc_auc_score(y_test_arr, proba_none_test):.3f} |")
    print(f"| Class weighting | {average_precision_score(y_test_arr, proba_cw_test):.3f} | {roc_auc_score(y_test_arr, proba_cw_test):.3f} |")

    # 2: does the val-selected cost-optimal threshold beat default 0.5, on test?
    sweep = optimize_threshold(y_val_arr, proba_cw_val, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    default_cost = expected_cost(y_test_arr, proba_cw_test, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    optimal_cost = expected_cost(
        y_test_arr, proba_cw_test, sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    reduction_pct = 100 * (default_cost - optimal_cost) / default_cost

    print("\n## Cost-sensitive threshold optimization\n")
    print(f"val-selected threshold: {sweep.optimal_threshold:.2f}")
    print(f"default (0.50) cost on test: ${default_cost:,.2f}")
    print(f"optimized cost on test: ${optimal_cost:,.2f}")
    print(f"cost reduction vs. default: {reduction_pct:.1f}%")
    print(f"optimized threshold beats default: {optimal_cost <= default_cost}")


if __name__ == "__main__":
    main()

"""Repeats Experiment 4 (training-objective vs. decision-policy) on Sparkov — the primary
dataset found cost-weighted training alone beats threshold tuning alone or combined; this checks
whether that holds on a dataset where threshold tuning itself has already been shown not to help
(see run_sparkov_validation.py / run_sparkov_bootstrap_analysis.py).

A. standard training, threshold 0.5
B. standard training, optimized threshold (selected on val)
C. cost-weighted training (sample_weight = $500/$5 per row), threshold 0.5
D. cost-weighted training, optimized threshold (selected on val)
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold
from src.models.cost_sensitive_training import cost_sample_weights
from src.models.evaluation import evaluate_full
from src.models.run_sparkov_bootstrap_analysis import PROCESSED_DIR


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ("Time", TARGET_COLUMN)]
    return df[feature_cols], df[TARGET_COLUMN]


def _build_pipeline() -> Pipeline:
    classifier = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1)
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    pipeline_standard = _build_pipeline()
    pipeline_standard.fit(X_train, y_train)
    val_proba_standard = pipeline_standard.predict_proba(X_val)[:, 1]
    test_proba_standard = pipeline_standard.predict_proba(X_test)[:, 1]

    pipeline_cost = _build_pipeline()
    weights = cost_sample_weights(y_train.to_numpy(), DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    pipeline_cost.fit(X_train, y_train, classifier__sample_weight=weights)
    val_proba_cost = pipeline_cost.predict_proba(X_val)[:, 1]
    test_proba_cost = pipeline_cost.predict_proba(X_test)[:, 1]

    def _row(name: str, test_proba, threshold: float) -> tuple:
        m = evaluate_full(y_test_arr, test_proba, threshold=threshold)
        cost = expected_cost(y_test_arr, test_proba, threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        return name, threshold, m["pr_auc"], m["recall"], cost

    sweep_standard = optimize_threshold(
        y_val_arr, val_proba_standard, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
    )
    sweep_cost = optimize_threshold(
        y_val_arr, val_proba_cost, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
    )

    rows = [
        _row("A: standard training, threshold 0.5", test_proba_standard, 0.5),
        _row(
            f"B: standard training, optimized threshold ({sweep_standard.optimal_threshold:.2f})",
            test_proba_standard,
            sweep_standard.optimal_threshold,
        ),
        _row("C: cost-weighted training, threshold 0.5", test_proba_cost, 0.5),
        _row(
            f"D: cost-weighted training, optimized threshold ({sweep_cost.optimal_threshold:.2f})",
            test_proba_cost,
            sweep_cost.optimal_threshold,
        ),
    ]

    print("| Configuration | Threshold | PR-AUC | Recall | Expected cost |")
    print("|---|---|---|---|---|")
    for name, threshold, pr_auc, recall, cost in rows:
        print(f"| {name} | {threshold:.2f} | {pr_auc:.3f} | {recall:.3f} | ${cost:,.2f} |")


if __name__ == "__main__":
    main()

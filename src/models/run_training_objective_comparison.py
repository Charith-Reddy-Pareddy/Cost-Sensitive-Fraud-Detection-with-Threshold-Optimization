"""Training-objective vs. decision-policy: does changing *how the model is trained* (cost-
weighted sample weights) add anything beyond changing *how its output is used* (threshold
tuning)?

Four configurations, one train/val/test split, threshold selected on val, everything reported
on the untouched test split:

A. Standard XGBoost, default threshold 0.5 (no training-time or decision-time adjustment)
B. Standard XGBoost, cost-optimized threshold (decision-policy adjustment only)
C. Cost-weighted training (sample_weight = $500/$5 per row), default threshold 0.5
   (training-objective adjustment only)
D. Cost-weighted training + cost-optimized threshold (both)
"""

from pathlib import Path

import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold
from src.models.cost_sensitive_training import cost_sample_weights
from src.models.evaluation import evaluate_full
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    # standard training
    pipeline_standard = build_pipeline("none")
    pipeline_standard.fit(X_train, y_train)
    val_proba_standard = pipeline_standard.predict_proba(X_val)[:, 1]
    test_proba_standard = pipeline_standard.predict_proba(X_test)[:, 1]

    # cost-weighted training: sample_weight is a fit-time argument, not a pipeline hyperparameter
    pipeline_cost = build_pipeline("none")
    weights = cost_sample_weights(y_train.to_numpy(), DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    pipeline_cost.fit(X_train, y_train, classifier__sample_weight=weights)
    val_proba_cost = pipeline_cost.predict_proba(X_val)[:, 1]
    test_proba_cost = pipeline_cost.predict_proba(X_test)[:, 1]

    def _row(name: str, val_proba, test_proba, threshold: float) -> tuple:
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
        _row("A: standard training, threshold 0.5", val_proba_standard, test_proba_standard, 0.5),
        _row(
            f"B: standard training, optimized threshold ({sweep_standard.optimal_threshold:.2f})",
            val_proba_standard,
            test_proba_standard,
            sweep_standard.optimal_threshold,
        ),
        _row("C: cost-weighted training, threshold 0.5", val_proba_cost, test_proba_cost, 0.5),
        _row(
            f"D: cost-weighted training, optimized threshold ({sweep_cost.optimal_threshold:.2f})",
            val_proba_cost,
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

"""Paired bootstrap: is "cost-weighted training beats cost-weighted training + threshold tuning"
(Experiment 4's C vs. D, RESEARCH_REPORT.md) a real, stable difference, or noise from a single
57k-row test split?

C and D use the exact same cost-weighted model and the exact same test rows — they differ only
in which threshold is applied (0.5 vs. the val-selected optimum) — which makes this a genuinely
*paired* comparison: the right tool is `paired_bootstrap_comparison`, which resamples both
configurations on the same bootstrap draw each time, rather than bootstrapping each independently
and comparing intervals that would ignore how correlated the two configurations' fates already
are on any given resample.
"""

from pathlib import Path

import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.bootstrap_evaluation import paired_bootstrap_comparison
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    expected_cost,
    optimize_threshold,
)
from src.models.cost_sensitive_training import cost_sample_weights
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
N_BOOTSTRAP = 1000


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def _cost_fn(threshold: float):
    return lambda y_true, y_proba: expected_cost(
        y_true, y_proba, threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    pipeline_cost = build_pipeline("none")
    weights = cost_sample_weights(y_train.to_numpy(), DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    pipeline_cost.fit(X_train, y_train, classifier__sample_weight=weights)
    val_proba = pipeline_cost.predict_proba(X_val)[:, 1]
    test_proba = pipeline_cost.predict_proba(X_test)[:, 1]

    sweep = optimize_threshold(y_val_arr, val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    tuned_threshold = sweep.optimal_threshold

    # A = config C (cost-weighted training, threshold 0.5), B = config D (+ tuned threshold);
    # same model, same test rows, same scores — only the threshold differs.
    result = paired_bootstrap_comparison(
        y_test_arr,
        test_proba,
        test_proba,
        _cost_fn(0.5),
        _cost_fn(tuned_threshold),
        lower_is_better=True,
        n_bootstrap=N_BOOTSTRAP,
    )

    print("Config C: cost-weighted training, threshold 0.5")
    print(f"Config D: cost-weighted training, tuned threshold {tuned_threshold:.4f} (selected on val)\n")
    print(f"Observed cost, C: ${_cost_fn(0.5)(y_test_arr, test_proba):,.2f}")
    print(f"Observed cost, D: ${_cost_fn(tuned_threshold)(y_test_arr, test_proba):,.2f}")
    print(f"Observed diff (C - D): ${result.point_diff:,.2f}  (negative means C is cheaper, i.e. C beats D)\n")
    print(f"Paired bootstrap ({N_BOOTSTRAP} resamples):")
    print(f"  mean diff (C - D):   ${result.mean_diff:,.2f}")
    print(f"  median diff (C - D): ${result.median_diff:,.2f}")
    print(f"  95% CI:              [${result.lower:,.2f}, ${result.upper:,.2f}]")
    print(f"  P(C beats D):        {result.prob_a_better:.3f}")


if __name__ == "__main__":
    main()

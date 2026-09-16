"""Compare threshold-search strategies: a 101-point grid, a 1001-point grid, the exact O(n log n)
empirical search, and the closed-form Bayes-optimal threshold.

The grid used everywhere else in this project (`np.linspace(0, 1, 101)`, i.e. a step of 0.01) is
a discretization choice, not a fundamental limit — the true cost-minimizing threshold can fall
between grid points. This script quantifies how much that matters in practice, and separately
reports the theoretical Bayes-optimal threshold (`cost_fp / (cost_fn + cost_fp)`, valid under
perfectly calibrated probabilities) so it can be compared against what the raw, uncalibrated
XGBoost scores actually produce empirically.

Threshold selection happens on validation; costs are reported on the untouched test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    bayes_optimal_threshold,
    exact_optimal_threshold,
    expected_cost,
    optimize_threshold,
)
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    pipeline = build_pipeline("class_weight")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)

    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_val_arr = y_val.to_numpy()
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    default_cost = expected_cost(
        y_test_arr, y_test_proba, threshold=0.5, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
    )

    strategies = {
        "101-point grid (step 0.01)": optimize_threshold(
            y_val_arr, y_val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
        ),
        "1001-point grid (step 0.001)": optimize_threshold(
            y_val_arr,
            y_val_proba,
            cost_fn=DEFAULT_COST_FALSE_NEGATIVE,
            cost_fp=DEFAULT_COST_FALSE_POSITIVE,
            thresholds=np.linspace(0.0, 1.0, 1001),
        ),
        "exact empirical (every observed score)": exact_optimal_threshold(
            y_val_arr, y_val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
        ),
    }

    print("Selected on validation, reported on untouched test:\n")
    print("| Strategy | Threshold (val) | Val cost | Test cost | Test cost reduction vs. default |")
    print("|---|---|---|---|---|")
    for name, sweep in strategies.items():
        test_cost = expected_cost(
            y_test_arr, y_test_proba, sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
        )
        reduction_pct = 100 * (default_cost - test_cost) / default_cost
        print(
            f"| {name} | {sweep.optimal_threshold:.4f} | ${sweep.optimal_cost:,.2f} | "
            f"${test_cost:,.2f} | {reduction_pct:+.2f}% |"
        )

    bayes_threshold = bayes_optimal_threshold(cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    bayes_test_cost = expected_cost(
        y_test_arr, y_test_proba, bayes_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    bayes_reduction_pct = 100 * (default_cost - bayes_test_cost) / default_cost
    print(
        f"| Bayes-optimal (theoretical, cost_fp/(cost_fn+cost_fp)) | {bayes_threshold:.4f} | — | "
        f"${bayes_test_cost:,.2f} | {bayes_reduction_pct:+.2f}% |"
    )

    exact_sweep = strategies["exact empirical (every observed score)"]
    grid_101 = strategies["101-point grid (step 0.01)"]
    print(
        f"\n1001-point grid vs. exact-empirical val cost gap: "
        f"${strategies['1001-point grid (step 0.001)'].optimal_cost - exact_sweep.optimal_cost:,.2f}"
    )
    print(f"101-point grid vs. exact-empirical val cost gap: ${grid_101.optimal_cost - exact_sweep.optimal_cost:,.2f}")
    print(
        f"empirical optimum ({exact_sweep.optimal_threshold:.4f}) vs. Bayes-optimal "
        f"({bayes_threshold:.4f}) gap: {abs(exact_sweep.optimal_threshold - bayes_threshold):.4f} — "
        "this divergence is expected under a miscalibrated model; see run_calibration_analysis.py."
    )


if __name__ == "__main__":
    main()

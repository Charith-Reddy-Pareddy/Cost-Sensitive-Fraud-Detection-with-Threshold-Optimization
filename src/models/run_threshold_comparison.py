"""Two tables for the README:

1. A fixed set of thresholds compared head-to-head (precision/recall/FP/FN/cost) — makes the
   core threshold-optimization tradeoff concrete instead of abstract.
2. The cost-ratio sensitivity sweep enriched with precision/recall/FPR at each ratio's optimal
   threshold, not just the threshold value — makes the whole sweep the primary artifact rather
   than the single $500/$5 scenario being the headline.

Every threshold that's *selected* (the cost-optimal one, and each ratio's optimum in the sweep)
is chosen on the validation split; the precision/recall/FP/FN/cost columns are always computed
on the untouched test split, so nothing reported here is fit to the set it's reported on.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    cost_ratio_sensitivity_sweep,
    expected_cost,
)
from src.models.evaluation import evaluate_full
from src.serving.model_loader import load_production_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SENSITIVITY_RATIOS = np.array([1, 5, 10, 25, 50, 100, 250, 500])


def main() -> None:
    pipeline, metadata = load_production_pipeline()

    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    X_val, y_val = val_df[RAW_FEATURE_COLUMNS], val_df[TARGET_COLUMN].to_numpy()
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    fixed_thresholds = sorted({0.50, 0.20, round(metadata["optimal_threshold"], 2), 0.05, 0.02})

    print("## Fixed threshold comparison (reported on test)\n")
    print("| Threshold | Precision | Recall | FP | FN | Expected cost |")
    print("|---|---|---|---|---|---|")
    for t in fixed_thresholds:
        m = evaluate_full(y_test, y_test_proba, threshold=t)
        cost = expected_cost(y_test, y_test_proba, t, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        marker = " (cost-optimal, selected on val)" if abs(t - metadata["optimal_threshold"]) < 1e-9 else ""
        print(
            f"| {t:.2f}{marker} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['false_positives']} | {m['false_negatives']} | ${cost:,.2f} |"
        )

    print("\n## Cost-ratio sensitivity sweep (thresholds selected on val, reported on test)\n")
    print("| Cost ratio | Optimal threshold | Precision | Recall | FPR |")
    print("|---|---|---|---|---|")
    results = cost_ratio_sensitivity_sweep(y_val, y_val_proba, cost_ratios=SENSITIVITY_RATIOS)
    for r in results:
        m = evaluate_full(y_test, y_test_proba, threshold=r["optimal_threshold"])
        print(
            f"| {r['cost_ratio']:.0f} | {r['optimal_threshold']:.2f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['false_positive_rate']:.4f} |"
        )


if __name__ == "__main__":
    main()

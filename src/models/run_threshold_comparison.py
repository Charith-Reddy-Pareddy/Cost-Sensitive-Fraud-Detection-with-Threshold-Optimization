"""Two tables for the README:

1. A fixed set of thresholds compared head-to-head (precision/recall/FP/FN/cost) — makes the
   core threshold-optimization tradeoff concrete instead of abstract.
2. The cost-ratio sensitivity sweep enriched with precision/recall/FPR at each ratio's optimal
   threshold, not just the threshold value — makes the whole sweep the primary artifact rather
   than the single $500/$5 scenario being the headline.
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
FIXED_THRESHOLDS = [0.50, 0.20, 0.08, 0.05, 0.02]
SENSITIVITY_RATIOS = np.array([1, 5, 10, 25, 50, 100, 250, 500])


def main() -> None:
    pipeline, metadata = load_production_pipeline()
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("## Fixed threshold comparison\n")
    print("| Threshold | Precision | Recall | FP | FN | Expected cost |")
    print("|---|---|---|---|---|---|")
    for t in FIXED_THRESHOLDS:
        m = evaluate_full(y_test, y_proba, threshold=t)
        cost = expected_cost(y_test, y_proba, t, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        marker = " (cost-optimal)" if abs(t - metadata["optimal_threshold"]) < 1e-9 else ""
        print(
            f"| {t:.2f}{marker} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['false_positives']} | {m['false_negatives']} | ${cost:,.2f} |"
        )

    print("\n## Cost-ratio sensitivity sweep (enriched)\n")
    print("| Cost ratio | Optimal threshold | Precision | Recall | FPR |")
    print("|---|---|---|---|---|")
    results = cost_ratio_sensitivity_sweep(y_test, y_proba, cost_ratios=SENSITIVITY_RATIOS)
    for r in results:
        m = evaluate_full(y_test, y_proba, threshold=r["optimal_threshold"])
        print(
            f"| {r['cost_ratio']:.0f} | {r['optimal_threshold']:.2f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['false_positive_rate']:.4f} |"
        )


if __name__ == "__main__":
    main()

"""Exports two datasets as JSON for the interactive explorers on the GitHub Pages dashboard
(docs/index.html):

1. The full 101-point threshold sweep (precision/recall/FP/FN/cost at every threshold, on the
   real, untouched test set) — for the threshold explorer.
2. A cost-ratio sweep (optimal threshold selected on val, metrics reported on test, at each of
   50 log-spaced cost ratios) — for the cost-ratio explorer, matching the val-select/test-report
   protocol used everywhere else in this project.
"""

import json
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
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
N_COST_RATIO_POINTS = 50


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def _export_threshold_sweep(pipeline, metadata, X_test, y_test, y_proba) -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        m = evaluate_full(y_test, y_proba, threshold=float(t))
        cost = expected_cost(y_test, y_proba, float(t), DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "fp": m["false_positives"],
                "fn": m["false_negatives"],
                "tp": m["true_positives"],
                "cost": round(cost, 2),
            }
        )

    payload = {
        "optimal_threshold": round(metadata["optimal_threshold"], 2),
        "cost_fn": DEFAULT_COST_FALSE_NEGATIVE,
        "cost_fp": DEFAULT_COST_FALSE_POSITIVE,
        "n_test": len(y_test),
        "n_fraud": int(y_test.sum()),
        "sweep": rows,
    }

    out_path = DOCS_DIR / "threshold_sweep_data.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"exported {len(rows)} threshold points to {out_path}")


def _export_cost_ratio_sweep(X_val, y_val, X_test, y_test, pipeline) -> None:
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    ratios = np.logspace(0, np.log10(500), N_COST_RATIO_POINTS)
    val_results = cost_ratio_sensitivity_sweep(y_val.to_numpy(), y_val_proba, cost_ratios=ratios)

    rows = []
    for r in val_results:
        threshold = r["optimal_threshold"]
        m = evaluate_full(y_test, y_test_proba, threshold=threshold)
        cost = expected_cost(y_test, y_test_proba, threshold, r["cost_fn"], r["cost_fp"])
        rows.append(
            {
                "ratio": round(r["cost_ratio"], 1),
                "threshold": round(threshold, 2),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "fpr": round(m["false_positive_rate"], 4),
                "cost": round(cost, 2),
            }
        )

    payload = {"sweep": rows}
    out_path = DOCS_DIR / "cost_ratio_sweep_data.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"exported {len(rows)} cost-ratio points to {out_path}")


def main() -> None:
    pipeline, metadata = load_production_pipeline()
    X_val, y_val = _load_split("val")
    X_test, y_test_s = _load_split("test")
    y_test = y_test_s.to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _export_threshold_sweep(pipeline, metadata, X_test, y_test, y_proba)
    _export_cost_ratio_sweep(X_val, y_val, X_test, y_test, pipeline)


if __name__ == "__main__":
    main()

"""Exports the full 101-point threshold sweep (precision/recall/FP/FN/cost at every threshold,
on the real, untouched test set) as JSON, for the interactive threshold explorer on the
GitHub Pages dashboard (docs/index.html).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost
from src.models.evaluation import evaluate_full
from src.serving.model_loader import load_production_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def main() -> None:
    pipeline, metadata = load_production_pipeline()
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]

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

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "threshold_sweep_data.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"exported {len(rows)} threshold points to {out_path}")


if __name__ == "__main__":
    main()

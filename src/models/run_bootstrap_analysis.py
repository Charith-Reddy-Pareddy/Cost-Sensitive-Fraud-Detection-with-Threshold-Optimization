"""Bootstrap confidence intervals for the headline numbers reported elsewhere in this project,
computed on the real production pipeline's predictions over the real test set."""

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_utils import use_local_tracking_store
from src.models.bootstrap_evaluation import bootstrap_ci
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost
from src.serving.model_loader import load_production_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
N_BOOTSTRAP = 1000


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
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-bootstrap")

    pipeline, metadata = load_production_pipeline()
    threshold = metadata["optimal_threshold"]

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "pr_auc": average_precision_score,
        f"precision_at_{threshold:.2f}": _precision_at(threshold),
        f"recall_at_{threshold:.2f}": _recall_at(threshold),
        f"expected_cost_at_{threshold:.2f}": _cost_at(threshold),
        "cost_reduction_pct_vs_default": _cost_reduction_pct(threshold),
    }

    with mlflow.start_run(run_name="bootstrap_ci"):
        mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
        mlflow.log_param("threshold", threshold)

        print(f"{'metric':32} {'estimate':>12} {'95% CI':>24}")
        for name, fn in metrics.items():
            result = bootstrap_ci(y_test, y_proba, fn, n_bootstrap=N_BOOTSTRAP)
            mlflow.log_metric(f"{name}_point", result.point_estimate)
            mlflow.log_metric(f"{name}_lower", result.lower)
            mlflow.log_metric(f"{name}_upper", result.upper)
            print(f"{name:32} {result.point_estimate:12.4f} [{result.lower:10.4f}, {result.upper:10.4f}]")


if __name__ == "__main__":
    main()

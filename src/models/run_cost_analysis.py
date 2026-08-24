"""Run the cost-sensitive decision engine against the class-weighted XGBoost model (the Day 3
winner): find the cost-minimizing threshold, plot cost vs. threshold, and sweep the cost ratio
to show how the optimum moves. Logged to MLflow; figures saved for the README.

Threshold selection happens on the validation split only; every reported dollar figure is the
val-selected threshold evaluated on the untouched test split, so these numbers are a genuine
out-of-sample result rather than a threshold chosen to look good on the set it's reported on.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_utils import use_local_tracking_store
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    cost_ratio_sensitivity_sweep,
    expected_cost,
    optimize_threshold,
)
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def _plot_cost_vs_threshold(thresholds: np.ndarray, costs: np.ndarray, optimal_threshold: float, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, costs, label="expected cost")
    plt.axvline(optimal_threshold, color="green", linestyle="--", label=f"optimal = {optimal_threshold:.2f}")
    plt.axvline(0.5, color="red", linestyle=":", label="default = 0.50")
    plt.xlabel("decision threshold")
    plt.ylabel("expected cost ($)")
    plt.title("Expected cost vs. decision threshold")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_sensitivity_sweep(results: list[dict], path: Path) -> None:
    ratios = [r["cost_ratio"] for r in results]
    thresholds = [r["optimal_threshold"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(ratios, thresholds, marker="o")
    plt.xscale("log")
    plt.xlabel("cost ratio (cost of missed fraud / cost of blocked transaction)")
    plt.ylabel("optimal threshold")
    plt.title("Optimal threshold shifts as the cost ratio changes")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-cost-engine")

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

    with mlflow.start_run(run_name="cost_threshold_optimization"):
        mlflow.log_param("cost_fn", DEFAULT_COST_FALSE_NEGATIVE)
        mlflow.log_param("cost_fp", DEFAULT_COST_FALSE_POSITIVE)

        # selection happens on val
        sweep = optimize_threshold(y_val_arr, y_val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)

        # reporting happens on the untouched test set, at the val-selected threshold
        default_cost = expected_cost(
            y_test_arr, y_test_proba, threshold=0.5, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
        )
        optimal_cost_test = expected_cost(
            y_test_arr, y_test_proba, threshold=sweep.optimal_threshold,
            cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE,
        )
        reduction_pct = 100 * (default_cost - optimal_cost_test) / default_cost

        mlflow.log_metrics(
            {
                "default_threshold_cost_test": default_cost,
                "optimal_threshold": sweep.optimal_threshold,
                "optimal_cost_test": optimal_cost_test,
                "cost_reduction_pct_test": reduction_pct,
            }
        )

        cost_plot_path = FIGURES_DIR / "cost_vs_threshold.png"
        _plot_cost_vs_threshold(sweep.thresholds, sweep.costs, sweep.optimal_threshold, cost_plot_path)
        mlflow.log_artifact(str(cost_plot_path))

        ratios = np.array([5, 10, 25, 50, 75, 100, 150, 200, 300, 500])
        sensitivity_results = cost_ratio_sensitivity_sweep(y_val_arr, y_val_proba, cost_ratios=ratios)
        sensitivity_plot_path = FIGURES_DIR / "cost_ratio_sensitivity.png"
        _plot_sensitivity_sweep(sensitivity_results, sensitivity_plot_path)
        mlflow.log_artifact(str(sensitivity_plot_path))

        print("(threshold selected on val, dollar figures evaluated on untouched test)")
        print(f"default (threshold=0.50) cost on test: ${default_cost:,.2f}")
        print(f"val-selected threshold: {sweep.optimal_threshold:.2f}, cost on test: ${optimal_cost_test:,.2f}")
        print(f"cost reduction vs. default (on test): {reduction_pct:.1f}%")
        print("cost-ratio sensitivity sweep (thresholds selected on val):")
        for r in sensitivity_results:
            print(f"  ratio={r['cost_ratio']:.0f} -> optimal_threshold={r['optimal_threshold']:.2f}")


if __name__ == "__main__":
    main()

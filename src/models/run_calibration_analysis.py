"""Does calibration change the cost-optimal threshold?

Trains the class-weighted XGBoost model on the *first* portion of the training period only,
holds out the *last* portion of the training period (chronologically after fit, before test) as
a calibration set, fits Platt scaling and isotonic regression on it, then compares raw vs.
calibrated Brier scores and cost-optimal thresholds on the real, untouched test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.data.ingest import chronological_split
from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_utils import use_local_tracking_store
from src.models.calibration import (
    apply_isotonic,
    apply_platt_scaling,
    brier_score,
    fit_isotonic,
    fit_platt_scaling,
    reliability_curve,
)
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, optimize_threshold
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    use_local_tracking_store()
    mlflow.set_experiment("fraud-detection-calibration")

    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")

    # carve a calibration slice out of the *end* of the training period, so it's still fully
    # chronologically before the test set and never touches it
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    fit_df, calib_df = chronological_split(train_df, test_size=0.15)
    X_fit, y_fit = fit_df[RAW_FEATURE_COLUMNS], fit_df[TARGET_COLUMN]
    X_calib, y_calib = calib_df[RAW_FEATURE_COLUMNS], calib_df[TARGET_COLUMN]

    pipeline = build_pipeline("class_weight")
    n_pos, n_neg = int(y_fit.sum()), len(y_fit) - int(y_fit.sum())
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_fit, y_fit)

    calib_proba = pipeline.predict_proba(X_calib)[:, 1]
    test_proba_raw = pipeline.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.to_numpy()

    platt_model = fit_platt_scaling(calib_proba, y_calib.to_numpy())
    test_proba_platt = apply_platt_scaling(platt_model, test_proba_raw)

    iso_model = fit_isotonic(calib_proba, y_calib.to_numpy())
    test_proba_iso = apply_isotonic(iso_model, test_proba_raw)

    with mlflow.start_run(run_name="calibration_comparison"):
        results = {}
        for name, proba in [("raw", test_proba_raw), ("platt", test_proba_platt), ("isotonic", test_proba_iso)]:
            brier = brier_score(y_test_arr, proba)
            sweep = optimize_threshold(
                y_test_arr, proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
            )
            results[name] = {
                "brier_score": brier,
                "optimal_threshold": sweep.optimal_threshold,
                "optimal_cost": sweep.optimal_cost,
            }
            mlflow.log_metric(f"{name}_brier_score", brier)
            mlflow.log_metric(f"{name}_optimal_threshold", sweep.optimal_threshold)
            mlflow.log_metric(f"{name}_optimal_cost", sweep.optimal_cost)

        # quantile-binning the full, severely-imbalanced test set collapses almost every bin
        # near zero (fraud is 0.17% of rows), which makes the plot uninformative even though
        # the underlying Brier/threshold numbers above are computed correctly on the full set.
        # For the plot only, oversample fraud (same rationale as the Day 5 SHAP sampling) so
        # bins actually spread across the probability range that matters for a decision.
        rng = np.random.default_rng(0)
        fraud_idx = np.where(y_test_arr == 1)[0]
        legit_idx = np.where(y_test_arr == 0)[0]
        legit_sample_idx = rng.choice(legit_idx, size=min(len(fraud_idx) * 20, len(legit_idx)), replace=False)
        plot_idx = np.concatenate([fraud_idx, legit_sample_idx])

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
        for name, proba in [("raw", test_proba_raw), ("platt", test_proba_platt), ("isotonic", test_proba_iso)]:
            prob_true, prob_pred = reliability_curve(y_test_arr[plot_idx], proba[plot_idx], n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label=name)
        plt.xlabel("mean predicted probability (per bin)")
        plt.ylabel("observed fraud rate (per bin)")
        plt.title("Reliability diagram (fraud-oversampled for legibility)")
        plt.legend()
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = FIGURES_DIR / "calibration_reliability.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(plot_path))

        print(f"{'method':10} {'brier':>10} {'threshold':>10} {'cost':>12}")
        for name, r in results.items():
            print(f"{name:10} {r['brier_score']:10.5f} {r['optimal_threshold']:10.3f} {r['optimal_cost']:12,.2f}")


if __name__ == "__main__":
    main()

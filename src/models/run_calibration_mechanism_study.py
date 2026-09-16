"""Calibration as the central question, and why cost-weighted training + threshold tuning
underperforms cost-weighted training alone (Experiment 4 in RESEARCH_REPORT.md).

Five configurations against the theoretical cost threshold:
  1. raw XGBoost (standard training, uncalibrated scores)
  2. Platt-scaled XGBoost (standard training, Platt calibration on val)
  3. isotonic-calibrated XGBoost (standard training, isotonic calibration on val)
  4. cost-weighted XGBoost (sample_weight = dollar cost, uncalibrated scores)
  5. calibrated cost-weighted XGBoost (cost-weighted training, isotonic calibration on val)

The working hypothesis (from Experiment 4's "combining training-time and decision-time
cost-sensitivity is worse than either alone" finding): cost-weighted training doesn't just shift
where the decision boundary should be — it distorts the score away from being a calibrated
P(fraud) estimate in the first place, because the fit is no longer optimizing to match observed
frequencies, it's optimizing a cost-weighted objective. If so, then applying `exact_optimal_threshold`
*on top of* those already-cost-shifted scores double-counts the cost asymmetry: once during
training (via the sample weights) and again during threshold selection (via the cost-weighted
argmin) — each step separately reasonable, but their combination overshoots.

This script tests that directly: Brier score and reliability curves quantify how well-calibrated
each config's scores are, and the empirical-vs-Bayes threshold gap for each config shows how far
its "natural" cost-optimal decision point has drifted from what perfect calibration would predict.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.calibration import apply_isotonic, apply_platt_scaling, brier_score, fit_isotonic, fit_platt_scaling, reliability_curve
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    bayes_optimal_threshold,
    exact_optimal_threshold,
    expected_cost,
)
from src.models.cost_sensitive_training import cost_sample_weights
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def _plot_score_distributions(scores: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(scores), figsize=(6 * len(scores), 4), sharey=True)
    for ax, (name, (proba, y)) in zip(axes, scores.items(), strict=True):
        bins = np.linspace(0, 1, 41)
        ax.hist(proba[y == 0], bins=bins, alpha=0.6, label="legitimate", density=True, color="steelblue")
        ax.hist(proba[y == 1], bins=bins, alpha=0.6, label="fraud", density=True, color="firebrick")
        ax.set_title(name)
        ax.set_xlabel("predicted score")
        ax.legend()
    axes[0].set_ylabel("density")
    fig.suptitle("Predicted score distributions by true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    for name, (prob_true, prob_pred) in curves.items():
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.xlabel("mean predicted score (per bin)")
    plt.ylabel("observed fraud rate (per bin)")
    plt.title("Reliability: standard vs. cost-weighted training")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_cost_surface(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for name, (thresholds, costs) in curves.items():
        plt.plot(thresholds, costs, label=name)
    plt.xlabel("decision threshold")
    plt.ylabel("expected cost ($, validation)")
    plt.title("Cost surface: standard vs. cost-weighted training")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    # standard training
    pipeline_standard = build_pipeline("none")
    pipeline_standard.fit(X_train, y_train)
    val_raw = pipeline_standard.predict_proba(X_val)[:, 1]
    test_raw = pipeline_standard.predict_proba(X_test)[:, 1]

    # cost-weighted training
    pipeline_cost = build_pipeline("none")
    weights = cost_sample_weights(y_train.to_numpy(), DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    pipeline_cost.fit(X_train, y_train, classifier__sample_weight=weights)
    val_cost = pipeline_cost.predict_proba(X_val)[:, 1]
    test_cost = pipeline_cost.predict_proba(X_test)[:, 1]

    platt_model = fit_platt_scaling(val_raw, y_val_arr)
    val_platt, test_platt = apply_platt_scaling(platt_model, val_raw), apply_platt_scaling(platt_model, test_raw)

    iso_model = fit_isotonic(val_raw, y_val_arr)
    val_iso, test_iso = apply_isotonic(iso_model, val_raw), apply_isotonic(iso_model, test_raw)

    iso_model_cw = fit_isotonic(val_cost, y_val_arr)
    val_cost_calib, test_cost_calib = apply_isotonic(iso_model_cw, val_cost), apply_isotonic(iso_model_cw, test_cost)

    configs = [
        ("1. raw (standard training)", val_raw, test_raw),
        ("2. Platt-calibrated (standard training)", val_platt, test_platt),
        ("3. isotonic-calibrated (standard training)", val_iso, test_iso),
        ("4. cost-weighted (uncalibrated)", val_cost, test_cost),
        ("5. cost-weighted + isotonic calibration", val_cost_calib, test_cost_calib),
    ]

    bayes_threshold = bayes_optimal_threshold(DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    default_test_cost = expected_cost(y_test_arr, test_raw, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)

    print(f"Theoretical Bayes-optimal threshold: {bayes_threshold:.4f}\n")
    print("| Configuration | Brier (test) | Empirical threshold (val) | Gap to Bayes | Test cost | vs. default |")
    print("|---|---|---|---|---|---|")
    rows = []
    for name, val_proba, test_proba in configs:
        sweep = exact_optimal_threshold(y_val_arr, val_proba, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        cost_at_optimal = expected_cost(
            y_test_arr, test_proba, sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
        )
        brier = brier_score(y_test_arr, test_proba)
        gap = abs(sweep.optimal_threshold - bayes_threshold)
        reduction_pct = 100 * (default_test_cost - cost_at_optimal) / default_test_cost
        rows.append((name, brier, sweep.optimal_threshold, gap, cost_at_optimal, reduction_pct))
        print(f"| {name} | {brier:.5f} | {sweep.optimal_threshold:.4f} | {gap:.4f} | ${cost_at_optimal:,.2f} | {reduction_pct:+.2f}% |")

    print(f"\nDefault-threshold (0.5) test cost, standard-trained raw scores: ${default_test_cost:,.2f}")
    cost_default_on_cost_weighted = expected_cost(
        y_test_arr, test_cost, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    print(f"Default-threshold (0.5) test cost, cost-weighted-trained raw scores: ${cost_default_on_cost_weighted:,.2f}")

    raw_brier = rows[0][1]
    cost_weighted_brier = rows[3][1]
    print(
        f"\nBrier score, standard training: {raw_brier:.5f}; cost-weighted training: {cost_weighted_brier:.5f} "
        f"({'worse' if cost_weighted_brier > raw_brier else 'better'} calibration under cost-weighting)"
    )

    _plot_score_distributions(
        {
            "standard training": (test_raw, y_test_arr),
            "cost-weighted training": (test_cost, y_test_arr),
        },
        FIGURES_DIR / "score_distributions_standard_vs_cost_weighted.png",
    )

    prob_true_raw, prob_pred_raw = reliability_curve(y_test_arr, test_raw, n_bins=10)
    prob_true_cost, prob_pred_cost = reliability_curve(y_test_arr, test_cost, n_bins=10)
    _plot_reliability(
        {
            "standard training": (prob_true_raw, prob_pred_raw),
            "cost-weighted training": (prob_true_cost, prob_pred_cost),
        },
        FIGURES_DIR / "reliability_standard_vs_cost_weighted.png",
    )

    grid = np.linspace(0.0, 1.0, 101)
    costs_raw = np.array(
        [expected_cost(y_val_arr, val_raw, t, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE) for t in grid]
    )
    costs_cost = np.array(
        [expected_cost(y_val_arr, val_cost, t, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE) for t in grid]
    )
    _plot_cost_surface(
        {
            "standard training": (grid, costs_raw),
            "cost-weighted training": (grid, costs_cost),
        },
        FIGURES_DIR / "cost_surface_standard_vs_cost_weighted.png",
    )


if __name__ == "__main__":
    main()

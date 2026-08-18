"""Ablation study: what does each design decision actually contribute?

Configs 1-3 share one XGBoost fit on the full training set (matching Days 2-4). Config 4 needs
a calibration holdout, so it's fit on the same 85% training slice used in the calibration
analysis — trained on slightly less data than 1-3, noted here rather than glossed over.
"""

from pathlib import Path

import pandas as pd

from src.data.ingest import chronological_split
from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.calibration import apply_isotonic, fit_isotonic
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold
from src.models.evaluation import evaluate_full
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")
    y_test_arr = y_test.to_numpy()

    rows = []

    # 1: no weighting, default threshold
    pipeline_none = build_pipeline("none")
    pipeline_none.fit(X_train, y_train)
    proba_none = pipeline_none.predict_proba(X_test)[:, 1]
    m = evaluate_full(y_test_arr, proba_none, threshold=0.5)
    cost = expected_cost(y_test_arr, proba_none, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    rows.append(("no weighting + threshold 0.5", m["pr_auc"], cost))

    # 2: class weighting, default threshold
    pipeline_cw = build_pipeline("class_weight")
    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline_cw.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline_cw.fit(X_train, y_train)
    proba_cw = pipeline_cw.predict_proba(X_test)[:, 1]
    m = evaluate_full(y_test_arr, proba_cw, threshold=0.5)
    cost = expected_cost(y_test_arr, proba_cw, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    rows.append(("class weighting + threshold 0.5", m["pr_auc"], cost))

    # 3: class weighting, cost-optimized threshold
    sweep = optimize_threshold(y_test_arr, proba_cw, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    m = evaluate_full(y_test_arr, proba_cw, threshold=sweep.optimal_threshold)
    rows.append((f"class weighting + optimized threshold ({sweep.optimal_threshold:.2f})", m["pr_auc"], sweep.optimal_cost))

    # 4: class weighting + isotonic calibration + optimized threshold (needs a calib holdout)
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    fit_df, calib_df = chronological_split(train_df, test_size=0.15)
    X_fit, y_fit = fit_df[RAW_FEATURE_COLUMNS], fit_df[TARGET_COLUMN]
    X_calib, y_calib = calib_df[RAW_FEATURE_COLUMNS], calib_df[TARGET_COLUMN]

    pipeline_calib = build_pipeline("class_weight")
    n_pos_f, n_neg_f = int(y_fit.sum()), len(y_fit) - int(y_fit.sum())
    pipeline_calib.set_params(classifier__scale_pos_weight=n_neg_f / n_pos_f)
    pipeline_calib.fit(X_fit, y_fit)

    calib_proba = pipeline_calib.predict_proba(X_calib)[:, 1]
    iso_model = fit_isotonic(calib_proba, y_calib.to_numpy())
    proba_calibrated = apply_isotonic(iso_model, pipeline_calib.predict_proba(X_test)[:, 1])

    sweep_calib = optimize_threshold(
        y_test_arr, proba_calibrated, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
    )
    m = evaluate_full(y_test_arr, proba_calibrated, threshold=sweep_calib.optimal_threshold)
    rows.append(
        (
            f"class weighting + isotonic calibration + optimized threshold ({sweep_calib.optimal_threshold:.2f})",
            m["pr_auc"],
            sweep_calib.optimal_cost,
        )
    )

    print("| Configuration | PR-AUC | Expected cost |")
    print("|---|---|---|")
    for name, pr_auc, cost in rows:
        print(f"| {name} | {pr_auc:.3f} | ${cost:,.2f} |")


if __name__ == "__main__":
    main()

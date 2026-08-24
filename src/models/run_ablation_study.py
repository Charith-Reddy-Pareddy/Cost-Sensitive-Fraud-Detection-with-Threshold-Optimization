"""Ablation study: what does each design decision actually contribute?

All four configs share one train/val/test split: fit on train, select any threshold or
calibrator on val, report PR-AUC and cost on the untouched test split — the same protocol used
everywhere else in this project, so this table isn't inflated by fitting to the set it's
reported on.
"""

from pathlib import Path

import pandas as pd

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
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr = y_val.to_numpy()
    y_test_arr = y_test.to_numpy()

    rows = []

    # 1: no weighting, default threshold
    pipeline_none = build_pipeline("none")
    pipeline_none.fit(X_train, y_train)
    proba_none_test = pipeline_none.predict_proba(X_test)[:, 1]
    m = evaluate_full(y_test_arr, proba_none_test, threshold=0.5)
    cost = expected_cost(y_test_arr, proba_none_test, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    rows.append(("no weighting + threshold 0.5", m["pr_auc"], cost))

    # 2: class weighting, default threshold
    pipeline_cw = build_pipeline("class_weight")
    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline_cw.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline_cw.fit(X_train, y_train)
    proba_cw_val = pipeline_cw.predict_proba(X_val)[:, 1]
    proba_cw_test = pipeline_cw.predict_proba(X_test)[:, 1]
    m = evaluate_full(y_test_arr, proba_cw_test, threshold=0.5)
    cost = expected_cost(y_test_arr, proba_cw_test, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    rows.append(("class weighting + threshold 0.5", m["pr_auc"], cost))

    # 3: class weighting, cost-optimized threshold (selected on val)
    sweep = optimize_threshold(y_val_arr, proba_cw_val, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    m = evaluate_full(y_test_arr, proba_cw_test, threshold=sweep.optimal_threshold)
    test_cost = expected_cost(
        y_test_arr, proba_cw_test, sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    rows.append((f"class weighting + optimized threshold ({sweep.optimal_threshold:.2f})", m["pr_auc"], test_cost))

    # 4: class weighting + isotonic calibration + optimized threshold (calibrator fit on val too)
    iso_model = fit_isotonic(proba_cw_val, y_val_arr)
    val_calibrated = apply_isotonic(iso_model, proba_cw_val)
    test_calibrated = apply_isotonic(iso_model, proba_cw_test)

    sweep_calib = optimize_threshold(
        y_val_arr, val_calibrated, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE
    )
    m = evaluate_full(y_test_arr, test_calibrated, threshold=sweep_calib.optimal_threshold)
    test_cost_calib = expected_cost(
        y_test_arr, test_calibrated, sweep_calib.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )
    rows.append(
        (
            f"class weighting + isotonic calibration + optimized threshold ({sweep_calib.optimal_threshold:.2f})",
            m["pr_auc"],
            test_cost_calib,
        )
    )

    print("| Configuration | PR-AUC | Expected cost |")
    print("|---|---|---|")
    for name, pr_auc, cost in rows:
        print(f"| {name} | {pr_auc:.3f} | ${cost:,.2f} |")


if __name__ == "__main__":
    main()

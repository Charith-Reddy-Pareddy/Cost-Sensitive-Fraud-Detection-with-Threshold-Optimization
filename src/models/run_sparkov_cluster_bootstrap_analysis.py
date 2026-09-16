"""Cluster (card-level) bootstrap vs. plain row-level bootstrap, on Sparkov.

The IID bootstrap used everywhere else in this project (`bootstrap_ci`) resamples individual
rows, which implicitly assumes each row is independent evidence. That's not true on Sparkov:
`cc_num` means the same card appears in many transactions, and those transactions share
correlated structure (the same spending pattern, the same compromise event if the card is
eventually used fraudulently). The primary dataset has no entity identifier so this comparison
isn't possible there; Sparkov's `cc_num` makes it possible here.

Same train/val/test protocol as `run_sparkov_bootstrap_analysis.py` — model fit on train,
threshold selected on val, both bootstrap methods applied to the same untouched test predictions.
"""

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.data.ingest import three_way_chronological_split
from src.data.ingest_sparkov import TARGET_COLUMN, engineer_features, load_raw
from src.models.bootstrap_evaluation import bootstrap_ci, cluster_bootstrap_ci
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    expected_cost,
    optimize_threshold,
)

N_BOOTSTRAP = 1000


def _build_pipeline(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )


def _precision_at(threshold):
    return lambda y_true, y_proba: precision_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)


def _recall_at(threshold):
    return lambda y_true, y_proba: recall_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)


def _cost_at(threshold):
    return lambda y_true, y_proba: expected_cost(
        y_true, y_proba, threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )


def main() -> None:
    raw = load_raw()
    features = engineer_features(raw)
    # engineer_features drops cc_num; carry it alongside (same row order/index as `raw`) so the
    # split below keeps each row's card identity attached, purely for clustering the bootstrap —
    # it is never used as a model feature.
    features["cc_num"] = raw["cc_num"].to_numpy()
    features = features.rename(columns={"unix_time": "Time"})

    train, val, test = three_way_chronological_split(features)
    feature_cols = [c for c in features.columns if c not in ("Time", TARGET_COLUMN, "cc_num")]

    X_train, y_train = train[feature_cols], train[TARGET_COLUMN]
    X_val, y_val = val[feature_cols], val[TARGET_COLUMN]
    X_test, y_test = test[feature_cols], test[TARGET_COLUMN]
    test_cc_num = test["cc_num"].to_numpy()

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline = _build_pipeline(scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    sweep = optimize_threshold(y_val.to_numpy(), val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
    threshold = sweep.optimal_threshold

    y_test_arr = y_test.to_numpy()
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    n_unique_cards = len(pd.unique(test_cc_num))
    print(f"test set: {len(y_test_arr)} rows, {n_unique_cards} unique cards, threshold (val) = {threshold:.4f}\n")

    metrics = {
        "pr_auc": average_precision_score,
        f"precision_at_{threshold:.2f}": _precision_at(threshold),
        f"recall_at_{threshold:.2f}": _recall_at(threshold),
        f"expected_cost_at_{threshold:.2f}": _cost_at(threshold),
    }

    print(f"{'metric':28} {'estimate':>12} {'IID 95% CI':>26} {'cluster 95% CI':>26} {'width ratio':>12}")
    for name, fn in metrics.items():
        iid = bootstrap_ci(y_test_arr, test_proba, fn, n_bootstrap=N_BOOTSTRAP)
        cluster = cluster_bootstrap_ci(y_test_arr, test_proba, test_cc_num, fn, n_bootstrap=N_BOOTSTRAP)
        iid_width = iid.upper - iid.lower
        cluster_width = cluster.upper - cluster.lower
        ratio = cluster_width / iid_width if iid_width > 0 else float("nan")
        print(
            f"{name:28} {iid.point_estimate:12.4f} "
            f"[{iid.lower:10.4f}, {iid.upper:10.4f}] "
            f"[{cluster.lower:10.4f}, {cluster.upper:10.4f}] "
            f"{ratio:12.2f}"
        )


if __name__ == "__main__":
    main()

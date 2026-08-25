"""Walk-forward evaluation on Sparkov — does the primary dataset's temporal-stability finding
(2 of 4 folds) also describe Sparkov, or was that finding itself dataset-specific?

Same design as run_temporal_evaluation.py: 5 equal chronological blocks, fold k trains on an
expanding window (blocks 0..k-1), selects the threshold on the first half of block k, reports
on the second half.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import TARGET_COLUMN, engineer_features, load_raw
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold

N_BLOCKS = 5


def _build_pipeline(scale_pos_weight: float) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def _make_blocks(df: pd.DataFrame, n_blocks: int) -> list[pd.DataFrame]:
    ordered = df.sort_values("unix_time").reset_index(drop=True)
    edges = np.linspace(0, len(ordered), n_blocks + 1).astype(int)
    return [ordered.iloc[edges[i] : edges[i + 1]].reset_index(drop=True) for i in range(n_blocks)]


def main() -> None:
    features = engineer_features(load_raw())
    feature_cols = [c for c in features.columns if c not in ("unix_time", TARGET_COLUMN)]
    blocks = _make_blocks(features, N_BLOCKS)

    rows = []
    for k in range(1, N_BLOCKS):
        train_df = pd.concat(blocks[:k], ignore_index=True)
        window = blocks[k]
        half = len(window) // 2
        val_df, test_df = window.iloc[:half], window.iloc[half:]

        X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
        X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN].to_numpy()
        X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN].to_numpy()

        if y_val.sum() == 0 or y_test.sum() == 0:
            rows.append((k, len(train_df), None, None, None, "skipped: no fraud in val or test half"))
            continue

        n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
        pipeline = _build_pipeline(scale_pos_weight=n_neg / n_pos)
        pipeline.fit(X_train, y_train)

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]

        sweep = optimize_threshold(y_val, val_proba, cost_fn=DEFAULT_COST_FALSE_NEGATIVE, cost_fp=DEFAULT_COST_FALSE_POSITIVE)
        default_cost = expected_cost(y_test, test_proba, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        optimal_cost = expected_cost(
            y_test, test_proba, sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
        )
        beats_default = optimal_cost <= default_cost

        rows.append((k, len(train_df), sweep.optimal_threshold, default_cost, optimal_cost, "beats" if beats_default else "worse"))

    print("| Fold | Train rows | Threshold (val) | Default cost (test) | Optimized cost (test) | Result |")
    print("|---|---|---|---|---|---|")
    n_valid = 0
    n_beats = 0
    for k, n_train, threshold, default_cost, optimal_cost, result in rows:
        if threshold is None:
            print(f"| {k} | {n_train} | — | — | — | {result} |")
            continue
        n_valid += 1
        n_beats += result == "beats"
        print(f"| {k} | {n_train} | {threshold:.2f} | ${default_cost:,.2f} | ${optimal_cost:,.2f} | {result} |")

    print(f"\noptimized threshold beat default in {n_beats}/{n_valid} valid folds")


if __name__ == "__main__":
    main()

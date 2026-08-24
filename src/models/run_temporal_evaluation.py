"""Repeated temporal (walk-forward) evaluation.

A single chronological split answers "did this work on this one test window." Walk-forward
evaluation asks the sharper question: does cost-sensitive threshold optimization *consistently*
beat the default 0.5 threshold across different time windows, or did the single-split result
just get lucky?

This dataset is a single day, so genuine multi-day concept drift can't be tested here — that
limitation is real and stated in the README. What walk-forward *can* test, honestly, is whether
the conclusion holds up across different intra-day windows rather than resting on one split.

The full (sorted-by-Time) dataset is cut into 5 equal contiguous blocks. Fold k trains on
blocks[0:k] (expanding window), selects the threshold on the first half of block k, and reports
on the second half of block k — so training data only ever grows forward in time and no fold's
test half is ever used for anything but final reporting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest import load_raw
from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, expected_cost, optimize_threshold
from src.models.imbalance_comparison import build_pipeline

N_BLOCKS = 5


def _make_blocks(df: pd.DataFrame, n_blocks: int) -> list[pd.DataFrame]:
    ordered = df.sort_values("Time").reset_index(drop=True)
    edges = np.linspace(0, len(ordered), n_blocks + 1).astype(int)
    return [ordered.iloc[edges[i] : edges[i + 1]].reset_index(drop=True) for i in range(n_blocks)]


def main() -> None:
    df = load_raw()
    blocks = _make_blocks(df, N_BLOCKS)

    rows = []
    for k in range(1, N_BLOCKS):
        train_df = pd.concat(blocks[:k], ignore_index=True)
        window = blocks[k]
        half = len(window) // 2
        val_df, test_df = window.iloc[:half], window.iloc[half:]

        X_train, y_train = train_df[RAW_FEATURE_COLUMNS], train_df[TARGET_COLUMN]
        X_val, y_val = val_df[RAW_FEATURE_COLUMNS], val_df[TARGET_COLUMN].to_numpy()
        X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET_COLUMN].to_numpy()

        if y_val.sum() == 0 or y_test.sum() == 0:
            rows.append((k, len(train_df), None, None, None, "skipped: no fraud in val or test half"))
            continue

        pipeline = build_pipeline("class_weight")
        n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
        pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
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

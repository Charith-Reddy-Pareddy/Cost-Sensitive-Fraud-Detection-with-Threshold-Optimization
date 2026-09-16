"""When should cost asymmetry be incorporated during training versus decision-time thresholding,
and what happens when it's applied to both?

Every other experiment in this project answers this question on one or two real datasets, at
one point in imbalance/signal/cost-ratio/sample-size space. That's informative but doesn't
distinguish "this is how cost-sensitivity works, generally" from "this is what happened to work
on these two specific datasets." This script runs a controlled factorial study on synthetic data,
varying four factors independently:

- **class imbalance** (fraud fraction: 0.005, 0.02, 0.08 — severe to mild)
- **signal strength** (`class_sep` in `make_classification`: 0.5, 1.5, 3.0 — weak to strong
  separability between classes)
- **cost ratio** (cost_fn / cost_fp: 10, 100 — moderate to severe asymmetry)
- **sample size** (5,000 and 50,000 rows)

giving 3x3x2x2 = 36 cells, each repeated over 3 random seeds (108 runs total) to average out
single-draw noise. For each run: train a standard model and a cost-weighted model (same
mechanism as `cost_sample_weights` elsewhere in this project), select the exact-search
cost-optimal threshold for each on a validation split, and compare four configurations on a test
split — A: standard @ 0.5, B: standard + tuned threshold, C: cost-weighted @ 0.5, D:
cost-weighted + tuned threshold — exactly Experiment 4's A/B/C/D, just repeated across a
controlled grid instead of once on one real dataset.

**Scope, stated upfront:** this covers 4 of the 6 factors suggested for this kind of study
(imbalance, signal strength, cost ratio, sample size). Calibration error is measured as an
*outcome* (Brier score of each trained model) rather than independently injected — doing that
properly needs a principled way to distort calibration without also changing discrimination, which
is its own sub-project. Temporal shift needs a synthetic generator with a drifting decision
boundary, which this script doesn't build. Both are listed in Future work rather than
approximated poorly here.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.calibration import brier_score
from src.models.cost_engine import exact_optimal_threshold, expected_cost
from src.models.cost_sensitive_training import cost_sample_weights

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

IMBALANCE_LEVELS = [0.005, 0.02, 0.08]
SIGNAL_LEVELS = [0.5, 1.5, 3.0]
COST_RATIO_LEVELS = [10, 100]
SAMPLE_SIZE_LEVELS = [5000, 50000]
SEEDS = [0, 1, 2]
MIN_POSITIVES_PER_SPLIT = 5


def _generate(n_samples: int, imbalance: float, class_sep: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[1 - imbalance, imbalance],
        class_sep=class_sep,
        flip_y=0.01,
        random_state=seed,
    )


def _run_one(n_samples: int, imbalance: float, class_sep: float, cost_ratio: float, seed: int) -> dict | None:
    X, y = _generate(n_samples, imbalance, class_sep, seed)

    # a stratified 3-way split needs at least a few positives per split just to succeed at all,
    # let alone give a meaningful comparison — check before splitting, since the split itself
    # raises if any class has fewer members than the number of groups it's stratified into.
    if int(y.sum()) < 3 * MIN_POSITIVES_PER_SPLIT:
        return None

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)

    if min(y_train.sum(), y_val.sum(), y_test.sum()) < MIN_POSITIVES_PER_SPLIT:
        return None  # too few positive rows at this cell for a meaningful comparison

    cost_fn, cost_fp = float(cost_ratio), 1.0

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    standard = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.2, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=n_neg / n_pos
    )
    standard.fit(X_train, y_train)

    cost_weighted = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.2, eval_metric="aucpr", n_jobs=-1)
    weights = cost_sample_weights(y_train, cost_fn, cost_fp)
    cost_weighted.fit(X_train, y_train, sample_weight=weights)

    val_std, test_std = standard.predict_proba(X_val)[:, 1], standard.predict_proba(X_test)[:, 1]
    val_cw, test_cw = cost_weighted.predict_proba(X_val)[:, 1], cost_weighted.predict_proba(X_test)[:, 1]

    sweep_std = exact_optimal_threshold(y_val, val_std, cost_fn, cost_fp)
    sweep_cw = exact_optimal_threshold(y_val, val_cw, cost_fn, cost_fp)

    cost_a = expected_cost(y_test, test_std, 0.5, cost_fn, cost_fp)
    cost_b = expected_cost(y_test, test_std, sweep_std.optimal_threshold, cost_fn, cost_fp)
    cost_c = expected_cost(y_test, test_cw, 0.5, cost_fn, cost_fp)
    cost_d = expected_cost(y_test, test_cw, sweep_cw.optimal_threshold, cost_fn, cost_fp)

    costs = {"A": cost_a, "B": cost_b, "C": cost_c, "D": cost_d}
    best_config = min(costs, key=costs.get)

    return {
        "imbalance": imbalance,
        "class_sep": class_sep,
        "cost_ratio": cost_ratio,
        "n_samples": n_samples,
        "seed": seed,
        "cost_a": cost_a,
        "cost_b": cost_b,
        "cost_c": cost_c,
        "cost_d": cost_d,
        "brier_standard": brier_score(y_test, test_std),
        "brier_cost_weighted": brier_score(y_test, test_cw),
        "threshold_tuning_helps": cost_b < cost_a,  # decision-time tuning beats standard default
        "cost_weighting_helps": cost_c < cost_a,  # training-time cost-sensitivity beats standard default
        "combining_beats_training_alone": cost_d < cost_c,  # does tuning on top of cost-weighting help further?
        "best_config": best_config,
    }


def main() -> None:
    rows = []
    skipped = 0
    combos = list(itertools.product(IMBALANCE_LEVELS, SIGNAL_LEVELS, COST_RATIO_LEVELS, SAMPLE_SIZE_LEVELS, SEEDS))
    for imbalance, class_sep, cost_ratio, n_samples, seed in combos:
        result = _run_one(n_samples, imbalance, class_sep, cost_ratio, seed)
        if result is None:
            skipped += 1
            continue
        rows.append(result)

    df = pd.DataFrame(rows)
    print(f"{len(rows)} runs completed, {skipped} skipped (fewer than {MIN_POSITIVES_PER_SPLIT} positives in some split)\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "synthetic_cost_sensitivity_study.csv", index=False)
    print(f"wrote {len(df)} rows to results/synthetic_cost_sensitivity_study.csv\n")

    print("=== Overall (averaged over seeds and all other factors) ===")
    print(f"threshold tuning helps (B beats A):        {df['threshold_tuning_helps'].mean():.1%} of runs")
    print(f"cost-weighted training helps (C beats A):  {df['cost_weighting_helps'].mean():.1%} of runs")
    print(f"combining helps further (D beats C):       {df['combining_beats_training_alone'].mean():.1%} of runs")
    print(f"best config overall:\n{df['best_config'].value_counts(normalize=True).round(3).to_dict()}\n")

    print("=== By imbalance (mean cost-weighted-training benefit and combining-helps rate) ===")
    by_imbalance = df.groupby("imbalance").agg(
        cost_weighting_helps_rate=("cost_weighting_helps", "mean"),
        combining_helps_rate=("combining_beats_training_alone", "mean"),
        mean_brier_gap=("brier_cost_weighted", "mean"),
    )
    by_imbalance["mean_brier_gap"] = by_imbalance["mean_brier_gap"] - df.groupby("imbalance")["brier_standard"].mean()
    print(by_imbalance.round(4).to_string())
    print()

    print("=== By signal strength (class_sep) ===")
    by_signal = df.groupby("class_sep").agg(
        cost_weighting_helps_rate=("cost_weighting_helps", "mean"),
        combining_helps_rate=("combining_beats_training_alone", "mean"),
    )
    print(by_signal.round(4).to_string())
    print()

    print("=== By cost ratio ===")
    by_ratio = df.groupby("cost_ratio").agg(
        cost_weighting_helps_rate=("cost_weighting_helps", "mean"),
        combining_helps_rate=("combining_beats_training_alone", "mean"),
    )
    print(by_ratio.round(4).to_string())
    print()

    print("=== By sample size ===")
    by_n = df.groupby("n_samples").agg(
        cost_weighting_helps_rate=("cost_weighting_helps", "mean"),
        combining_helps_rate=("combining_beats_training_alone", "mean"),
    )
    print(by_n.round(4).to_string())


if __name__ == "__main__":
    main()

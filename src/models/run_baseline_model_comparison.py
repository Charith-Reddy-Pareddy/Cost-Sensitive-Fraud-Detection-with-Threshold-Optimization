"""Are this project's findings XGBoost-specific, or do they hold across model families?

Every experiment so far uses one model: class-weighted XGBoost. That leaves an open question —
is "cost-sensitive threshold optimization has a real but fragile benefit" a property of this
*method*, or an artifact of this one specific gradient-boosted-tree implementation? Two
additional, deliberately different model families, each imbalance-aware in its own way:

- **Balanced Random Forest** (`imblearn.ensemble`) — bagging, not boosting: each tree trains on a
  randomly *undersampled*, class-balanced bootstrap, a structurally different way of handling
  imbalance from XGBoost's `scale_pos_weight`.
- **Calibrated logistic regression** — a linear model, deliberately the simplest possible
  baseline, wrapped in `CalibratedClassifierCV` (isotonic) so it starts from a model family that's
  usually well-calibrated by construction, unlike the tree ensembles studied so far.

A third family, LightGBM, was tried and dropped: under identical preprocessing and even with no
class weighting at all, it scored PR-AUC ~0.02-0.05 on this specific dataset versus XGBoost's 0.82
under the same no-weighting condition — a genuine, reproducible anomaly (confirmed not a broken
install: the identical LightGBM setup reaches PR-AUC ~0.74-1.0 on synthetic data at a matched
0.17% imbalance ratio), isolated to something about this dataset's real feature distributions that
brief hyperparameter changes (`is_unbalance`, `min_child_samples`, `min_split_gain`, thread count,
scaled vs. raw features, dropping Time/Amount) did not resolve. Rather than report a broken number,
it's excluded here and documented as an honest limitation — see RESEARCH_REPORT.md.

Same protocol as everywhere else: fit on train, select the (exact-search) cost-optimal threshold
on val, report Brier score and cost on the untouched test split.
"""

from pathlib import Path

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor
from src.models.calibration import brier_score
from src.models.cost_engine import DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE, exact_optimal_threshold, expected_cost

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    scale_pos_weight = n_neg / n_pos

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    models = {
        "XGBoost (class-weighted)": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
        ),
        "Balanced Random Forest": BalancedRandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=0, sampling_strategy="all", replacement=True, n_jobs=-1
        ),
        "Calibrated logistic regression": CalibratedClassifierCV(
            LogisticRegression(class_weight="balanced", max_iter=1000), method="isotonic", cv=3
        ),
    }

    default_cost_by_model = {}
    print("| Model | PR-AUC (test) | Brier (test) | Threshold (val) | Test cost | vs. default |")
    print("|---|---|---|---|---|---|")
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        val_proba = model.predict_proba(X_val_t)[:, 1]
        test_proba = model.predict_proba(X_test_t)[:, 1]

        pr_auc = average_precision_score(y_test_arr, test_proba)
        brier = brier_score(y_test_arr, test_proba)

        sweep = exact_optimal_threshold(y_val_arr, val_proba, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        default_cost = expected_cost(y_test_arr, test_proba, 0.5, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
        optimal_cost = expected_cost(
            y_test_arr, test_proba, sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
        )
        default_cost_by_model[name] = default_cost
        reduction_pct = 100 * (default_cost - optimal_cost) / default_cost if default_cost else 0.0

        print(
            f"| {name} | {pr_auc:.3f} | {brier:.5f} | {sweep.optimal_threshold:.4f} | "
            f"${optimal_cost:,.2f} | {reduction_pct:+.2f}% |"
        )

    print("\nDefault-threshold (0.5) test cost by model:")
    for name, cost in default_cost_by_model.items():
        print(f"  {name}: ${cost:,.2f}")


if __name__ == "__main__":
    main()

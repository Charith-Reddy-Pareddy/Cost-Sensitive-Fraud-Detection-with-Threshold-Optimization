"""Decision-policy comparison: amount-proportional costs, a decision curve, and capacity-
constrained review, evaluated together because the first genuinely changes the other two.

Every other script in this project uses one flat cost per class ($500 for any missed fraud, $5
for any blocked legitimate transaction). That's a simplification: a missed $9,000 fraud and a
missed $9 fraud are not the same loss. This script switches to `amount_proportional_fn_cost`
(cost_fn_i = transaction amount, floored) and asks three questions against that more realistic
cost model, fit/selected on val and reported on the untouched test split throughout:

1. Does the optimal threshold or the qualitative conclusion change under amount-proportional
   costs, versus the flat $500/$5 figure used everywhere else in this project?
2. What does the decision curve (fraud dollars caught vs. legitimate transactions blocked) look
   like as the threshold sweeps — the Pareto frontier a fraud-ops team actually faces?
3. If review capacity is fixed (e.g. an analyst team can only review K transactions/day), how do
   four policies compare: a static cost-optimal threshold (uncapped), top-K by raw score,
   cost-sensitive top-K (ranked by expected dollar loss avoided = proba * amount), and top-K by a
   calibrated version of the same ranking? Under the *flat* cost model these last three would all
   produce the identical ranking (raw score, expected value, and calibrated expected value all
   preserve rank order when every fraud costs the same) — amount-proportional costs are what
   makes them genuinely different policies worth comparing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.calibration import apply_isotonic, fit_isotonic
from src.models.cost_engine import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    amount_proportional_fn_cost,
    exact_optimal_threshold,
    exact_optimal_threshold_variable_cost,
    expected_cost,
    expected_cost_variable,
)
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
REVIEW_CAPACITY_K = 100  # illustrative fixed review capacity for the test window


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def _decision_curve(y_true: np.ndarray, y_proba: np.ndarray, amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each threshold in a fine grid: (legitimate transactions blocked, fraud dollars caught).
    Both are monotonically non-decreasing as the threshold drops (more gets flagged), which is
    exactly what makes this a Pareto frontier — you cannot buy more fraud-dollars-caught without
    also buying more legitimate transactions blocked."""
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    amt_sorted = amounts[order]
    is_fraud = y_sorted == 1
    fraud_dollars_caught = np.cumsum(np.where(is_fraud, amt_sorted, 0.0))
    legit_blocked = np.cumsum(np.where(~is_fraud, 1, 0))
    return legit_blocked, fraud_dollars_caught


def _topk_indices(y_proba: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-y_proba)[:k]


def _policy_outcome(y_true: np.ndarray, amounts: np.ndarray, flagged_idx: np.ndarray) -> dict:
    flagged = np.zeros(len(y_true), dtype=bool)
    flagged[flagged_idx] = True
    fraud_caught = int(((y_true == 1) & flagged).sum())
    fraud_total = int((y_true == 1).sum())
    fraud_dollars_caught = float(amounts[(y_true == 1) & flagged].sum())
    fraud_dollars_total = float(amounts[y_true == 1].sum())
    legit_blocked = int(((y_true == 0) & flagged).sum())
    return {
        "n_flagged": int(flagged.sum()),
        "fraud_caught": fraud_caught,
        "fraud_total": fraud_total,
        "recall": fraud_caught / fraud_total if fraud_total else 0.0,
        "fraud_dollars_caught": fraud_dollars_caught,
        "fraud_dollars_total": fraud_dollars_total,
        "dollar_recall": fraud_dollars_caught / fraud_dollars_total if fraud_dollars_total else 0.0,
        "legit_blocked": legit_blocked,
    }


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()
    amounts_val, amounts_test = X_val["Amount"].to_numpy(), X_test["Amount"].to_numpy()

    pipeline = build_pipeline("class_weight")
    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- 1. flat vs. amount-proportional cost ---
    flat_sweep = exact_optimal_threshold(y_val_arr, val_proba, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE)
    flat_test_cost = expected_cost(
        y_test_arr, test_proba, flat_sweep.optimal_threshold, DEFAULT_COST_FALSE_NEGATIVE, DEFAULT_COST_FALSE_POSITIVE
    )

    cost_fn_val = amount_proportional_fn_cost(amounts_val, loss_rate=1.0, min_cost=5.0)
    cost_fp_val = np.full(len(y_val_arr), DEFAULT_COST_FALSE_POSITIVE)
    amount_sweep = exact_optimal_threshold_variable_cost(y_val_arr, val_proba, cost_fn_val, cost_fp_val)

    cost_fn_test = amount_proportional_fn_cost(amounts_test, loss_rate=1.0, min_cost=5.0)
    cost_fp_test = np.full(len(y_test_arr), DEFAULT_COST_FALSE_POSITIVE)
    amount_test_cost_at_amount_threshold = expected_cost_variable(
        y_test_arr, test_proba, amount_sweep.optimal_threshold, cost_fn_test, cost_fp_test
    )
    amount_test_cost_at_flat_threshold = expected_cost_variable(
        y_test_arr, test_proba, flat_sweep.optimal_threshold, cost_fn_test, cost_fp_test
    )

    print("=== 1. Flat vs. amount-proportional cost ===")
    print(f"Flat-cost optimal threshold (val): {flat_sweep.optimal_threshold:.4f}, test cost: ${flat_test_cost:,.2f} (flat $/txn)")
    print(f"Amount-proportional optimal threshold (val): {amount_sweep.optimal_threshold:.4f}")
    print(f"  test cost at amount-proportional threshold: ${amount_test_cost_at_amount_threshold:,.2f}")
    print(f"  test cost at flat-selected threshold instead: ${amount_test_cost_at_flat_threshold:,.2f}")
    print(f"  total fraud-dollar exposure in test: ${cost_fn_test[y_test_arr == 1].sum():,.2f}\n")

    # --- 2. decision curve ---
    legit_blocked, fraud_dollars_caught = _decision_curve(y_test_arr, test_proba, amounts_test)
    plt.figure(figsize=(7, 5))
    plt.plot(legit_blocked, fraud_dollars_caught)
    plt.xlabel("legitimate transactions blocked")
    plt.ylabel("fraud dollars caught ($)")
    plt.title("Decision curve: fraud dollars caught vs. legitimate transactions blocked")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "decision_curve.png", dpi=150)
    plt.close()
    print("=== 2. Decision curve saved to reports/figures/decision_curve.png ===")
    for target_blocked in (10, 50, 100, 300):
        i = np.searchsorted(legit_blocked, target_blocked)
        if i < len(fraud_dollars_caught):
            print(f"  at {target_blocked} legit txns blocked: ${fraud_dollars_caught[i]:,.2f} fraud dollars caught")
    print()

    # --- 3. capacity-constrained policies (K = REVIEW_CAPACITY_K) ---
    K = REVIEW_CAPACITY_K
    iso_model = fit_isotonic(val_proba, y_val_arr)
    test_proba_calibrated = apply_isotonic(iso_model, test_proba)

    static_flagged = np.where(test_proba >= flat_sweep.optimal_threshold)[0]
    topk_raw_flagged = _topk_indices(test_proba, K)
    expected_loss = test_proba * amounts_test  # expected dollar loss avoided by flagging
    topk_cost_sensitive_flagged = _topk_indices(expected_loss, K)
    expected_loss_calibrated = test_proba_calibrated * amounts_test
    topk_calibrated_flagged = _topk_indices(expected_loss_calibrated, K)

    policies = {
        f"static threshold ({flat_sweep.optimal_threshold:.4f}, uncapped)": static_flagged,
        f"top-K by raw score (K={K})": topk_raw_flagged,
        f"top-K cost-sensitive: proba*amount (K={K})": topk_cost_sensitive_flagged,
        f"top-K calibrated: calibrated_proba*amount (K={K})": topk_calibrated_flagged,
    }

    print(f"=== 3. Capacity-constrained review policies (K={K}) ===")
    print("| Policy | # flagged | Fraud caught | Recall | $ caught | $ recall | Legit blocked |")
    print("|---|---|---|---|---|---|---|")
    for name, idx in policies.items():
        r = _policy_outcome(y_test_arr, amounts_test, idx)
        print(
            f"| {name} | {r['n_flagged']} | {r['fraud_caught']}/{r['fraud_total']} | {r['recall']:.3f} | "
            f"${r['fraud_dollars_caught']:,.2f} | {r['dollar_recall']:.3f} | {r['legit_blocked']} |"
        )

    overlap_raw_cs = len(set(topk_raw_flagged) & set(topk_cost_sensitive_flagged))
    print(f"\nOverlap between top-K-by-score and top-K-cost-sensitive selections: {overlap_raw_cs}/{K}")


if __name__ == "__main__":
    main()

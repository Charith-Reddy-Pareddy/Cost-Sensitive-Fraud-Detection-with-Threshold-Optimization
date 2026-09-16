"""Cost-sensitive threshold optimization.

Assigns an explicit dollar cost to a missed fraud (false negative) and to a blocked legitimate
transaction (false positive), then picks the classification threshold that minimizes total
expected cost rather than defaulting to 0.5. The dollar figures below are illustrative
assumptions chosen to demonstrate the method — not sourced fraud-loss figures — and are labeled
as such everywhere they're used.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_COST_FALSE_NEGATIVE = 500.0  # illustrative: average loss from a missed fraud
DEFAULT_COST_FALSE_POSITIVE = 5.0  # illustrative: cost of blocking/reviewing a legit transaction


@dataclass
class ThresholdSweepResult:
    thresholds: np.ndarray
    costs: np.ndarray
    optimal_threshold: float
    optimal_cost: float


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    return false_negatives * cost_fn + false_positives * cost_fp


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
    thresholds: np.ndarray | None = None,
) -> ThresholdSweepResult:
    """Sweep candidate thresholds and return the one minimizing total expected cost."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    costs = np.array([expected_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds])
    best_idx = int(np.argmin(costs))

    return ThresholdSweepResult(
        thresholds=thresholds,
        costs=costs,
        optimal_threshold=float(thresholds[best_idx]),
        optimal_cost=float(costs[best_idx]),
    )


def exact_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
) -> ThresholdSweepResult:
    """Find the cost-minimizing threshold exactly, in O(n log n), instead of checking a fixed
    grid. `expected_cost` is a step function of the threshold that can only change value at a
    predicted score itself (since predictions use `proba >= threshold`), so the true minimum is
    guaranteed to sit at one of the observed scores — a grid can straddle it and miss it entirely
    if no grid point lands close enough. This function checks every one of those candidate
    thresholds by sorting scores once and tracking cumulative FN/FP counts, rather than
    recomputing the confusion matrix from scratch at each of a fixed number of grid points.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_proba)

    n_fraud = int((y_true == 1).sum())
    n_legit = n - n_fraud

    if n == 0:
        return ThresholdSweepResult(thresholds=np.array([]), costs=np.array([]), optimal_threshold=0.5, optimal_cost=0.0)

    # threshold above every score: nobody flagged -> all fraud is a false negative
    no_flag_cost = n_fraud * cost_fn

    order = np.argsort(-y_proba, kind="mergesort")  # descending, stable so ties keep dataset order
    sorted_scores = y_proba[order]
    sorted_labels = y_true[order]

    # after flagging the top k scores as positive: FN = fraud not yet flagged, FP = legit flagged
    cum_fraud_flagged = np.cumsum(sorted_labels == 1)
    cum_legit_flagged = np.cumsum(sorted_labels == 0)
    fn_counts = n_fraud - cum_fraud_flagged
    fp_counts = cum_legit_flagged
    costs_after_k = fn_counts * cost_fn + fp_counts * cost_fp

    # tied scores must move together under a single threshold (>=), so only the last position in
    # each run of equal scores is a valid candidate — flagging some but not all tied scores isn't
    # achievable by any single threshold value.
    is_last_of_tie_group = np.ones(n, dtype=bool)
    is_last_of_tie_group[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    candidate_costs = costs_after_k[is_last_of_tie_group]
    candidate_thresholds = sorted_scores[is_last_of_tie_group]

    best_flag_idx = int(np.argmin(candidate_costs))
    best_flag_cost = float(candidate_costs[best_flag_idx])
    best_flag_threshold = float(candidate_thresholds[best_flag_idx])

    if no_flag_cost <= best_flag_cost:
        optimal_cost = no_flag_cost
        optimal_threshold = float(sorted_scores[0]) + 1e-9  # just above the highest score
    else:
        optimal_cost = best_flag_cost
        optimal_threshold = best_flag_threshold

    return ThresholdSweepResult(
        thresholds=candidate_thresholds[::-1],
        costs=candidate_costs[::-1],
        optimal_threshold=optimal_threshold,
        optimal_cost=optimal_cost,
    )


def amount_proportional_fn_cost(amount: np.ndarray, loss_rate: float = 1.0, min_cost: float = 5.0) -> np.ndarray:
    """Per-row false-negative (missed-fraud) cost proportional to the transaction amount, instead
    of one flat figure for every fraud regardless of size. `loss_rate` is the fraction of the
    transaction amount assumed lost if the fraud isn't caught (1.0 = full loss, no recovery — the
    conservative/illustrative default used here; real chargeback/recovery rates are not modeled).
    `min_cost` floors the cost for very small transactions, representing the fixed
    investigation/reputational cost of a missed fraud that exists even when the dollar amount is
    trivial. Like the flat $500/$5 figures elsewhere in this project, these are illustrative
    assumptions, not sourced fraud-loss data.
    """
    return np.maximum(loss_rate * np.asarray(amount, dtype=float), min_cost)


def expected_cost_variable(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
) -> float:
    """Like `expected_cost`, but `cost_fn`/`cost_fp` are per-row arrays (e.g. amount-proportional
    costs from `amount_proportional_fn_cost`) rather than one flat dollar figure applied to every
    row — a missed $9,000 fraud and a missed $9 fraud are not the same cost, which a single flat
    `cost_fn` cannot represent."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cost_fn = np.asarray(cost_fn, dtype=float)
    cost_fp = np.asarray(cost_fp, dtype=float)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    return float(cost_fn[fn_mask].sum() + cost_fp[fp_mask].sum())


def exact_optimal_threshold_variable_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
) -> ThresholdSweepResult:
    """`exact_optimal_threshold`'s O(n log n) exact search over every observed score, generalized
    to per-row costs (see `expected_cost_variable`) instead of a single flat `cost_fn`/`cost_fp`.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    cost_fn = np.asarray(cost_fn, dtype=float)
    cost_fp = np.asarray(cost_fp, dtype=float)
    n = len(y_proba)

    if n == 0:
        return ThresholdSweepResult(thresholds=np.array([]), costs=np.array([]), optimal_threshold=0.5, optimal_cost=0.0)

    no_flag_cost = float(cost_fn[y_true == 1].sum())

    order = np.argsort(-y_proba, kind="mergesort")
    sorted_scores = y_proba[order]
    sorted_labels = y_true[order]
    sorted_fn_cost = cost_fn[order]
    sorted_fp_cost = cost_fp[order]

    # as the top-k highest-scored rows get flagged: fraud rows among them stop costing cost_fn
    # (recovered), legit rows among them start costing cost_fp (incurred)
    fn_cost_recovered = np.cumsum(np.where(sorted_labels == 1, sorted_fn_cost, 0.0))
    fp_cost_incurred = np.cumsum(np.where(sorted_labels == 0, sorted_fp_cost, 0.0))
    costs_after_k = (no_flag_cost - fn_cost_recovered) + fp_cost_incurred

    is_last_of_tie_group = np.ones(n, dtype=bool)
    is_last_of_tie_group[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    candidate_costs = costs_after_k[is_last_of_tie_group]
    candidate_thresholds = sorted_scores[is_last_of_tie_group]

    best_flag_idx = int(np.argmin(candidate_costs))
    best_flag_cost = float(candidate_costs[best_flag_idx])
    best_flag_threshold = float(candidate_thresholds[best_flag_idx])

    if no_flag_cost <= best_flag_cost:
        optimal_cost = no_flag_cost
        optimal_threshold = float(sorted_scores[0]) + 1e-9
    else:
        optimal_cost = best_flag_cost
        optimal_threshold = best_flag_threshold

    return ThresholdSweepResult(
        thresholds=candidate_thresholds[::-1],
        costs=candidate_costs[::-1],
        optimal_threshold=optimal_threshold,
        optimal_cost=optimal_cost,
    )


def bayes_optimal_threshold(
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
) -> float:
    """The theoretical cost-minimizing threshold under perfectly calibrated probabilities.

    Flagging a transaction with true fraud probability p as positive costs (1-p)*cost_fp in
    expectation (the legitimate-transaction case); not flagging it costs p*cost_fn (the missed-
    fraud case). Flagging is the better decision exactly when p*cost_fn > (1-p)*cost_fp, which
    solves to p > cost_fp / (cost_fn + cost_fp). This threshold is optimal *if and only if* the
    model's predicted probabilities are well-calibrated — P(y=1 | score=s) actually equals s. A
    raw classifier's scores are typically not calibrated (see the calibration comparison in
    run_calibration_analysis.py), which is exactly why the empirical optimum found by
    `exact_optimal_threshold` on raw scores can diverge from this value.
    """
    return cost_fp / (cost_fn + cost_fp)


def cost_ratio_sensitivity_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_ratios: np.ndarray,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """For each cost ratio (cost_fn / cost_fp), find the optimal threshold. Shows how the
    optimum moves as the relative cost of missing fraud changes — a static single-ratio chart
    can't show this, which is why this sweep exists as its own analysis rather than a footnote.
    """
    results = []
    for ratio in cost_ratios:
        cost_fn = ratio * cost_fp
        sweep = optimize_threshold(y_true, y_proba, cost_fn=cost_fn, cost_fp=cost_fp, thresholds=thresholds)
        results.append(
            {
                "cost_ratio": float(ratio),
                "cost_fn": cost_fn,
                "cost_fp": cost_fp,
                "optimal_threshold": sweep.optimal_threshold,
                "optimal_cost": sweep.optimal_cost,
            }
        )
    return results

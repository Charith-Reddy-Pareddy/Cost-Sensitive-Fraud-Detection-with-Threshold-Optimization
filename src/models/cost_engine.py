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

import numpy as np

from src.models.cost_engine import cost_ratio_sensitivity_sweep, expected_cost, optimize_threshold


def test_expected_cost_counts_fn_and_fp_correctly():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.9, 0.1])  # 1 FN (missed the 2nd fraud), 1 FP (flagged 3rd)
    cost = expected_cost(y_true, y_proba, threshold=0.5, cost_fn=100.0, cost_fp=10.0)
    assert cost == 100.0 + 10.0


def test_expected_cost_zero_when_perfectly_separated():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.1, 0.1])
    cost = expected_cost(y_true, y_proba, threshold=0.5, cost_fn=100.0, cost_fp=10.0)
    assert cost == 0.0


def test_optimize_threshold_beats_or_matches_default():
    rng = np.random.default_rng(0)
    n = 2000
    y_true = (rng.random(n) < 0.05).astype(int)
    # noisy but informative score: fraud tends to score higher
    y_proba = np.clip(y_true * 0.6 + rng.normal(scale=0.25, size=n) + 0.2, 0, 1)

    default_cost = expected_cost(y_true, y_proba, threshold=0.5, cost_fn=500.0, cost_fp=5.0)
    sweep = optimize_threshold(y_true, y_proba, cost_fn=500.0, cost_fp=5.0)

    assert sweep.optimal_cost <= default_cost
    assert 0.0 <= sweep.optimal_threshold <= 1.0


def test_optimize_threshold_picks_the_minimum_not_the_maximum():
    """Direct, deterministic pin for the argmin/argmax regression: a hand-crafted cost curve
    with an unambiguous minimum away from both ends, so a min<->max mixup fails immediately
    and obviously rather than depending on random data happening to expose it."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    thresholds = np.array([0.0, 0.5, 1.0])

    # threshold=0.0 -> everyone flagged -> 7 FP, cost 70
    # threshold=1.0 -> no one flagged -> 3 FN, cost 30
    # threshold=0.5 -> perfect split -> 0 FN, 0 FP, cost 0 (the true minimum)
    sweep = optimize_threshold(y_true, y_proba, cost_fn=10.0, cost_fp=10.0, thresholds=thresholds)

    assert sweep.optimal_threshold == 0.5
    assert sweep.optimal_cost == 0.0


def test_higher_cost_ratio_lowers_optimal_threshold():
    rng = np.random.default_rng(0)
    n = 2000
    y_true = (rng.random(n) < 0.05).astype(int)
    y_proba = np.clip(y_true * 0.6 + rng.normal(scale=0.25, size=n) + 0.2, 0, 1)

    results = cost_ratio_sensitivity_sweep(y_true, y_proba, cost_ratios=np.array([5, 50, 500]))
    thresholds = [r["optimal_threshold"] for r in results]

    # missing fraud becomes relatively more expensive as the ratio grows, so the optimizer
    # should be willing to accept more false positives to catch more fraud -> lower threshold
    assert thresholds[0] >= thresholds[1] >= thresholds[2]

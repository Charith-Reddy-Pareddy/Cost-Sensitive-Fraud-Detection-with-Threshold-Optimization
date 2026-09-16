import numpy as np

from src.models.cost_engine import (
    amount_proportional_fn_cost,
    bayes_optimal_threshold,
    cost_ratio_sensitivity_sweep,
    exact_optimal_threshold,
    exact_optimal_threshold_variable_cost,
    expected_cost,
    expected_cost_variable,
    optimize_threshold,
)


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


def test_exact_optimal_threshold_never_worse_than_101_point_grid():
    rng = np.random.default_rng(1)
    n = 3000
    y_true = (rng.random(n) < 0.05).astype(int)
    y_proba = np.clip(y_true * 0.6 + rng.normal(scale=0.25, size=n) + 0.2, 0, 1)

    grid_sweep = optimize_threshold(y_true, y_proba, cost_fn=500.0, cost_fp=5.0)
    exact_sweep = exact_optimal_threshold(y_true, y_proba, cost_fn=500.0, cost_fp=5.0)

    # the exact search checks every achievable threshold, so it can only match or beat a grid
    assert exact_sweep.optimal_cost <= grid_sweep.optimal_cost


def test_exact_optimal_threshold_finds_minimum_a_coarse_grid_straddles():
    # scores land at 0.501 and 0.499 either side of the true break; a 101-point grid (step 0.01)
    # never lands there and picks a worse threshold on one side or the other
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.7, 0.501, 0.499, 0.3, 0.1])

    exact_sweep = exact_optimal_threshold(y_true, y_proba, cost_fn=10.0, cost_fp=10.0)
    grid_sweep = optimize_threshold(y_true, y_proba, cost_fn=10.0, cost_fp=10.0)

    assert exact_sweep.optimal_cost == 0.0  # perfectly separable at 0.501
    assert exact_sweep.optimal_cost <= grid_sweep.optimal_cost


def test_exact_optimal_threshold_handles_tied_scores():
    # two frauds and one legit transaction share the exact same score; a threshold can't flag
    # one without flagging all three, so the achievable minimum isn't zero
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.5, 0.5, 0.5, 0.1])

    sweep = exact_optimal_threshold(y_true, y_proba, cost_fn=10.0, cost_fp=10.0)

    # flag all three at 0.5: 0 FN, 1 FP -> cost 10; flag nobody: 2 FN -> cost 20
    assert sweep.optimal_cost == 10.0


def test_bayes_optimal_threshold_matches_closed_form():
    assert bayes_optimal_threshold(cost_fn=500.0, cost_fp=5.0) == 5.0 / 505.0
    assert bayes_optimal_threshold(cost_fn=5.0, cost_fp=5.0) == 0.5
    # missing fraud is 100x worse than a false positive -> only flag when very confident it's NOT fraud is wrong;
    # the threshold should be low, since even a small fraud probability is expensive to ignore
    assert bayes_optimal_threshold(cost_fn=500.0, cost_fp=5.0) < 0.5


def test_amount_proportional_fn_cost_scales_with_amount_and_floors():
    amounts = np.array([0.0, 1.0, 100.0, 10000.0])
    costs = amount_proportional_fn_cost(amounts, loss_rate=1.0, min_cost=5.0)
    assert list(costs) == [5.0, 5.0, 100.0, 10000.0]


def test_expected_cost_variable_matches_flat_expected_cost_for_constant_arrays():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.9, 0.1])
    flat = expected_cost(y_true, y_proba, threshold=0.5, cost_fn=100.0, cost_fp=10.0)
    variable = expected_cost_variable(
        y_true, y_proba, threshold=0.5, cost_fn=np.full(4, 100.0), cost_fp=np.full(4, 10.0)
    )
    assert flat == variable


def test_expected_cost_variable_uses_actual_per_row_amount():
    # one missed fraud: a $50 one, not the $9000 one that got caught -> cost should reflect $50
    y_true = np.array([1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.1])  # first fraud missed, second caught, legit correctly clear
    amounts = np.array([50.0, 9000.0, 20.0])
    cost_fn = amount_proportional_fn_cost(amounts, loss_rate=1.0, min_cost=0.0)
    cost = expected_cost_variable(y_true, y_proba, threshold=0.5, cost_fn=cost_fn, cost_fp=np.full(3, 5.0))
    assert cost == 50.0


def test_exact_optimal_threshold_variable_cost_never_worse_than_flat_equivalent():
    rng = np.random.default_rng(2)
    n = 2000
    y_true = (rng.random(n) < 0.05).astype(int)
    y_proba = np.clip(y_true * 0.6 + rng.normal(scale=0.25, size=n) + 0.2, 0, 1)
    amounts = rng.uniform(1, 500, size=n)
    cost_fn = amount_proportional_fn_cost(amounts, loss_rate=1.0, min_cost=5.0)
    cost_fp = np.full(n, 5.0)

    sweep = exact_optimal_threshold_variable_cost(y_true, y_proba, cost_fn, cost_fp)
    default_cost = expected_cost_variable(y_true, y_proba, 0.5, cost_fn, cost_fp)

    assert sweep.optimal_cost <= default_cost
    assert 0.0 <= sweep.optimal_threshold <= 1.0 + 1e-6


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

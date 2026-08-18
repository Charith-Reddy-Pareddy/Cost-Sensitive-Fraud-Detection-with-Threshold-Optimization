import numpy as np
from sklearn.metrics import average_precision_score

from src.models.bootstrap_evaluation import bootstrap_ci


def test_point_estimate_matches_direct_metric_computation():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.2).astype(int)
    y_proba = np.clip(y_true * 0.5 + rng.normal(scale=0.3, size=500) + 0.25, 0, 1)

    result = bootstrap_ci(y_true, y_proba, average_precision_score, n_bootstrap=200)
    assert result.point_estimate == average_precision_score(y_true, y_proba)


def test_ci_bounds_bracket_the_point_estimate_typically():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.2).astype(int)
    y_proba = np.clip(y_true * 0.5 + rng.normal(scale=0.3, size=500) + 0.25, 0, 1)

    result = bootstrap_ci(y_true, y_proba, average_precision_score, n_bootstrap=500)
    assert result.lower <= result.point_estimate <= result.upper


def test_more_bootstrap_samples_narrows_or_holds_interval_width_stable():
    rng = np.random.default_rng(0)
    y_true = (rng.random(2000) < 0.2).astype(int)
    y_proba = np.clip(y_true * 0.5 + rng.normal(scale=0.3, size=2000) + 0.25, 0, 1)

    small = bootstrap_ci(y_true, y_proba, average_precision_score, n_bootstrap=50, seed=1)
    large = bootstrap_ci(y_true, y_proba, average_precision_score, n_bootstrap=2000, seed=1)

    # both should be genuine, finite intervals around roughly the same point estimate
    assert small.point_estimate == large.point_estimate
    assert large.upper - large.lower > 0

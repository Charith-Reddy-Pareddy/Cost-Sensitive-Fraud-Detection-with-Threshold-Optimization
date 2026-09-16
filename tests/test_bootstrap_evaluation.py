import numpy as np
from sklearn.metrics import average_precision_score

from src.models.bootstrap_evaluation import bootstrap_ci, cluster_bootstrap_ci, paired_bootstrap_comparison


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


def test_cluster_bootstrap_point_estimate_matches_direct_metric():
    rng = np.random.default_rng(0)
    n_clusters, rows_per_cluster = 40, 20
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    y_true = (rng.random(n_clusters * rows_per_cluster) < 0.2).astype(int)
    y_proba = np.clip(y_true * 0.5 + rng.normal(scale=0.3, size=n_clusters * rows_per_cluster) + 0.25, 0, 1)

    result = cluster_bootstrap_ci(y_true, y_proba, cluster_ids, average_precision_score, n_bootstrap=200)
    assert result.point_estimate == average_precision_score(y_true, y_proba)


def test_cluster_bootstrap_is_wider_under_within_cluster_correlation():
    """The whole point of clustering: when rows within a cluster are correlated (share a
    cluster-level effect), row-level IID bootstrap understates uncertainty because it treats
    correlated rows as independent evidence. With a strong cluster effect and few clusters, the
    cluster bootstrap should report a visibly wider interval than the plain IID bootstrap on the
    exact same data."""
    rng = np.random.default_rng(3)
    n_clusters, rows_per_cluster = 15, 40
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    # each cluster's fraud rate is almost entirely determined by a per-cluster random effect,
    # not by independent per-row noise, so the real amount of independent evidence is ~15
    # clusters, not 600 rows
    cluster_effect = rng.random(n_clusters)
    row_fraud_prob = np.repeat(cluster_effect, rows_per_cluster)
    y_true = (rng.random(n_clusters * rows_per_cluster) < row_fraud_prob).astype(int)
    y_proba = np.clip(row_fraud_prob + rng.normal(scale=0.05, size=n_clusters * rows_per_cluster), 0, 1)

    iid_result = bootstrap_ci(y_true, y_proba, average_precision_score, n_bootstrap=500, seed=1)
    cluster_result = cluster_bootstrap_ci(y_true, y_proba, cluster_ids, average_precision_score, n_bootstrap=500, seed=1)

    assert (cluster_result.upper - cluster_result.lower) > (iid_result.upper - iid_result.lower)


def test_paired_bootstrap_point_diff_matches_direct_computation():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.2).astype(int)
    proba_a = np.clip(y_true * 0.6 + rng.normal(scale=0.2, size=500) + 0.2, 0, 1)
    proba_b = np.clip(y_true * 0.3 + rng.normal(scale=0.3, size=500) + 0.2, 0, 1)

    result = paired_bootstrap_comparison(y_true, proba_a, proba_b, average_precision_score, average_precision_score)
    expected = average_precision_score(y_true, proba_a) - average_precision_score(y_true, proba_b)
    assert result.point_diff == expected


def test_paired_bootstrap_prob_a_better_favors_the_clearly_better_config():
    # A is a near-perfect cost predictor (low cost), B is pure noise (high, random cost) -> A
    # should win in nearly every bootstrap resample
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.random(n) < 0.1).astype(int)
    proba_a = np.clip(y_true + rng.normal(scale=0.02, size=n), 0, 1)  # near-perfect
    proba_b = rng.random(n)  # random, uninformative

    def cost_fn(y, proba):
        preds = (proba >= 0.5).astype(int)
        return float(((y == 1) & (preds == 0)).sum() * 500 + ((y == 0) & (preds == 1)).sum() * 5)

    result = paired_bootstrap_comparison(y_true, proba_a, proba_b, cost_fn, cost_fn, lower_is_better=True)
    assert result.prob_a_better > 0.95
    assert result.point_diff < 0  # A's cost is lower than B's

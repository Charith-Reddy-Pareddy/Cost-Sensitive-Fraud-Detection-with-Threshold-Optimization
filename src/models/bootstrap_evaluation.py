"""Bootstrap confidence intervals for point-estimate metrics.

A single chronological test split (57k rows, 75 fraud) can be noisy — resampling with
replacement from that same split and recomputing a metric each time gives a distribution to
report a confidence interval from, without needing repeated retraining.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    samples: np.ndarray


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """`metric_fn(y_true, y_proba) -> float`. Resamples (y_true, y_proba) pairs together with
    replacement — never resample the two arrays independently, that would break the pairing
    between a row's label and its score."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_proba)

    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_proba[idx])

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])

    return BootstrapResult(point_estimate=point_estimate, lower=float(lower), upper=float(upper), samples=samples)


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Like `bootstrap_ci`, but resamples whole clusters (e.g. a card's transactions) with
    replacement, instead of individual rows.

    Plain row-level (IID) bootstrap assumes every row is an independent draw. That's false when
    several rows share a cluster — e.g. the same card appears in dozens of transactions that are
    correlated with each other (same fraud ring, same spending pattern, same period of being
    compromised). Treating them as independent understates the true uncertainty: the IID bootstrap
    can shuffle rows from the same card across many resamples and still report a tight interval,
    when in reality all that card's evidence is really one correlated block, not N independent
    votes. Resampling clusters instead of rows preserves that block structure and is the standard
    fix (Efron & Tibshirani's "cluster bootstrap" / a block bootstrap over natural groups).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    cluster_ids = np.asarray(cluster_ids)

    point_estimate = metric_fn(y_true, y_proba)

    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    indices_by_cluster = [np.where(inverse == c)[0] for c in range(n_clusters)]

    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_cluster_idx = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([indices_by_cluster[c] for c in sampled_cluster_idx])
        samples[i] = metric_fn(y_true[idx], y_proba[idx])

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])

    return BootstrapResult(point_estimate=point_estimate, lower=float(lower), upper=float(upper), samples=samples)


@dataclass
class PairedBootstrapResult:
    point_diff: float
    mean_diff: float
    median_diff: float
    lower: float
    upper: float
    prob_a_better: float
    diffs: np.ndarray


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric_fn_a: Callable[[np.ndarray, np.ndarray], float],
    metric_fn_b: Callable[[np.ndarray, np.ndarray], float],
    lower_is_better: bool = True,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Compare two configurations evaluated on the *same* underlying rows (e.g. two thresholds
    or two training objectives scored on the same test set), resampling once per iteration and
    applying both metric functions to that *same* resampled index set — not resampling A and B
    independently. Pairing isolates the difference between A and B from the sampling noise they
    both share, which is the point: two configurations bootstrapped independently can each look
    individually noisy while their difference is actually quite stable, because they rise and
    fall together across resamples of the same underlying data.

    `metric_fn_a`/`metric_fn_b` take `(y_true, y_proba) -> float`, matching `bootstrap_ci`.
    Reports `diff = metric_a - metric_b`; `prob_a_better` is `P(A beats B)` under
    `lower_is_better` (True for a cost metric, where smaller is better).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n = len(y_true)

    point_a = metric_fn_a(y_true, proba_a)
    point_b = metric_fn_b(y_true, proba_b)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn_a(y_true[idx], proba_a[idx]) - metric_fn_b(y_true[idx], proba_b[idx])

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(diffs, [alpha, 1 - alpha])
    prob_a_better = float((diffs < 0).mean()) if lower_is_better else float((diffs > 0).mean())

    return PairedBootstrapResult(
        point_diff=float(point_a - point_b),
        mean_diff=float(diffs.mean()),
        median_diff=float(np.median(diffs)),
        lower=float(lower),
        upper=float(upper),
        prob_a_better=prob_a_better,
        diffs=diffs,
    )

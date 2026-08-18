"""Bootstrap confidence intervals for point-estimate metrics.

A single chronological test split (57k rows, 75 fraud) can be noisy — resampling with
replacement from that same split and recomputing a metric each time gives a distribution to
report a confidence interval from, without needing repeated retraining.
"""

from dataclasses import dataclass
from typing import Callable

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

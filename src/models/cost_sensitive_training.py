"""Cost-sensitive training: weight each training example by its dollar cost, not just its
class frequency.

Class weighting (elsewhere in this project) reweights by inverse class frequency — it makes
the training objective imbalance-aware, but it has no idea a missed fraud costs $500 and a
blocked transaction costs $5. This module weights each row directly by the relevant dollar
figure (`cost_fn` for fraud rows, `cost_fp` for legitimate rows), so the *training objective
itself* — not just the decision threshold applied afterward — is asymmetric in the same way the
business cost is. This is what lets `run_training_objective_comparison.py` ask whether
training-time cost-sensitivity and decision-time threshold tuning are redundant or additive.
"""

import numpy as np


def cost_sample_weights(y: np.ndarray, cost_fn: float, cost_fp: float) -> np.ndarray:
    """Per-row training weight: `cost_fn` for a fraud row, `cost_fp` for a legitimate row."""
    y = np.asarray(y)
    return np.where(y == 1, cost_fn, cost_fp).astype(float)

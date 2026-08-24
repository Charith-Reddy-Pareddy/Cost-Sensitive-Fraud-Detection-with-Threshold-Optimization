import numpy as np

from src.models.cost_sensitive_training import cost_sample_weights


def test_weights_assigned_by_class():
    y = np.array([1, 0, 1, 0, 0])
    weights = cost_sample_weights(y, cost_fn=500.0, cost_fp=5.0)
    np.testing.assert_array_equal(weights, [500.0, 5.0, 500.0, 5.0, 5.0])


def test_weights_are_float_array():
    y = np.array([0, 1])
    weights = cost_sample_weights(y, cost_fn=500.0, cost_fp=5.0)
    assert weights.dtype == np.float64

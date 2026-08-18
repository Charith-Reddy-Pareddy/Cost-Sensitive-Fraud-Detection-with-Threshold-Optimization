import numpy as np

from src.models.calibration import (
    apply_isotonic,
    apply_platt_scaling,
    brier_score,
    fit_isotonic,
    fit_platt_scaling,
    reliability_curve,
)


def _miscalibrated_scenario(seed: int, n: int = 2000):
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0, 1, size=n)
    y = rng.binomial(1, p_true)
    raw_score = p_true**3  # systematically under-confident: a monotonic but wrong mapping
    return raw_score, y


def test_brier_score_perfect_predictions_is_zero():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_proba) == 0.0


def test_brier_score_worst_case_predictions_is_one():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_proba) == 1.0


def test_platt_scaling_improves_miscalibrated_scores():
    raw_calib, y_calib = _miscalibrated_scenario(seed=0)
    raw_test, y_test = _miscalibrated_scenario(seed=1)

    raw_brier = brier_score(y_test, raw_test)
    platt_model = fit_platt_scaling(raw_calib, y_calib)
    calibrated_test = apply_platt_scaling(platt_model, raw_test)
    calibrated_brier = brier_score(y_test, calibrated_test)

    assert calibrated_brier < raw_brier


def test_isotonic_improves_miscalibrated_scores():
    raw_calib, y_calib = _miscalibrated_scenario(seed=0)
    raw_test, y_test = _miscalibrated_scenario(seed=1)

    raw_brier = brier_score(y_test, raw_test)
    iso_model = fit_isotonic(raw_calib, y_calib)
    calibrated_test = apply_isotonic(iso_model, raw_test)
    calibrated_brier = brier_score(y_test, calibrated_test)

    assert calibrated_brier < raw_brier


def test_reliability_curve_returns_matching_length_arrays():
    raw_score, y = _miscalibrated_scenario(seed=0)
    prob_true, prob_pred = reliability_curve(y, raw_score, n_bins=10)
    assert len(prob_true) == len(prob_pred)
    assert len(prob_true) <= 10

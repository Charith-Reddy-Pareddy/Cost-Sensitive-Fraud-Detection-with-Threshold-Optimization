"""Probability calibration.

Threshold optimization implicitly assumes the model's scores mean something as probabilities —
that a batch of transactions scored 0.3 really does contain fraud about 30% of the time. Tree
ensembles trained on severely imbalanced data are not guaranteed to have that property. This
module fits Platt scaling (a 1D logistic regression on the raw score) and isotonic regression as
two standard calibration methods, and provides the pieces needed to check whether calibrating
the scores actually changes the cost-optimal threshold.

Calibrators are fit on a held-out *calibration* slice carved out of the training period — never
on the test set — so evaluating calibrated probabilities on the test set stays leakage-free.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def fit_platt_scaling(y_proba_calib: np.ndarray, y_true_calib: np.ndarray) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(np.asarray(y_proba_calib).reshape(-1, 1), y_true_calib)
    return model


def apply_platt_scaling(model: LogisticRegression, y_proba: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(y_proba).reshape(-1, 1))[:, 1]


def fit_isotonic(y_proba_calib: np.ndarray, y_true_calib: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(y_proba_calib, y_true_calib)
    return model


def apply_isotonic(model: IsotonicRegression, y_proba: np.ndarray) -> np.ndarray:
    return model.predict(y_proba)


def reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Equal-frequency ("quantile") binning — with fraud at 0.17% of rows, equal-width bins
    would leave most bins empty of positives; quantile bins keep each bin populated."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return prob_true, prob_pred


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_proba))

"""Comprehensive evaluation metrics for a fixed operating threshold.

PR-AUC is treated here as a model-selection signal, not a general statistics point: under
severe class imbalance (~578 legitimate transactions per fraud in this dataset), a
threshold-independent metric like ROC-AUC is flooded with easy true negatives and makes
performance look better than it is operationally. PR-AUC is far more sensitive to how the
model actually behaves in the region that matters — where fraud is rare.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true, y_proba, k: int) -> float:
    """Precision among the top-k highest-scored transactions — the metric an analyst queue
    of fixed size k would actually experience."""
    if k <= 0:
        return 0.0
    order = np.argsort(-np.asarray(y_proba))[:k]
    return float(np.asarray(y_true)[order].mean())


def evaluate_full(y_true, y_proba, threshold: float = 0.5, k: int = 100) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        f"precision_at_{k}": precision_at_k(y_true, y_proba, k),
    }

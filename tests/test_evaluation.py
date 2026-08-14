import numpy as np

from src.models.evaluation import evaluate_full, precision_at_k


def test_precision_at_k_ranks_by_score():
    y_true = np.array([0, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    # top 2 by score are indices 1 and 3, both fraud -> precision@2 = 1.0
    assert precision_at_k(y_true, y_proba, k=2) == 1.0
    # top 4 by score includes 3 of the frauds/legits mixed: indices 1,3,2,0 -> 2 fraud of 4
    assert precision_at_k(y_true, y_proba, k=4) == 0.5


def test_precision_at_k_zero_k():
    assert precision_at_k([0, 1], [0.1, 0.9], k=0) == 0.0


def test_evaluate_full_confusion_counts():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.9, 0.1])  # 1 TP, 1 FN, 1 FP, 1 TN at threshold 0.5
    metrics = evaluate_full(y_true, y_proba, threshold=0.5, k=2)

    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5


def test_evaluate_full_returns_expected_keys():
    y_true = np.array([1, 0, 1, 0, 1])
    y_proba = np.array([0.8, 0.2, 0.7, 0.3, 0.6])
    metrics = evaluate_full(y_true, y_proba, k=3)
    expected_keys = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "false_positive_rate",
        "false_negative_rate",
        "precision_at_3",
    }
    assert set(metrics) == expected_keys

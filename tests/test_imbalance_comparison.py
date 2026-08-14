import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS
from src.models.imbalance_comparison import train_strategy


def _synthetic_imbalanced_frame(n: int = 300, fraud_rate: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in RAW_FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    n_fraud = int(n * fraud_rate)
    y = pd.Series([0] * (n - n_fraud) + [1] * n_fraud)
    # give fraud rows a distinct signal on V1 so there's something learnable
    df.loc[y == 1, "V1"] += 4.0
    # shuffle so a positional train/test slice below still contains both classes, instead of
    # all fraud rows landing in whichever slice happens to hold the tail of the frame
    shuffled_idx = rng.permutation(n)
    return df.iloc[shuffled_idx].reset_index(drop=True), y.iloc[shuffled_idx].reset_index(drop=True)


def test_all_three_strategies_train_and_evaluate():
    X, y = _synthetic_imbalanced_frame()
    X_train, y_train = X.iloc[:200], y.iloc[:200]
    X_test, y_test = X.iloc[200:], y.iloc[200:]

    for strategy in ("none", "class_weight", "smote"):
        metrics = train_strategy(strategy, X_train, y_train, X_test, y_test)
        assert set(metrics) == {"pr_auc", "roc_auc", "f1_at_0.5"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

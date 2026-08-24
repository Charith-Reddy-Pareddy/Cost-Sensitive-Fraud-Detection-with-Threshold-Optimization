"""Is the selected threshold robust to uncertainty in the assumed cost ratio?

Every other section treats $500/$5 as fixed. Here, instead, `cost_fn` is drawn from
Uniform(100, 1000) and `cost_fp` from Uniform(1, 20) independently, 500 times; for each draw the
cost-optimal threshold is selected on the validation split (never test). The resulting
distribution of thresholds says how much the "right" decision threshold actually moves under
realistic uncertainty about the illustrative cost figures, rather than pretending $500/$5 is
known with certainty.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.cost_engine import optimize_threshold
from src.models.imbalance_comparison import build_pipeline

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
N_DRAWS = 500
COST_FN_RANGE = (100.0, 1000.0)
COST_FP_RANGE = (1.0, 20.0)


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    y_val_arr = y_val.to_numpy()

    pipeline = build_pipeline("class_weight")
    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline.set_params(classifier__scale_pos_weight=n_neg / n_pos)
    pipeline.fit(X_train, y_train)
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    rng = np.random.default_rng(0)
    cost_fn_draws = rng.uniform(*COST_FN_RANGE, size=N_DRAWS)
    cost_fp_draws = rng.uniform(*COST_FP_RANGE, size=N_DRAWS)

    thresholds = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        sweep = optimize_threshold(y_val_arr, val_proba, cost_fn=cost_fn_draws[i], cost_fp=cost_fp_draws[i])
        thresholds[i] = sweep.optimal_threshold

    print(f"cost_fn ~ Uniform{COST_FN_RANGE}, cost_fp ~ Uniform{COST_FP_RANGE}, {N_DRAWS} draws")
    print(f"threshold: mean={thresholds.mean():.3f} std={thresholds.std():.3f}")
    print(f"threshold: median={np.median(thresholds):.3f} IQR=[{np.percentile(thresholds, 25):.3f}, {np.percentile(thresholds, 75):.3f}]")
    print(f"threshold: min={thresholds.min():.3f} max={thresholds.max():.3f}")

    plt.figure(figsize=(7, 4))
    plt.hist(thresholds, bins=30)
    plt.axvline(np.median(thresholds), color="green", linestyle="--", label=f"median = {np.median(thresholds):.3f}")
    plt.xlabel("cost-optimal threshold")
    plt.ylabel("count (of 500 cost-ratio draws)")
    plt.title("Threshold sensitivity to cost uncertainty")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "cost_uncertainty_threshold_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

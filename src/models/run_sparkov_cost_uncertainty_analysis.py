"""Repeats Experiment 3 (cost-ratio uncertainty) on Sparkov — the last experiment from the
primary dataset not yet replicated there (see RESEARCH_REPORT.md Future Work).

Same design: cost_fn ~ Uniform(100, 1000), cost_fp ~ Uniform(1, 20), 500 independent draws, each
selecting its cost-optimal threshold on the validation split.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import TARGET_COLUMN
from src.models.cost_engine import optimize_threshold
from src.models.run_sparkov_bootstrap_analysis import PROCESSED_DIR

FIGURES_DIR = PROCESSED_DIR.parents[2] / "reports" / "figures"
N_DRAWS = 500
COST_FN_RANGE = (100.0, 1000.0)
COST_FP_RANGE = (1.0, 20.0)


def _load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ("Time", TARGET_COLUMN)]
    return df[feature_cols], df[TARGET_COLUMN]


def _build_pipeline(scale_pos_weight: float) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def main() -> None:
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    y_val_arr = y_val.to_numpy()

    n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
    pipeline = _build_pipeline(scale_pos_weight=n_neg / n_pos)
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
    plt.title("Sparkov: threshold sensitivity to cost uncertainty")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "sparkov_cost_uncertainty_threshold_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

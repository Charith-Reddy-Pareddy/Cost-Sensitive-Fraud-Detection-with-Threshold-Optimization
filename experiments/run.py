"""Config-driven experiment entry point: `python -m experiments.run` (defaults to
`configs/primary.yaml`), or `python -m experiments.run --config-name=sparkov`,
`--config-name=cost_sweep`, `--config-name=temporal`, or an override like
`python -m experiments.run cost.fn=1000 model.max_depth=6`.

This project's default pattern is one hardcoded script per experiment (`src/models/run_*.py`,
~30 of them) — appropriate for one-off ablations, but it means changing a cost figure or a
hyperparameter means editing code. This Hydra entry point covers the three operations someone
would actually want to re-run under many different configs — cost-optimal threshold selection,
the cost-ratio sensitivity sweep, and walk-forward temporal evaluation — parameterized by
dataset, cost, model hyperparameters, and threshold-search strategy. It's deliberately not a
rewrite of every existing script behind Hydra: most of those are single-purpose comparisons
(baselines, calibration, ablations) that don't benefit from being re-run under a config sweep,
and mechanically wrapping all ~30 would be a large refactor for little real payoff. See
configs/*.yaml for the available configs.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBClassifier

from src.models.cost_engine import (
    bayes_optimal_threshold,
    cost_ratio_sensitivity_sweep,
    exact_optimal_threshold,
    expected_cost,
    optimize_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_primary_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN

    df = pd.read_parquet(ROOT / "data" / "processed" / f"{name}.parquet")
    return df[RAW_FEATURE_COLUMNS], df[TARGET_COLUMN]


def _load_sparkov_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    from src.data.ingest_sparkov import TARGET_COLUMN

    df = pd.read_parquet(ROOT / "data" / "processed" / "sparkov" / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ("Time", TARGET_COLUMN)]
    return df[feature_cols], df[TARGET_COLUMN]


def _load_primary_full_chronological() -> pd.DataFrame:
    from src.data.ingest import load_raw
    from src.features.pipeline import RAW_FEATURE_COLUMNS, TARGET_COLUMN

    df = load_raw()
    return df[RAW_FEATURE_COLUMNS + [TARGET_COLUMN]]


def _load_sparkov_full_chronological() -> pd.DataFrame:
    from src.data.ingest_sparkov import engineer_features, load_raw

    features = engineer_features(load_raw())
    return features.rename(columns={"unix_time": "Time"})


def _train_model(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    clf = XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        eval_metric="aucpr",
        n_jobs=-1,
        scale_pos_weight=n_neg / n_pos,
    )
    clf.fit(X_train, y_train)
    return clf


def _run_cost_analysis(cfg: DictConfig, X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    model = _train_model(cfg, X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    y_val_arr, y_test_arr = y_val.to_numpy(), y_test.to_numpy()

    if cfg.threshold_search == "exact":
        sweep = exact_optimal_threshold(y_val_arr, val_proba, cfg.cost.fn, cfg.cost.fp)
    elif cfg.threshold_search == "grid":
        sweep = optimize_threshold(y_val_arr, val_proba, cost_fn=cfg.cost.fn, cost_fp=cfg.cost.fp)
    else:
        raise ValueError(f"unknown threshold_search: {cfg.threshold_search}")

    default_cost = expected_cost(y_test_arr, test_proba, 0.5, cfg.cost.fn, cfg.cost.fp)
    optimal_cost = expected_cost(y_test_arr, test_proba, sweep.optimal_threshold, cfg.cost.fn, cfg.cost.fp)
    bayes = bayes_optimal_threshold(cfg.cost.fn, cfg.cost.fp)
    reduction = 100 * (default_cost - optimal_cost) / default_cost if default_cost else 0.0

    print(f"dataset={cfg.dataset} threshold_search={cfg.threshold_search} cost_fn={cfg.cost.fn} cost_fp={cfg.cost.fp}")
    print(f"val-selected threshold: {sweep.optimal_threshold:.4f} (Bayes-optimal: {bayes:.4f})")
    print(f"default (0.5) cost on test: ${default_cost:,.2f}")
    print(f"optimal cost on test: ${optimal_cost:,.2f}  ({reduction:+.2f}%)")


def _run_cost_sweep(cfg: DictConfig, X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    model = _train_model(cfg, X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    y_val_arr = y_val.to_numpy()

    ratios = np.array(list(cfg.cost_ratios))
    results = cost_ratio_sensitivity_sweep(y_val_arr, val_proba, cost_ratios=ratios, cost_fp=cfg.cost.fp)
    print(f"dataset={cfg.dataset} cost-ratio sensitivity sweep (thresholds selected on val):")
    for r in results:
        print(f"  ratio={r['cost_ratio']:.0f} -> optimal_threshold={r['optimal_threshold']:.4f}")


def _run_temporal(cfg: DictConfig) -> None:
    load_full = _load_primary_full_chronological if cfg.dataset == "primary" else _load_sparkov_full_chronological
    full = load_full()

    target_col = "Class" if cfg.dataset == "primary" else "is_fraud"
    feature_cols = [c for c in full.columns if c not in ("Time", target_col)]

    ordered = full.sort_values("Time").reset_index(drop=True)
    n_blocks = cfg.n_blocks
    edges = np.linspace(0, len(ordered), n_blocks + 1).astype(int)
    blocks = [ordered.iloc[edges[i] : edges[i + 1]].reset_index(drop=True) for i in range(n_blocks)]

    print(f"dataset={cfg.dataset} walk-forward, {n_blocks} blocks, threshold_search={cfg.threshold_search}")
    print("| Fold | Train rows | Threshold (val) | Default cost (test) | Optimized cost (test) | Result |")
    for k in range(1, n_blocks):
        train_df = pd.concat(blocks[:k], ignore_index=True)
        window = blocks[k]
        half = len(window) // 2
        val_df, test_df = window.iloc[:half], window.iloc[half:]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col].to_numpy()
        X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

        if y_val.sum() == 0 or y_test.sum() == 0:
            print(f"| {k} | {len(train_df)} | — | — | — | skipped: no fraud in val or test half |")
            continue

        model = _train_model(cfg, X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        sweep = exact_optimal_threshold(y_val, val_proba, cfg.cost.fn, cfg.cost.fp)
        default_cost = expected_cost(y_test, test_proba, 0.5, cfg.cost.fn, cfg.cost.fp)
        optimal_cost = expected_cost(y_test, test_proba, sweep.optimal_threshold, cfg.cost.fn, cfg.cost.fp)
        result = "improved" if optimal_cost < default_cost else "tied" if optimal_cost == default_cost else "worsened"

        print(
            f"| {k} | {len(train_df)} | {sweep.optimal_threshold:.4f} | ${default_cost:,.2f} | "
            f"${optimal_cost:,.2f} | {result} |"
        )


@hydra.main(config_path="../configs", config_name="primary", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "temporal":
        _run_temporal(cfg)
        return

    load_split = _load_primary_split if cfg.dataset == "primary" else _load_sparkov_split
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    if cfg.mode == "cost_analysis":
        _run_cost_analysis(cfg, X_train, y_train, X_val, y_val, X_test, y_test)
    elif cfg.mode == "cost_sweep":
        _run_cost_sweep(cfg, X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()

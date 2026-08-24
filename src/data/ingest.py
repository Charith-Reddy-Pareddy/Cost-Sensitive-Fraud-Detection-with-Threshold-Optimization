"""Load the raw Kaggle creditcard.csv and produce a leakage-safe chronological split.

The dataset is a single day of transactions ordered by `Time` (seconds since the first
transaction). A random train/test split would let the model see transactions that happen
"after" the ones it's tested on, which does not reflect how the model would actually be
used in production. Splitting chronologically instead means the test set is strictly later
in time than the training set, matching real deployment.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "creditcard.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def chronological_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on `Time` order rather than randomly, so no future transaction leaks into training."""
    ordered = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_size))
    train = ordered.iloc[:split_idx].reset_index(drop=True)
    test = ordered.iloc[split_idx:].reset_index(drop=True)
    return train, test


def three_way_chronological_split(
    df: pd.DataFrame, val_size: float = 0.17, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test, all chronological. Validation is where every model-selection
    decision happens (threshold, calibration) — test is touched exactly once, for final
    reporting, so it stays a genuine held-out evaluation rather than something optimized
    against."""
    ordered = df.sort_values("Time").reset_index(drop=True)
    n = len(ordered)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    test_start = n - n_test
    val_start = test_start - n_val
    train = ordered.iloc[:val_start].reset_index(drop=True)
    val = ordered.iloc[val_start:test_start].reset_index(drop=True)
    test = ordered.iloc[test_start:].reset_index(drop=True)
    return train, val, test


def main() -> None:
    df = load_raw()
    train, val, test = three_way_chronological_split(df)

    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)
    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)

    print(f"train: {len(train)} rows ({train['Class'].sum()} fraud)")
    print(f"val:   {len(val)} rows ({val['Class'].sum()} fraud)")
    print(f"test:  {len(test)} rows ({test['Class'].sum()} fraud)")


if __name__ == "__main__":
    main()

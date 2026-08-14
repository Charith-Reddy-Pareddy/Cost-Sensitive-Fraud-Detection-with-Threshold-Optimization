import pandas as pd

from src.data.ingest import chronological_split


def _toy_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": range(n),
            "Amount": [1.0] * n,
            "Class": [0] * n,
        }
    )


def test_split_preserves_all_rows():
    df = _toy_frame(100)
    train, test = chronological_split(df, test_size=0.2)
    assert len(train) + len(test) == 100


def test_split_is_chronological_not_random():
    df = _toy_frame(100)
    train, test = chronological_split(df, test_size=0.2)
    assert train["Time"].max() < test["Time"].min()


def test_default_split_ratio():
    df = _toy_frame(100)
    train, test = chronological_split(df)
    assert len(test) == 20

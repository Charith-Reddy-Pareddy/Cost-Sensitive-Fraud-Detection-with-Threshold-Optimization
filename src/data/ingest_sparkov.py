"""Ingest and feature-engineer the Sparkov-simulated fraud dataset — the second dataset used to
check whether this project's conclusions (class-weighting beats the unweighted baseline,
threshold optimization beats the default) generalize beyond the primary, heavily-anonymized
Kaggle dataset.

Unlike the primary dataset (PCA components only), Sparkov has merchant, category, and
geolocation fields, which is exactly what enables real feature engineering instead of working
only with anonymized components: a customer-to-merchant distance, transaction hour, customer
age, and city population, alongside the raw amount and one-hot category.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "sparkov" / "fraudTrain.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "sparkov"

USE_COLUMNS = [
    "trans_date_trans_time",
    "category",
    "amt",
    "gender",
    "lat",
    "long",
    "city_pop",
    "dob",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

TARGET_COLUMN = "is_fraud"


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r_earth_km = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r_earth_km * 2 * np.arcsin(np.sqrt(a))


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, usecols=USE_COLUMNS)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    trans_time = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])

    df["hour"] = trans_time.dt.hour
    df["age_years"] = (trans_time - dob).dt.days / 365.25
    df["distance_km"] = _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["is_male"] = (df["gender"] == "M").astype(int)

    category_dummies = pd.get_dummies(df["category"], prefix="category", dtype=int)

    features = pd.concat(
        [df[["amt", "hour", "age_years", "distance_km", "city_pop", "is_male", "unix_time"]], category_dummies],
        axis=1,
    )
    features[TARGET_COLUMN] = df[TARGET_COLUMN].values
    return features


def main() -> None:
    from src.data.ingest import three_way_chronological_split

    df = load_raw()
    features = engineer_features(df)
    # three_way_chronological_split expects a "Time" column to sort/split on
    features = features.rename(columns={"unix_time": "Time"})

    train, val, test = three_way_chronological_split(features)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print(f"train: {len(train)} rows ({train[TARGET_COLUMN].sum()} fraud)")
    print(f"val:   {len(val)} rows ({val[TARGET_COLUMN].sum()} fraud)")
    print(f"test:  {len(test)} rows ({test[TARGET_COLUMN].sum()} fraud)")
    print(f"feature columns: {[c for c in train.columns if c not in ('Time', TARGET_COLUMN)]}")


if __name__ == "__main__":
    main()

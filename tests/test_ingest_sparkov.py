import pandas as pd

from src.data.ingest_sparkov import _haversine_km, engineer_features


def test_haversine_zero_distance_for_same_point():
    import numpy as np

    d = _haversine_km(np.array([40.0]), np.array([-74.0]), np.array([40.0]), np.array([-74.0]))
    assert d[0] == 0.0


def test_haversine_known_distance_nyc_to_la():
    import numpy as np

    # NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) is ~3936 km
    d = _haversine_km(np.array([40.7128]), np.array([-74.0060]), np.array([34.0522]), np.array([-118.2437]))
    assert 3800 < d[0] < 4050


def _toy_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_date_trans_time": ["2019-01-01 12:00:00", "2019-01-02 03:00:00"],
            "cc_num": [1, 2],
            "category": ["grocery_pos", "shopping_net"],
            "amt": [50.0, 200.0],
            "gender": ["F", "M"],
            "lat": [40.0, 41.0],
            "long": [-74.0, -75.0],
            "city_pop": [10000, 20000],
            "dob": ["1990-01-01", "1985-06-15"],
            "unix_time": [1546344000, 1546398000],
            "merch_lat": [40.1, 41.2],
            "merch_long": [-74.1, -75.2],
            "is_fraud": [0, 1],
        }
    )


def test_engineer_features_produces_expected_columns():
    features = engineer_features(_toy_raw_frame())
    expected_cols = [
        "amt", "hour", "age_years", "distance_km", "city_pop", "is_male",
        "card_txn_count_24h", "card_amt_sum_24h", "unix_time", "is_fraud",
    ]
    for col in expected_cols:
        assert col in features.columns
    assert "category_grocery_pos" in features.columns
    assert "category_shopping_net" in features.columns


def test_engineer_features_hour_extracted_correctly():
    features = engineer_features(_toy_raw_frame())
    assert list(features["hour"]) == [12, 3]


def test_engineer_features_is_male_binary():
    features = engineer_features(_toy_raw_frame())
    assert list(features["is_male"]) == [0, 1]

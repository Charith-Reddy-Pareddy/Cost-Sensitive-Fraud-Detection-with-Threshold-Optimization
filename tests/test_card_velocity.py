import pandas as pd

from src.data.ingest_sparkov import add_card_velocity_features


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cc_num": [1, 1, 1, 2, 2],
            "trans_date_trans_time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",  # card1, A: first ever -> nothing before it
                    "2020-01-01 01:00:00",  # card1, B: 1h after A -> sees A
                    "2020-01-02 00:00:00",  # card1, C: 24h after A, 23h after B -> sees both
                    "2020-01-01 00:00:00",  # card2, D: first ever
                    "2020-01-01 12:00:00",  # card2, E: 12h after D -> sees D
                ]
            ),
            "amt": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )


def test_card_velocity_is_causal_and_correct():
    result = add_card_velocity_features(_toy_frame(), window="24h")
    result = result.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    counts = result["card_txn_count_24h"].tolist()
    sums = result["card_amt_sum_24h"].tolist()

    # rows in order: card1-A, card1-B, card1-C, card2-D, card2-E
    assert counts == [0, 1, 2, 0, 1]
    assert sums == [0.0, 10.0, 30.0, 0.0, 100.0]


def test_card_velocity_never_counts_its_own_row():
    # a single transaction, alone, must always see zero prior transactions
    df = pd.DataFrame(
        {
            "cc_num": [1],
            "trans_date_trans_time": pd.to_datetime(["2020-01-01 00:00:00"]),
            "amt": [50.0],
        }
    )
    result = add_card_velocity_features(df, window="24h")
    assert result["card_txn_count_24h"].iloc[0] == 0
    assert result["card_amt_sum_24h"].iloc[0] == 0.0


def test_card_velocity_outside_window_is_excluded():
    df = pd.DataFrame(
        {
            "cc_num": [1, 1],
            "trans_date_trans_time": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-03 00:00:00"]),
            "amt": [10.0, 20.0],
        }
    )
    result = add_card_velocity_features(df, window="24h")
    result = result.sort_values("trans_date_trans_time").reset_index(drop=True)
    # second row is 48h after the first, well outside the 24h window
    assert result["card_txn_count_24h"].iloc[1] == 0
    assert result["card_amt_sum_24h"].iloc[1] == 0.0

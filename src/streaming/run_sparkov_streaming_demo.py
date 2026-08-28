"""Connects the Redis per-card sliding window to a live Sparkov model: for each replayed
transaction, `card_txn_count_24h` / `card_amt_sum_24h` come from Redis in real time and are fed
directly into `pipeline.predict_proba(...)` — not just computed and logged alongside a
prediction, the way the primary dataset's streaming demo explicitly can't do (no entity ID
there). This is what closes that gap.

Trains fresh on the full engineered Sparkov training set (same as run_sparkov_validation.py),
then replays the most recent `N_REPLAY_ROWS` transactions in time order against Redis.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.ingest_sparkov import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    engineer_features,
    load_raw,
    raw_row_to_static_features,
)
from src.streaming.redis_features_sparkov import card_velocity_features, get_redis_client

N_REPLAY_ROWS = 5000


def _build_pipeline(scale_pos_weight: float) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="aucpr", n_jobs=-1, scale_pos_weight=scale_pos_weight
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("classifier", classifier)])


def main() -> None:
    raw = load_raw()
    features_full = engineer_features(raw)

    n_pos = int(features_full[TARGET_COLUMN].sum())
    n_neg = len(features_full) - n_pos
    pipeline = _build_pipeline(scale_pos_weight=n_neg / n_pos)
    pipeline.fit(features_full[FEATURE_COLUMNS], features_full[TARGET_COLUMN])

    replay = raw.sort_values("unix_time").tail(N_REPLAY_ROWS).reset_index(drop=True)
    redis_client = get_redis_client()

    flagged = 0
    correct = 0
    fraud_rows_seen = 0
    for _, row in replay.iterrows():
        now_ms = int(row["unix_time"]) * 1000
        velocity = card_velocity_features(redis_client, row["cc_num"], row["amt"], now_ms=now_ms)
        static = raw_row_to_static_features(row)
        feature_vector = pd.DataFrame([{**static, **velocity}])[FEATURE_COLUMNS]

        proba = pipeline.predict_proba(feature_vector)[0, 1]
        predicted_fraud = proba >= 0.5
        true_fraud = int(row[TARGET_COLUMN]) == 1

        flagged += predicted_fraud
        correct += predicted_fraud == true_fraud
        if true_fraud:
            fraud_rows_seen += 1
            print(
                f"cc_num={row['cc_num']} amt=${row['amt']:.2f} "
                f"live_card_txn_count_24h={velocity['card_txn_count_24h']} "
                f"live_card_amt_sum_24h=${velocity['card_amt_sum_24h']:.2f} "
                f"fraud_probability={proba:.4f} predicted={'fraud' if predicted_fraud else 'legit'}"
            )

    print(
        f"\nreplayed {len(replay)} transactions ({fraud_rows_seen} fraud), "
        f"flagged {flagged}, accuracy {correct / len(replay):.3f}"
    )


if __name__ == "__main__":
    main()

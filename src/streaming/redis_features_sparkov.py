"""Per-card Redis sliding-window feature store for Sparkov.

Unlike the primary dataset's global window (no entity ID available there — see
`redis_features.py`), this is keyed by `cc_num`, and computes the *same* causal window the
offline training feature (`card_txn_count_24h` / `card_amt_sum_24h` in
`src/data/ingest_sparkov.py`) uses: strictly prior transactions only, current transaction
excluded from its own count. That's what makes it safe to feed straight into the model instead
of just logging it alongside a prediction.
"""

import os
import time

import redis

WINDOW_SECONDS = 24 * 60 * 60


def get_redis_client() -> redis.Redis:
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, decode_responses=True)


def _key(cc_num) -> str:
    return f"card_window:{cc_num}"


def card_velocity_features(client: redis.Redis, cc_num, amount: float, now_ms: int | None = None) -> dict:
    """Returns this transaction's live `card_txn_count_24h` / `card_amt_sum_24h`, computed from
    transactions strictly before `now_ms`, then records this transaction for future lookups —
    in that order, so a transaction never counts itself."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    key = _key(cc_num)

    client.zremrangebyscore(key, "-inf", now_ms - WINDOW_SECONDS * 1000)
    members = client.zrange(key, 0, -1)
    amounts = [float(m.split(":")[1]) for m in members]
    features = {"card_txn_count_24h": len(amounts), "card_amt_sum_24h": sum(amounts)}

    member = f"{now_ms}:{amount}:{time.perf_counter_ns()}"
    client.zadd(key, {member: now_ms})

    return features

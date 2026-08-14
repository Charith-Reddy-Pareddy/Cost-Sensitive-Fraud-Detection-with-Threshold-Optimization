"""Redis-backed sliding-window feature store.

The primary Kaggle dataset has no card/merchant/entity ID to key a per-entity window on (it's
anonymized PCA components plus Amount/Time only) — see the README for why. So this window is
global rather than per-entity: a rolling transaction count and amount sum over the last
`WINDOW_SECONDS` of *stream* time, kept as a Redis sorted set (score = event timestamp in ms,
so old entries can be trimmed with `ZREMRANGEBYSCORE`).
"""

import os
import time

import redis

WINDOW_SECONDS = 30
REDIS_KEY = "txn_sliding_window"


def get_redis_client() -> redis.Redis:
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, decode_responses=True)


def record_transaction(client: redis.Redis, amount: float, now_ms: int | None = None) -> None:
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    # member must be unique per event even if two transactions share an amount and millisecond
    member = f"{now_ms}:{amount}:{time.perf_counter_ns()}"
    client.zadd(REDIS_KEY, {member: now_ms})
    client.zremrangebyscore(REDIS_KEY, "-inf", now_ms - WINDOW_SECONDS * 1000)


def sliding_window_stats(client: redis.Redis, now_ms: int | None = None) -> dict:
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    client.zremrangebyscore(REDIS_KEY, "-inf", now_ms - WINDOW_SECONDS * 1000)
    members = client.zrange(REDIS_KEY, 0, -1)
    amounts = [float(m.split(":")[1]) for m in members]
    return {
        "window_seconds": WINDOW_SECONDS,
        "transaction_count": len(amounts),
        "amount_sum": sum(amounts),
    }

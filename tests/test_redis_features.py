import time

import pytest

from src.streaming.redis_features import get_redis_client, record_transaction, sliding_window_stats


def _redis_available() -> bool:
    try:
        get_redis_client().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="requires a running redis (e.g. via docker compose)")


@pytest.fixture()
def client():
    c = get_redis_client()
    c.delete("txn_sliding_window")
    yield c
    c.delete("txn_sliding_window")


def test_records_and_counts_transactions(client):
    now_ms = int(time.time() * 1000)
    record_transaction(client, amount=10.0, now_ms=now_ms)
    record_transaction(client, amount=25.0, now_ms=now_ms)

    stats = sliding_window_stats(client, now_ms=now_ms)
    assert stats["transaction_count"] == 2
    assert stats["amount_sum"] == 35.0


def test_expires_entries_outside_window(client):
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - 60_000  # 60s ago, outside the 30s window

    record_transaction(client, amount=100.0, now_ms=old_ms)
    record_transaction(client, amount=5.0, now_ms=now_ms)

    stats = sliding_window_stats(client, now_ms=now_ms)
    assert stats["transaction_count"] == 1
    assert stats["amount_sum"] == 5.0

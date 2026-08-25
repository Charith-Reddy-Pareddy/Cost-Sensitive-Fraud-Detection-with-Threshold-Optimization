import pytest

from src.streaming.redis_features_sparkov import card_velocity_features, get_redis_client


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
    c.delete("card_window:CARD_A", "card_window:CARD_B")
    yield c
    c.delete("card_window:CARD_A", "card_window:CARD_B")


def test_first_transaction_sees_nothing(client):
    features = card_velocity_features(client, "CARD_A", amount=50.0, now_ms=1_000_000)
    assert features == {"card_txn_count_24h": 0, "card_amt_sum_24h": 0.0}


def test_second_transaction_sees_the_first_not_itself(client):
    card_velocity_features(client, "CARD_A", amount=50.0, now_ms=1_000_000)
    features = card_velocity_features(client, "CARD_A", amount=30.0, now_ms=1_000_000 + 60_000)
    assert features == {"card_txn_count_24h": 1, "card_amt_sum_24h": 50.0}


def test_different_cards_do_not_share_state(client):
    card_velocity_features(client, "CARD_A", amount=50.0, now_ms=1_000_000)
    features = card_velocity_features(client, "CARD_B", amount=30.0, now_ms=1_000_000 + 60_000)
    assert features == {"card_txn_count_24h": 0, "card_amt_sum_24h": 0.0}


def test_transactions_outside_window_expire(client):
    card_velocity_features(client, "CARD_A", amount=50.0, now_ms=0)
    # 25 hours later, outside the 24h window
    features = card_velocity_features(client, "CARD_A", amount=30.0, now_ms=25 * 60 * 60 * 1000)
    assert features == {"card_txn_count_24h": 0, "card_amt_sum_24h": 0.0}

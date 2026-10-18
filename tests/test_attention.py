import numpy as np

from cryptogo.attention import DAY, align_completed_day, attention_block, daily_surprise


def test_flat_attention_has_no_surprise():
    values = np.full(40, 100.0)
    surprise, burst = daily_surprise(values)
    assert np.allclose(surprise[20:], 0.0)
    assert np.allclose(burst[20:], 0.0)


def test_yesterday_spike_is_visible_and_today_is_not():
    days = np.arange(40, dtype=np.int64) * DAY
    values = np.full(40, 10.0)
    values[20] = 10_000.0
    surprise, _burst = daily_surprise(values)
    during = np.array([20 * DAY + 3 * 3600])
    after = np.array([21 * DAY + 3600])
    seen_today, observed_today = align_completed_day(days, surprise, during)
    seen_after, observed_after = align_completed_day(days, surprise, after)
    assert observed_today[0] == 1.0
    assert abs(seen_today[0]) < 1e-5
    assert observed_after[0] == 1.0
    assert seen_after[0] > 1.0


def test_missing_source_is_unobserved():
    decision = np.array([10 * DAY], dtype=np.int64)
    block = attention_block(decision, None, None, None, None)
    assert block.shape == (1, 6)
    assert np.all(block == 0)

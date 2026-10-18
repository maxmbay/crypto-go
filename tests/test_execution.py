import numpy as np

from cryptogo.execution import corwin_schultz, limit_through, round_trip_fills, trailing_mean


def test_corwin_schultz_on_a_repeated_two_percent_range():
    high = np.array([1.02, 1.02])
    low = np.array([0.98, 0.98])
    spread = corwin_schultz(high, low)
    assert spread[0] != spread[0]
    # 1.02/0.98 is a 4% high-low. Repeated, the two-day range equals the one-day range,
    # and Corwin-Schultz attributes that range to the spread.
    assert abs(spread[1] - 0.04) < 1e-4


def test_limit_at_the_last_close_needs_a_print_through_it():
    close = np.array([10.0, 10.0, 10.5])
    low = np.array([10.0, 9.0, 10.4])
    high = np.array([10.0, 10.2, 11.0])
    buy, sell, valid = limit_through(close, high, low)
    assert valid.all()
    assert buy[0] and not buy[1]
    assert sell[0] and sell[1]


def test_round_trip_needs_both_sides():
    buy = np.array([True, True, False])
    sell = np.array([True, False, True])
    valid = np.array([True, True, True])
    ok = round_trip_fills(buy, sell, valid, horizon=2)
    assert ok[0]
    assert not ok[1]


def test_trailing_mean_skips_missing_hours():
    values = np.array([1.0, np.nan, 3.0])
    out = trailing_mean(values, window=3)
    assert abs(out[2] - 2.0) < 1e-9

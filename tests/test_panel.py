import numpy as np

from cryptogo.ledger import simulate
from cryptogo.panel import HOUR, build_samples_for_asset, is_eligible_base, parse_pair


def test_parse_usd_pairs():
    assert parse_pair("XBTUSD") == ("BTC", "USD")
    assert parse_pair("XXBTZUSD") == ("BTC", "USD")
    assert parse_pair("SOLUSD") == ("SOL", "USD")
    assert parse_pair("ETHUSDT") is None
    assert parse_pair("USDTZUSD") == ("USDT", "USD")
    assert is_eligible_base("USDT") is False
    assert is_eligible_base("EUR") is False
    assert is_eligible_base("SOL") is True


def test_label_uses_future_opens_only():
    n = 100
    ts = np.arange(n, dtype=np.int64) * HOUR
    close = np.exp(0.001 * np.arange(n)).astype(np.float64)
    open_obs = close.copy()
    high = close * 1.01
    low = close * 0.99
    volume = np.ones(n)
    trades = np.ones(n)
    zeros = np.zeros(n, dtype=np.float32)
    samples = build_samples_for_asset(
        ts, open_obs, close, high, low, volume, trades, zeros, zeros
    )
    # First kept decision is t=48: the t=47 window still contains the empty return at 0.
    t = 48
    expected = np.log(open_obs[t + 1 + 24] / open_obs[t + 1])
    assert np.isclose(samples["y"][0], expected)
    assert samples["ts"][0] == ts[t]

    changed_open = open_obs.copy()
    changed_open[t + 1] = open_obs[t + 1] * 1.1
    changed = build_samples_for_asset(
        ts, changed_open, close, high, low, volume, trades, zeros, zeros
    )
    assert not np.isclose(changed["y"][0], samples["y"][0])
    assert np.allclose(changed["x"][0], samples["x"][0])

    # The entry bar's close is not a feature. Blanking it must keep the sample.
    broken_close = close.copy()
    broken_close[t + 1] = np.nan
    kept = build_samples_for_asset(
        ts, open_obs, broken_close, high, low, volume, trades, zeros, zeros
    )
    assert kept["ts"][0] == ts[t]
    assert np.allclose(kept["x"][0], samples["x"][0])


def test_ledger_sizes_and_blocks_reentry():
    ts = np.array([0, HOUR], dtype=np.int64)
    asset = np.array([0, 0], dtype=np.int16)
    pred = np.array([0.1, 0.1], dtype=np.float32)
    y = np.array([0.05, 0.2], dtype=np.float32)
    result = simulate(ts, asset, pred, y, fee_per_side=0.0, max_positions=5)
    # One slot is a fifth of equity. The second hour is still inside the hold.
    assert result["trades"] == 1
    assert np.isclose(result["wealth"], 0.8 + 0.2 * np.exp(0.05))


def test_ledger_scales_a_slot_by_the_supplied_weight():
    ts = np.array([0], dtype=np.int64)
    asset = np.array([0], dtype=np.int16)
    pred = np.array([0.1], dtype=np.float32)
    y = np.array([0.05], dtype=np.float32)
    result = simulate(ts, asset, pred, y, fee_per_side=0.0, max_positions=5, size=np.array([0.5]))
    assert result["trades"] == 1
    assert np.isclose(result["mean_size"], 0.5)
    assert np.isclose(result["wealth"], 0.9 + 0.1 * np.exp(0.05))


def test_ledger_respects_the_fee_hurdle():
    ts = np.array([0], dtype=np.int64)
    asset = np.array([0], dtype=np.int16)
    pred = np.array([0.001], dtype=np.float32)
    y = np.array([0.05], dtype=np.float32)
    result = simulate(ts, asset, pred, y, fee_per_side=0.003, max_positions=5)
    assert result["trades"] == 0
    assert result["wealth"] == 1.0


def test_ledger_picks_the_higher_score():
    ts = np.array([0, 0], dtype=np.int64)
    asset = np.array([0, 1], dtype=np.int16)
    pred = np.array([0.01, 0.20], dtype=np.float32)
    y = np.array([0.0, 0.10], dtype=np.float32)
    result = simulate(ts, asset, pred, y, fee_per_side=0.0, max_positions=1)
    assert result["trades"] == 1
    assert np.isclose(result["wealth"], 0.0 + 1.0 * np.exp(0.10))

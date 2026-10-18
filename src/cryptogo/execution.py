"""Spread and maker-fill estimates from hourly highs and lows.

Kraken's hourly files do not contain the book. The Corwin-Schultz estimator
is a spread implied by the high-low range. A resting buy at the last traded
price fills on the next hour only when that hour prints at or below it.
Neither number is a queue position.
"""

from __future__ import annotations

import math

import numpy as np


def corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Proportional spread for each hour after the first. Uses that hour and the one before it."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    spread = np.full(high.shape[0], np.nan, dtype=np.float64)
    left_high, right_high = high[:-1], high[1:]
    left_low, right_low = low[:-1], low[1:]
    usable = (
        np.isfinite(left_high)
        & np.isfinite(right_high)
        & np.isfinite(left_low)
        & np.isfinite(right_low)
        & (left_high > left_low)
        & (right_high > right_low)
        & (left_low > 0)
        & (right_low > 0)
    )
    if not np.any(usable):
        return spread
    beta = np.full(usable.shape[0], np.nan)
    gamma = np.full(usable.shape[0], np.nan)
    beta[usable] = np.log(left_high[usable] / left_low[usable]) ** 2 + np.log(
        right_high[usable] / right_low[usable]
    ) ** 2
    two_high = np.maximum(left_high, right_high)
    two_low = np.minimum(left_low, right_low)
    gamma[usable] = np.log(two_high[usable] / two_low[usable]) ** 2
    k = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = np.where(np.isfinite(alpha), np.maximum(alpha, 0.0), np.nan)
    value = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread[1:] = value
    return spread


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` finite observations, using only values at or before each index."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    counts = np.cumsum(finite.astype(np.float64))
    totals = np.cumsum(filled)
    for i in range(values.shape[0]):
        start = i - window
        total = totals[i] - (totals[start] if start >= 0 else 0.0)
        count = counts[i] - (counts[start] if start >= 0 else 0.0)
        if count > 0:
            out[i] = total / count
    return out


def limit_through(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whether a limit at close[t] trades on the next hour.

    Buy fills when the next low is at or below the close. Sell fills when the
    next high is at or above it. The returned arrays are indexed by the decision hour.
    """
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = close.shape[0] - 1
    valid = (
        np.isfinite(close[:-1])
        & (close[:-1] > 0)
        & np.isfinite(low[1:])
        & np.isfinite(high[1:])
        & (low[1:] > 0)
        & (high[1:] > 0)
    )
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    buy[valid] = low[1:][valid] <= close[:-1][valid]
    sell[valid] = high[1:][valid] >= close[:-1][valid]
    return buy, sell, valid


def summarize_fills(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ts: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    """Spread and fill rates for decisions whose timestamp is in [start, end)."""
    spread = trailing_mean(corwin_schultz(high, low), 24)
    buy, sell, valid = limit_through(close, high, low)
    both = round_trip_fills(buy, sell, valid, 24)
    # Decision t uses close[t], which is index t on the grid. buy has length n-1.
    decision_ts = ts[:-1]
    in_window = (decision_ts >= start) & (decision_ts < end)
    usable = in_window & valid
    spreads = spread[:-1][usable]
    spreads = spreads[np.isfinite(spreads)]
    defined = np.zeros(both.shape[0], dtype=bool)
    if both.shape[0] > 24:
        defined[:-24] = valid[:-24] & valid[24:]
    trip = both[in_window & defined]
    return {
        "hours": float(usable.sum()),
        "buy_fill": float(buy[usable].mean()) if np.any(usable) else float("nan"),
        "sell_fill": float(sell[usable].mean()) if np.any(usable) else float("nan"),
        "round_trip_fill": float(trip.mean()) if trip.size else float("nan"),
        "spread_p50": float(np.median(spreads)) if spreads.size else float("nan"),
        "spread_p90": float(np.quantile(spreads, 0.9)) if spreads.size else float("nan"),
        "spread_over_15bp": float(np.mean(spreads > 0.0015)) if spreads.size else float("nan"),
        "spread_over_30bp": float(np.mean(spreads > 0.0030)) if spreads.size else float("nan"),
        "range_p50": _median_range(high, low, close, ts, start, end),
    }


def _median_range(high, low, close, ts, start, end) -> float:
    mask = (ts >= start) & (ts < end) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (close > 0)
    if not np.any(mask):
        return float("nan")
    width = (high[mask] - low[mask]) / close[mask]
    width = width[np.isfinite(width) & (width >= 0)]
    return float(np.median(width)) if width.size else float("nan")


def round_trip_fills(buy: np.ndarray, sell: np.ndarray, valid: np.ndarray, horizon: int = 24) -> np.ndarray:
    """Decision hours where the buy fills next hour and the sell fills `horizon` hours later."""
    n = buy.shape[0]
    ok = np.zeros(n, dtype=bool)
    if n <= horizon:
        return ok
    entry = buy[:-horizon] & valid[:-horizon]
    exit_ = sell[horizon:] & valid[horizon:]
    ok[:-horizon] = entry & exit_
    return ok


def main() -> None:
    import json
    from pathlib import Path

    import polars as pl

    from cryptogo.panel import HOUR, TEST_START, VAL_START, discover_hourly_files, load_pair, parse_pair, select_universe

    root = Path(__file__).resolve().parents[2]
    grouped = discover_hourly_files(root / "data" / "kraken" / "hourly")
    selection = select_universe(grouped)
    rows = []
    for stem, paths in selection:
        frame = load_pair(paths)
        start = int(frame["ts"].min())
        end = int(frame["ts"].max())
        grid = pl.DataFrame({"ts": np.arange(start, end + HOUR, HOUR, dtype=np.int64)})
        merged = grid.join(frame.select(["ts", "high", "low", "close"]), on="ts", how="left").sort("ts")
        ts = merged["ts"].to_numpy()
        high = merged["high"].to_numpy()
        low = merged["low"].to_numpy()
        close = merged["close"].to_numpy()
        base = stem
        parsed = parse_pair(stem)
        if parsed is not None:
            base = parsed[0]
        windows = {
            "val": summarize_fills(close, high, low, ts, VAL_START, TEST_START),
            "test": summarize_fills(close, high, low, ts, TEST_START, int(ts.max()) + HOUR),
        }
        rows.append({"asset": base, "stem": stem, **{f"{key}_{name}": value for key, stats in windows.items() for name, value in stats.items()}})
        test = windows["test"]
        print(
            f"{base:<6} test spread p50={test['spread_p50']:.4%} p90={test['spread_p90']:.4%} "
            f"buy={test['buy_fill']:.3f} sell={test['sell_fill']:.3f} trip={test['round_trip_fill']:.3f} "
            f"range={test['range_p50']:.4%}",
            flush=True,
        )
    out = root / "outputs" / "execution_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

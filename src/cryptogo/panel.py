"""Kraken hourly panel, point-in-time universe, and 24-hour labels.

A sample at bar t uses only bars through t. The label buys the next hour's
observed open and sells the open 24 hours after that. Missing hours stay
missing: a short close fill (at most three hours) is only for feature
continuity, and it never creates a trade price.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

HOUR = 3600
LOOKBACK = 48
HORIZON = 24
# Bars are loaded from 2020 so 2021 training windows have history.
LOAD_START = 1577836800  # 2020-01-01 UTC
TRAIN_START = 1609459200  # 2021-01-01 UTC
VAL_START = 1751328000  # 2025-07-01 UTC
TEST_START = 1767225600  # 2026-01-01 UTC
MIN_HISTORY_BARS = 24 * 365
MIN_UNIVERSE_HOURS = 2000
UNIVERSE_SIZE = 24
FILL_LIMIT = 3

STABLES = {
    "USDT",
    "USDC",
    "DAI",
    "USDE",
    "USDS",
    "PYUSD",
    "TUSD",
    "FDUSD",
    "USDP",
    "GUSD",
    "USD1",
    "RLUSD",
    "USDG",
    "USDD",
    "FRAX",
    "LUSD",
    "CRVUSD",
    "GHO",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    "AUD",
    "CHF",
}
FORCED = ("BTC", "ETH")
FEATURE_NAMES = ("ret_1", "range", "log_vol", "log_trades", "btc_ret", "eth_ret")
N_FEATURES = len(FEATURE_NAMES)


def parse_pair(stem: str) -> tuple[str, str] | None:
    """Return (base, quote) for a Kraken USD pair stem, else None."""
    if stem.endswith("ZUSD"):
        base, quote = stem[:-4], "USD"
    elif stem.endswith("USD") and not stem.endswith("USDC"):
        base, quote = stem[:-3], "USD"
    else:
        return None
    if not base:
        return None
    return normalize_base(base), quote


def normalize_base(base: str) -> str:
    aliases = {
        "XBT": "BTC",
        "XXBT": "BTC",
        "XDG": "DOGE",
        "XXDG": "DOGE",
        "XETH": "ETH",
        "XLTC": "LTC",
        "XXRP": "XRP",
        "XXLM": "XLM",
        "XETC": "ETC",
        "XREP": "REP",
        "XZEC": "ZEC",
        "XMLN": "MLN",
    }
    return aliases.get(base, base)


def is_eligible_base(base: str) -> bool:
    if base in STABLES:
        return False
    upper = base.upper()
    if any(tag in upper for tag in ("3L", "3S", "BULL", "BEAR")):
        return False
    if upper.endswith(("UP", "DOWN")) and len(upper) > 4:
        return False
    return True


def pair_stem(path: Path) -> str | None:
    name = path.name
    suffix = "_60.csv"
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def log_cost(fee_per_side: float) -> float:
    if fee_per_side <= 0:
        return 0.0
    return math.log(1.0 + fee_per_side) - math.log(1.0 - fee_per_side)


def _read_ohlcvt(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        has_header=False,
        new_columns=["ts", "open", "high", "low", "close", "volume", "trades"],
        schema_overrides={
            "ts": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "trades": pl.Float64,
        },
    )
    if frame.height and int(frame["ts"][0]) > 10_000_000_000:
        frame = frame.with_columns((pl.col("ts") // 1000).alias("ts"))
    return frame.filter(
        (pl.col("ts") >= LOAD_START)
        & (pl.col("ts") < 1798761600)
        & (pl.col("close") > 0)
        & (pl.col("open") > 0)
    )


def load_pair(paths: list[Path]) -> pl.DataFrame:
    frames = [_read_ohlcvt(path) for path in paths]
    frame = pl.concat(frames, how="vertical")
    frame = frame.with_columns(((pl.col("ts") // HOUR) * HOUR).alias("ts"))
    # Full-archive and quarterly files can overlap. Keep the later file's bar.
    frame = frame.unique(subset=["ts"], keep="last").sort("ts")
    return frame


def to_hourly_grid(frame: pl.DataFrame) -> pl.DataFrame:
    """Regular hourly index. open_obs is blank on hours with no trades."""
    if frame.is_empty():
        return frame
    start = int(frame["ts"].min())
    end = int(frame["ts"].max())
    grid = pl.DataFrame({"ts": np.arange(start, end + HOUR, HOUR, dtype=np.int64)})
    merged = grid.join(frame, on="ts", how="left").sort("ts")
    merged = merged.with_columns(pl.col("open").alias("open_obs"))
    return merged.with_columns(
        pl.col("close").forward_fill(limit=FILL_LIMIT),
        pl.col("high").forward_fill(limit=FILL_LIMIT),
        pl.col("low").forward_fill(limit=FILL_LIMIT),
        pl.col("volume").fill_null(0.0),
        pl.col("trades").fill_null(0.0),
    )


def factor_returns(frame: pl.DataFrame) -> pl.DataFrame:
    grid = to_hourly_grid(frame)
    close = grid["close"].to_numpy()
    ret = np.full(close.shape[0], np.nan, dtype=np.float64)
    prev = close[:-1]
    nxt = close[1:]
    ok = np.isfinite(prev) & np.isfinite(nxt) & (prev > 0) & (nxt > 0)
    ret[1:] = np.where(ok, np.log(nxt / prev), np.nan)
    return pl.DataFrame({"ts": grid["ts"], "ret": ret})


def discover_hourly_files(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    if not root.exists():
        return grouped
    for path in sorted(root.rglob("*_60.csv")):
        stem = pair_stem(path)
        if stem is None:
            continue
        grouped.setdefault(stem, []).append(path)
    return grouped


def pair_summary(stem: str, paths: list[Path]) -> dict[str, float] | None:
    parsed = parse_pair(stem)
    if parsed is None:
        return None
    base, _quote = parsed
    if not is_eligible_base(base):
        return None
    frame = load_pair(paths)
    history = frame.filter(pl.col("ts") < VAL_START)
    if history.height < MIN_HISTORY_BARS:
        return None
    window_start = VAL_START - 180 * 24 * HOUR
    window = frame.filter((pl.col("ts") >= window_start) & (pl.col("ts") < VAL_START))
    if window.height < MIN_UNIVERSE_HOURS:
        return None
    dollar = float((window["volume"] * window["close"]).sum())
    return {
        "stem": stem,
        "base": base,
        "dollar": dollar,
        "history_bars": float(history.height),
    }


def select_universe(grouped: dict[str, list[Path]]) -> list[tuple[str, list[Path]]]:
    """Top USD names by dollar volume in the 180 days before validation.

    The window ends at the validation start, so the test period does not
    choose who is eligible. BTC and ETH are kept when they pass the same
    history screen.
    """
    summaries = []
    stems = list(grouped)
    for i, stem in enumerate(stems, start=1):
        if i % 100 == 0:
            print(f"  scanned {i}/{len(stems)} pairs", flush=True)
        summary = pair_summary(stem, grouped[stem])
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda row: row["dollar"], reverse=True)
    # One stem per normalized base: the more active listing wins.
    chosen: dict[str, dict[str, float]] = {}
    for row in summaries:
        base = str(row["base"])
        if base not in chosen:
            chosen[base] = row
    ranked = sorted(chosen.values(), key=lambda row: row["dollar"], reverse=True)
    picked = ranked[:UNIVERSE_SIZE]
    have = {str(row["base"]) for row in picked}
    for base in FORCED:
        if base in have or base not in chosen:
            continue
        picked.append(chosen[base])
    picked.sort(key=lambda row: row["dollar"], reverse=True)
    return [(str(row["stem"]), grouped[str(row["stem"])]) for row in picked]


def build_samples_for_asset(
    ts: np.ndarray,
    open_obs: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    trades: np.ndarray,
    btc_ret: np.ndarray,
    eth_ret: np.ndarray,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> dict[str, np.ndarray]:
    """Windows whose last bar is t, labeled by open[t+1] -> open[t+1+horizon]."""
    if len(ts) < lookback + horizon + 1:
        return _empty_samples()
    if len(ts) > 1 and not np.all(np.diff(ts) == HOUR):
        raise ValueError("timestamps must be a regular hourly grid")

    ret = np.full(len(ts), np.nan, dtype=np.float32)
    prev = close[:-1]
    nxt = close[1:]
    ok = np.isfinite(prev) & np.isfinite(nxt) & (prev > 0) & (nxt > 0)
    ret[1:] = np.where(ok, np.log(nxt / prev), np.nan).astype(np.float32)
    np.clip(ret, -0.5, 0.5, out=ret)

    span = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        range_ = np.where((close > 0) & np.isfinite(span), span / close, np.nan)
    range_ = np.clip(range_, 0.0, 0.5).astype(np.float32)
    log_vol = np.log1p(np.clip(volume, 0, None)).astype(np.float32)
    log_trades = np.log1p(np.clip(trades, 0, None)).astype(np.float32)
    features = np.column_stack(
        [
            ret,
            range_,
            log_vol,
            log_trades,
            btc_ret.astype(np.float32),
            eth_ret.astype(np.float32),
        ]
    ).astype(np.float32)

    # Exit open is index t + 1 + horizon, which must land inside the series.
    last_t = len(ts) - horizon - 2
    first_t = lookback - 1
    if last_t < first_t:
        return _empty_samples()
    indices = np.arange(first_t, last_t + 1)
    # Entry is the next hour. Exit is `horizon` hours after entry.
    entry = open_obs[indices + 1]
    exit_ = open_obs[indices + 1 + horizon]
    y = np.full(indices.shape[0], np.nan, dtype=np.float32)
    tradable = np.isfinite(entry) & np.isfinite(exit_) & (entry > 0) & (exit_ > 0)
    y[tradable] = np.log(exit_[tradable] / entry[tradable]).astype(np.float32)

    view = np.lib.stride_tricks.sliding_window_view(features, lookback, axis=0)
    starts = indices - lookback + 1
    windows = np.moveaxis(view[starts], -1, 1)

    finite_x = np.isfinite(windows).all(axis=(1, 2))
    keep = finite_x & np.isfinite(y)
    kept_t = indices[keep]
    hour = ((ts[kept_t] // HOUR) % 24).astype(np.float32)
    # 1970-01-01 was Thursday. Monday = 0.
    dow = ((ts[kept_t] // 86400 + 3) % 7).astype(np.float32)
    static = np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
    ).astype(np.float32)
    return {
        "ts": ts[kept_t].astype(np.int64),
        "x": windows[keep],
        "static": static,
        "y": y[keep].astype(np.float32),
    }


def _empty_samples() -> dict[str, np.ndarray]:
    return {
        "ts": np.empty(0, dtype=np.int64),
        "x": np.empty((0, LOOKBACK, N_FEATURES), dtype=np.float32),
        "static": np.empty((0, 4), dtype=np.float32),
        "y": np.empty(0, dtype=np.float32),
    }


def align_factor(grid_ts: np.ndarray, factor: pl.DataFrame) -> np.ndarray:
    lookup = dict(zip(factor["ts"].to_list(), factor["ret"].to_list(), strict=True))
    out = np.array([lookup.get(int(t), np.nan) for t in grid_ts], dtype=np.float32)
    # A three-hour hole in the factor should not drop every other name.
    last = np.nan
    age = FILL_LIMIT + 1
    for i, value in enumerate(out):
        if np.isfinite(value):
            last = float(value)
            age = 0
        else:
            age += 1
            if age <= FILL_LIMIT and np.isfinite(last):
                out[i] = last
    return out


def build_dataset(selection: list[tuple[str, list[Path]]]) -> dict[str, np.ndarray]:
    loaded: dict[str, pl.DataFrame] = {}
    for stem, paths in selection:
        base, _quote = parse_pair(stem)  # type: ignore[misc]
        loaded[base] = load_pair(paths)
        print(f"  loaded {base} ({stem}) rows={loaded[base].height}", flush=True)
    if "BTC" not in loaded or "ETH" not in loaded:
        raise RuntimeError(f"BTC and ETH are required factors, have {sorted(loaded)}")

    btc_ret = factor_returns(loaded["BTC"])
    eth_ret = factor_returns(loaded["ETH"])
    names = list(loaded)
    chunks_ts = []
    chunks_asset = []
    chunks_x = []
    chunks_static = []
    chunks_y = []
    holds: list[dict[str, float]] = []
    for asset_id, base in enumerate(names):
        grid = to_hourly_grid(loaded[base])
        ts = grid["ts"].to_numpy()
        samples = build_samples_for_asset(
            ts,
            grid["open_obs"].to_numpy(),
            grid["close"].to_numpy(),
            grid["high"].to_numpy(),
            grid["low"].to_numpy(),
            grid["volume"].to_numpy(),
            grid["trades"].to_numpy(),
            align_factor(ts, btc_ret),
            align_factor(ts, eth_ret),
        )
        n = samples["ts"].shape[0]
        if n == 0:
            continue
        chunks_ts.append(samples["ts"])
        chunks_asset.append(np.full(n, asset_id, dtype=np.int16))
        chunks_x.append(samples["x"])
        chunks_static.append(samples["static"])
        chunks_y.append(samples["y"])
        test = grid.filter(
            (pl.col("ts") >= TEST_START) & pl.col("open_obs").is_not_null() & (pl.col("open_obs") > 0)
        )
        if test.height >= 2:
            holds.append(
                {
                    "asset": asset_id,
                    "first_open": float(test["open_obs"][0]),
                    "last_open": float(test["open_obs"][-1]),
                }
            )
        print(f"  samples {base}: {n}", flush=True)

    return {
        "ts": np.concatenate(chunks_ts),
        "asset": np.concatenate(chunks_asset),
        "x": np.concatenate(chunks_x),
        "static": np.concatenate(chunks_static),
        "y": np.concatenate(chunks_y),
        "asset_names": np.array(names),
        "hold_asset": np.array([row["asset"] for row in holds], dtype=np.int16),
        "hold_first": np.array([row["first_open"] for row in holds], dtype=np.float64),
        "hold_last": np.array([row["last_open"] for row in holds], dtype=np.float64),
    }


def split_masks(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = (ts >= TRAIN_START) & (ts < VAL_START)
    val = (ts >= VAL_START) & (ts < TEST_START)
    test = ts >= TEST_START
    return train, val, test

"""Point-in-time attention from completed days.

A decision during UTC day D sees day D-1 and earlier. Same-day totals are not
used. Wikipedia pageviews are human article views. GDELT counts are news
articles. Neither series is a count of new authors, and neither can see
repeated promotional wording. The transform is a surprise against each
series' own trailing days, so a coin that is always popular is not a signal.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DAY = 86400
BASELINE_DAYS = 14
BURST_LAG = 7
MIN_BASELINE = 7
FEATURE_NAMES = (
    "wiki_surprise",
    "wiki_burst",
    "wiki_observed",
    "news_surprise",
    "news_burst",
    "news_observed",
)

# First title that returns a pageview history is kept.
WIKI_TITLES: dict[str, tuple[str, ...]] = {
    "BTC": ("Bitcoin",),
    "ETH": ("Ethereum",),
    "XRP": ("XRP (cryptocurrency)", "Ripple (payment protocol)"),
    "SOL": ("Solana (blockchain platform)",),
    "DOGE": ("Dogecoin",),
    "SUI": ("Sui (blockchain)",),
    "ADA": ("Cardano (blockchain platform)",),
    "PEPE": ("Pepe (cryptocurrency)", "Pepe (meme)"),
    "LTC": ("Litecoin",),
    "TAO": ("Bittensor",),
    "XCN": ("Onyxcoin", "Chain (cryptocurrency)"),
    "LINK": ("Chainlink (blockchain)",),
    "XLM": ("Stellar (payment network)",),
    "ONDO": ("Ondo Finance",),
    "WIF": ("dogwifhat",),
    "AAVE": ("Aave",),
    "CRV": ("Curve DAO (cryptocurrency)", "Curve (cryptocurrency)", "Curve Finance"),
    "AVAX": ("Avalanche (blockchain platform)",),
    "ALGO": ("Algorand",),
    "XMR": ("Monero",),
    "DOT": ("Polkadot (blockchain platform)",),
    "ENA": ("Ethena Labs", "Ethena (cryptocurrency)"),
    "BONK": ("Bonk (cryptocurrency)",),
    "TRX": ("Tron (cryptocurrency)", "TRON"),
}

NEWS_QUERIES: dict[str, str] = {
    "BTC": '"bitcoin"',
    "ETH": '"ethereum"',
    "XRP": '("XRP" OR "ripple")',
    "SOL": '"solana" (crypto OR blockchain OR token)',
    "DOGE": '"dogecoin"',
    "SUI": '("sui blockchain" OR "sui network")',
    "ADA": '"cardano"',
    "PEPE": '"pepe" (crypto OR memecoin OR token)',
    "LTC": '"litecoin"',
    "TAO": '"bittensor"',
    "XCN": '"onyxcoin"',
    "LINK": '"chainlink"',
    "XLM": '"stellar" (lumens OR crypto OR blockchain)',
    "ONDO": '("ondo finance" OR "ondo token")',
    "WIF": '"dogwifhat"',
    "AAVE": '"aave"',
    "CRV": '("curve finance" OR "curve dao")',
    "AVAX": '"avalanche" (crypto OR blockchain OR token)',
    "ALGO": '"algorand"',
    "XMR": '"monero"',
    "DOT": '"polkadot"',
    "ENA": '"ethena" (crypto OR stablecoin OR token)',
    "BONK": '"bonk" (crypto OR memecoin OR token)',
    "TRX": '"tron" (crypto OR blockchain OR token)',
}

USER_AGENT = "crypto-go/0.1 (research; github.com/maxmbay/crypto-go)"


def daily_surprise(
    values: np.ndarray,
    baseline_days: int = BASELINE_DAYS,
    burst_lag: int = BURST_LAG,
    min_baseline: int = MIN_BASELINE,
) -> tuple[np.ndarray, np.ndarray]:
    """Surprise and burst for a daily count series. Index i uses only days before i, plus i."""
    logv = np.log1p(np.asarray(values, dtype=np.float64))
    surprise = np.full(logv.shape[0], np.nan, dtype=np.float64)
    burst = np.full(logv.shape[0], np.nan, dtype=np.float64)
    for i, current in enumerate(logv):
        if not np.isfinite(current):
            continue
        window = logv[max(0, i - baseline_days) : i]
        finite = window[np.isfinite(window)]
        if finite.size < min_baseline:
            continue
        surprise[i] = current - float(finite.mean())
        earlier = i - burst_lag
        if earlier >= 0 and np.isfinite(logv[earlier]):
            burst[i] = current - float(logv[earlier])
    return surprise, burst


def align_completed_day(
    day_starts: np.ndarray,
    daily_values: np.ndarray,
    decision_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each decision to the previous completed UTC day's feature. Missing days are unobserved."""
    day_starts = np.asarray(day_starts, dtype=np.int64)
    daily_values = np.asarray(daily_values, dtype=np.float64)
    order = np.argsort(day_starts, kind="mergesort")
    day_starts = day_starts[order]
    daily_values = daily_values[order]
    if day_starts.size and np.any(np.diff(day_starts) == 0):
        keep = np.ones(day_starts.shape[0], dtype=bool)
        keep[:-1] = day_starts[1:] != day_starts[:-1]
        day_starts = day_starts[keep]
        daily_values = daily_values[keep]
    yesterday = (np.asarray(decision_ts, dtype=np.int64) // DAY) * DAY - DAY
    index = np.searchsorted(day_starts, yesterday)
    in_range = index < day_starts.shape[0]
    matched = np.zeros(yesterday.shape[0], dtype=bool)
    matched[in_range] = day_starts[index[in_range]] == yesterday[in_range]
    values = np.zeros(yesterday.shape[0], dtype=np.float32)
    observed = np.zeros(yesterday.shape[0], dtype=np.float32)
    if np.any(matched):
        taken = daily_values[index[matched]]
        finite = np.isfinite(taken)
        rows = np.flatnonzero(matched)[finite]
        values[rows] = taken[finite].astype(np.float32)
        observed[rows] = 1.0
    return values, observed


def on_calendar(day_starts: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put counts on every UTC midnight from the first day through the last. Gaps stay missing."""
    day_starts = np.asarray(day_starts, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if day_starts.size == 0:
        return day_starts, values
    full = np.arange(int(day_starts.min()), int(day_starts.max()) + DAY, DAY, dtype=np.int64)
    out = np.full(full.shape[0], np.nan, dtype=np.float64)
    index = np.searchsorted(full, day_starts)
    in_range = index < full.shape[0]
    matched = np.zeros(day_starts.shape[0], dtype=bool)
    matched[in_range] = full[index[in_range]] == day_starts[in_range]
    out[index[matched]] = values[matched]
    return full, out


def attention_block(
    decision_ts: np.ndarray,
    wiki_days: np.ndarray | None,
    wiki_counts: np.ndarray | None,
    news_days: np.ndarray | None,
    news_counts: np.ndarray | None,
) -> np.ndarray:
    """Six columns: wiki surprise, wiki burst, wiki observed, then the same for news."""
    columns = []
    for days, counts in ((wiki_days, wiki_counts), (news_days, news_counts)):
        if days is None or counts is None or len(days) == 0:
            columns.extend(
                [
                    np.zeros(decision_ts.shape[0], dtype=np.float32),
                    np.zeros(decision_ts.shape[0], dtype=np.float32),
                    np.zeros(decision_ts.shape[0], dtype=np.float32),
                ]
            )
            continue
        days, counts = on_calendar(days, counts)
        surprise, burst = daily_surprise(counts)
        surprise_at, observed = align_completed_day(days, surprise, decision_ts)
        burst_at, _burst_observed = align_completed_day(days, burst, decision_ts)
        columns.extend([surprise_at, burst_at, observed])
    return np.column_stack(columns).astype(np.float32)


def attach_attention(data: dict[str, np.ndarray], features: np.ndarray) -> dict[str, np.ndarray]:
    """Append attention columns to the in-memory static block. The market cache is left untouched."""
    if features.shape[0] != data["static"].shape[0]:
        raise ValueError(f"attention rows {features.shape[0]} != panel rows {data['static'].shape[0]}")
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} attention columns, got {features.shape[1]}")
    out = dict(data)
    out["static"] = np.concatenate([data["static"], features], axis=1).astype(np.float32)
    return out


def _request_json(url: str, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    delay = 8.0
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            if body.lstrip().startswith("Please limit"):
                raise urllib.error.HTTPError(url, 429, body[:120], hdrs=None, fp=None)
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"request failed: {url}") from last_error


def fetch_wiki_daily(titles: tuple[str, ...], start: str = "20200101", end: str = "20260930") -> tuple[str, np.ndarray, np.ndarray]:
    """Return the title that resolved, plus midnight timestamps and view counts."""
    last_error: Exception | None = None
    for title in titles:
        encoded = urllib.parse.quote(title, safe="")
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/user/{encoded}/daily/{start}/{end}"
        )
        try:
            payload = _request_json(url)
        except urllib.error.HTTPError as exc:
            last_error = exc
            continue
        items = payload.get("items") or []
        if not items:
            continue
        days = []
        views = []
        for item in items:
            stamp = str(item["timestamp"])
            year, month, day = int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8])
            midnight = int(
                np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "s").astype(np.int64)
            )
            days.append(midnight)
            views.append(float(item["views"]))
        return title, np.asarray(days, dtype=np.int64), np.asarray(views, dtype=np.float64)
    raise RuntimeError(f"no Wikipedia pageviews for {titles}") from last_error


def fetch_gdelt_daily(query: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """Article counts between YYYYMMDD bounds. GDELT asks for one request every five seconds."""
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        + urllib.parse.urlencode(
            {
                "query": query,
                "mode": "timelinevolraw",
                "format": "json",
                "startdatetime": f"{start}000000",
                "enddatetime": f"{end}000000",
            }
        )
    )
    payload = _request_json(url, timeout=90.0)
    timeline = payload.get("timeline") or []
    if not timeline:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    days = []
    counts = []
    for point in timeline[0].get("data") or []:
        stamp = str(point["date"])
        year, month, day = int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8])
        midnight = int(np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "s").astype(np.int64))
        days.append(midnight)
        counts.append(float(point["value"]))
    return np.asarray(days, dtype=np.int64), np.asarray(counts, dtype=np.float64)


def _chunks(start: str = "20200101", end: str = "20261001", span_days: int = 90) -> list[tuple[str, str]]:
    cursor = np.datetime64(f"{start[0:4]}-{start[4:6]}-{start[6:8]}")
    stop = np.datetime64(f"{end[0:4]}-{end[4:6]}-{end[6:8]}")
    out = []
    while cursor < stop:
        nxt = min(cursor + np.timedelta64(span_days, "D"), stop)
        out.append((str(cursor).replace("-", ""), str(nxt).replace("-", "")))
        cursor = nxt
    return out


def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def download_histories(root: Path, pause: float = 6.0) -> dict[str, dict]:
    """Cache Wikipedia and GDELT daily series under root. Existing files are kept."""
    root.mkdir(parents=True, exist_ok=True)
    wiki_dir = root / "wiki"
    news_dir = root / "news"
    wiki_dir.mkdir(exist_ok=True)
    news_dir.mkdir(exist_ok=True)
    manifest: dict[str, dict] = {}
    for asset, titles in WIKI_TITLES.items():
        path = wiki_dir / f"{asset}.npz"
        if path.exists():
            loaded = np.load(path)
            manifest[asset] = {"wiki_title": str(loaded["title"]), "wiki_days": int(loaded["days"].shape[0])}
            print(f"wiki {asset} cached {loaded['title']}", flush=True)
            continue
        try:
            title, days, counts = fetch_wiki_daily(titles)
        except Exception as exc:
            print(f"wiki {asset} missing ({exc})", flush=True)
            manifest[asset] = {"wiki_title": None, "wiki_days": 0}
            continue
        np.savez(path, title=np.array(title), days=days, counts=counts)
        manifest[asset] = {"wiki_title": title, "wiki_days": int(days.shape[0])}
        print(f"wiki {asset} {title} days={days.shape[0]}", flush=True)
        time.sleep(0.2)

    next_allowed = time.monotonic()
    for asset, query in NEWS_QUERIES.items():
        path = news_dir / f"{asset}.npz"
        if path.exists():
            loaded = np.load(path)
            manifest.setdefault(asset, {})["news_days"] = int(loaded["days"].shape[0])
            print(f"news {asset} cached days={loaded['days'].shape[0]}", flush=True)
            continue
        _sleep_until(next_allowed)
        try:
            days, counts = fetch_gdelt_daily(query, "20200101", "20261001")
        except Exception as exc:
            print(f"news {asset} failed ({exc})", flush=True)
            next_allowed = time.monotonic() + pause
            manifest.setdefault(asset, {})["news_days"] = 0
            continue
        next_allowed = time.monotonic() + pause
        span_days = ((int(days.max()) - int(days.min())) / DAY + 1) if days.size else 0
        density = days.shape[0] / span_days if span_days else 0.0
        if days.shape[0] < 30 or (span_days > 500 and density < 0.5):
            print(
                f"news {asset} rejected points={days.shape[0]} span={span_days:.0f} density={density:.2f}",
                flush=True,
            )
            manifest.setdefault(asset, {})["news_days"] = 0
            continue
        chunk_days = [days]
        chunk_counts = [counts]
        print(f"news {asset} points={days.shape[0]}", flush=True)
        if chunk_days:
            days = np.concatenate(chunk_days)
            counts = np.concatenate(chunk_counts)
            order = np.argsort(days, kind="mergesort")
            days = days[order]
            counts = counts[order]
            if days.size:
                keep = np.ones(days.shape[0], dtype=bool)
                keep[:-1] = np.diff(days) != 0
                days = days[keep]
                counts = counts[keep]
            np.savez(path, days=days, counts=counts, query=np.array(query))
            manifest.setdefault(asset, {})["news_days"] = int(days.shape[0])
        else:
            manifest.setdefault(asset, {})["news_days"] = 0
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _usable_wiki(days: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drop pages that are effectively unread. A title can resolve and still not be the asset."""
    if days.shape[0] < 60:
        return None, None
    finite = counts[np.isfinite(counts)]
    if finite.size < 60 or float(np.median(finite)) < 15.0:
        return None, None
    return days, counts


def build_feature_matrix(data: dict[str, np.ndarray], root: Path) -> np.ndarray:
    """Row-align cached daily series to the market panel. One block per asset id."""
    names = [str(name) for name in data["asset_names"].tolist()]
    features = np.zeros((data["ts"].shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    asset = data["asset"]
    for asset_id, name in enumerate(names):
        rows = asset == asset_id
        if not np.any(rows):
            continue
        wiki_path = root / "wiki" / f"{name}.npz"
        news_path = root / "news" / f"{name}.npz"
        wiki_days = wiki_counts = news_days = news_counts = None
        if wiki_path.exists():
            loaded = np.load(wiki_path)
            wiki_days, wiki_counts = _usable_wiki(loaded["days"], loaded["counts"])
        # A partial news cache is not a feature. Coins that happened to download
        # would look informed, and the rest would look quiet.
        news_ready = sum(1 for asset_name in names if (root / "news" / f"{asset_name}.npz").exists())
        if news_ready == len(names) and news_path.exists():
            loaded = np.load(news_path)
            news_days, news_counts = loaded["days"], loaded["counts"]
        features[rows] = attention_block(data["ts"][rows], wiki_days, wiki_counts, news_days, news_counts)
        observed = features[rows, 2].mean() if wiki_days is not None else 0.0
        news_observed = features[rows, 5].mean() if news_days is not None else 0.0
        print(f"aligned {name} wiki={observed:.2f} news={news_observed:.2f}", flush=True)
    return features


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "attention"
    download_histories(root)


if __name__ == "__main__":
    main()

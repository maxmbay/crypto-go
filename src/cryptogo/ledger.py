"""Paper ledger for a 24-hour hold with a fixed number of slots.

Equity moves when a position is opened (cash becomes principal) and when it
is settled at the realized open-to-open return. Mark-to-market inside the
hold is not available from the decision rows, so drawdown is exit-to-exit.
"""

from __future__ import annotations

import math

import numpy as np

from cryptogo.panel import HOUR, log_cost

HOLD_BARS = 24


def simulate(
    ts: np.ndarray,
    asset: np.ndarray,
    pred: np.ndarray,
    gross_y: np.ndarray,
    fee_per_side: float,
    max_positions: int = 5,
    size: np.ndarray | None = None,
) -> dict[str, float]:
    n = len(ts)
    if n == 0:
        return _result(1.0, 0, 0.0, 0.0)
    order = np.argsort(ts, kind="mergesort")
    ts_sorted = ts[order]
    breaks = np.flatnonzero(np.diff(ts_sorted)) + 1
    groups = np.split(order, breaks)
    fee_log = log_cost(fee_per_side)
    cash = 1.0
    # asset -> (principal, net log return, free at this decision ts)
    open_pos: dict[int, tuple[float, float, int]] = {}
    equity_curve = []
    trades = 0
    net_logs = []
    size_sum = 0.0

    for group in groups:
        now = int(ts[group[0]])
        settled = [asset_id for asset_id, pos in open_pos.items() if now >= pos[2]]
        for asset_id in settled:
            principal, net_y, _free = open_pos.pop(asset_id)
            cash += principal * math.exp(net_y)
            net_logs.append(net_y)
        slots = max_positions - len(open_pos)
        if slots > 0 and cash > 0:
            ranked = sorted(group.tolist(), key=lambda i: float(pred[i]), reverse=True)
            locked = cash + sum(pos[0] for pos in open_pos.values())
            slot_cash = locked / max_positions
            taken = 0
            for i in ranked:
                if taken >= slots:
                    break
                score = float(pred[i])
                if not math.isfinite(score) or score <= fee_log:
                    break
                asset_id = int(asset[i])
                if asset_id in open_pos:
                    continue
                weight = 1.0 if size is None else float(size[i])
                if not math.isfinite(weight) or weight <= 0.0:
                    continue
                weight = min(weight, 1.0)
                invest = min(slot_cash * weight, cash)
                if invest <= 1e-8:
                    break
                outcome = float(gross_y[i])
                if not math.isfinite(outcome):
                    continue
                cash -= invest
                open_pos[asset_id] = (invest, outcome - fee_log, now + HOLD_BARS * HOUR)
                trades += 1
                size_sum += weight
                taken += 1
        equity_curve.append(cash + sum(pos[0] for pos in open_pos.values()))

    for principal, net_y, _free in open_pos.values():
        cash += principal * math.exp(float(net_y))
        net_logs.append(net_y)
    equity_curve.append(cash)
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    mean_net = float(np.mean(net_logs)) if net_logs else 0.0
    result = _result(cash, trades, max_dd, mean_net)
    result["mean_size"] = size_sum / trades if trades else 0.0
    return result


def _result(wealth: float, trades: int, max_drawdown: float, mean_net: float) -> dict[str, float]:
    return {
        "wealth": wealth,
        "return": wealth - 1.0,
        "trades": float(trades),
        "max_drawdown": max_drawdown,
        "mean_net_log": mean_net,
    }


def cross_sectional_ic(
    ts: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    min_names: int = 8,
) -> float:
    """Mean hourly Spearman correlation between the forecast and the outcome."""
    if len(ts) == 0:
        return float("nan")
    order = np.argsort(ts, kind="mergesort")
    ts_sorted = ts[order]
    breaks = np.flatnonzero(np.diff(ts_sorted)) + 1
    groups = np.split(order, breaks)
    scores = []
    for group in groups:
        if group.size < min_names:
            continue
        left = pred[group]
        right = y[group]
        if np.nanstd(left) < 1e-12 or np.nanstd(right) < 1e-12:
            continue
        scores.append(_spearman(left, right))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def sign_hit(pred: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(y) & (pred != 0) & (y != 0)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(np.sign(pred[ok]) == np.sign(y[ok])))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks

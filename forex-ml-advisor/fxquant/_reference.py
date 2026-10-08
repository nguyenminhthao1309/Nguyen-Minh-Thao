"""Pure NumPy reference implementation of the C++ core.

Used automatically when the compiled extension is not available, and by the
test-suite to verify that the C++ code produces identical numbers. The
algorithms mirror cpp/src/*.cpp line by line; keep the two in sync.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _windows

_NAN = np.nan


def _as_array(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def ema(x, span: int) -> np.ndarray:
    _check_positive(span, "span")
    x = _as_array(x)
    out = np.full(x.size, _NAN)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    s = finite[0]
    prev = out[s] = x[s]
    for i in range(s + 1, x.size):
        if np.isfinite(x[i]):
            prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def sma(x, window: int) -> np.ndarray:
    _check_positive(window, "window")
    x = _as_array(x)
    out = np.full(x.size, _NAN)
    if x.size >= window:
        out[window - 1:] = _windows(x, window).mean(axis=1)
    return out


def rolling_std(x, window: int, ddof: int = 1) -> np.ndarray:
    _check_positive(window, "window")
    if window - ddof <= 0:
        raise ValueError("window must exceed ddof")
    x = _as_array(x)
    out = np.full(x.size, _NAN)
    if x.size >= window:
        out[window - 1:] = _windows(x, window).std(axis=1, ddof=ddof)
    return out


def rolling_zscore(x, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be >= 2")
    x = _as_array(x)
    out = np.full(x.size, _NAN)
    if x.size >= window:
        w = _windows(x, window)
        sd = w.std(axis=1, ddof=1)
        cur = x[window - 1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sd > 0, (cur - w.mean(axis=1)) / sd, 0.0)
        z[~np.isfinite(sd)] = _NAN
        out[window - 1:] = z
    return out


def rolling_percentile_rank(x, window: int) -> np.ndarray:
    _check_positive(window, "window")
    x = _as_array(x)
    out = np.full(x.size, _NAN)
    if x.size >= window:
        w = _windows(x, window)
        cur = x[window - 1:]
        valid = np.isfinite(w).sum(axis=1)
        below = (w <= cur[:, None]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[window - 1:] = np.where(np.isfinite(cur), below / valid, _NAN)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi(close, period: int) -> np.ndarray:
    _check_positive(period, "period")
    c = _as_array(close)
    n = c.size
    out = np.full(n, _NAN)
    if n <= period:
        return out
    d = np.diff(c)
    gains, losses = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    avg_gain = gains[:period].sum() / period
    avg_loss = losses[:period].sum() / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    if not (h.size == l.size == c.size):
        raise ValueError("high, low and close must have the same length")
    tr = np.full(c.size, _NAN)
    if c.size == 0:
        return tr
    tr[0] = h[0] - l[0]
    prev = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - prev), np.abs(l[1:] - prev)])
    return tr


def atr(high, low, close, period: int) -> np.ndarray:
    _check_positive(period, "period")
    tr = true_range(high, low, close)
    n = tr.size
    out = np.full(n, _NAN)
    if n < period:
        return out
    prev = out[period - 1] = tr[:period].mean()
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def adx(high, low, close, period: int) -> np.ndarray:
    _check_positive(period, "period")
    h, l = _as_array(high), _as_array(low)
    tr = true_range(high, low, close)
    n = tr.size
    out = np.full(n, _NAN)
    if n < 2 * period:
        return out
    up = np.zeros(n)
    down = np.zeros(n)
    up[1:] = h[1:] - h[:-1]
    down[1:] = l[:-1] - l[1:]
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)

    def dx(s_tr, s_pdm, s_mdm):
        if s_tr <= 0.0:
            return 0.0
        pdi, mdi = 100.0 * s_pdm / s_tr, 100.0 * s_mdm / s_tr
        denom = pdi + mdi
        return 0.0 if denom == 0.0 else 100.0 * abs(pdi - mdi) / denom

    s_tr = tr[1:period + 1].sum()
    s_pdm = pdm[1:period + 1].sum()
    s_mdm = mdm[1:period + 1].sum()
    dxv = np.full(n, _NAN)
    dxv[period] = dx(s_tr, s_pdm, s_mdm)
    for i in range(period + 1, n):
        s_tr = s_tr - s_tr / period + tr[i]
        s_pdm = s_pdm - s_pdm / period + pdm[i]
        s_mdm = s_mdm - s_mdm / period + mdm[i]
        dxv[i] = dx(s_tr, s_pdm, s_mdm)

    prev = out[2 * period - 1] = dxv[period:2 * period].sum() / period
    for i in range(2 * period, n):
        prev = (prev * (period - 1) + dxv[i]) / period
        out[i] = prev
    return out


def stochastic_k(high, low, close, period: int) -> np.ndarray:
    _check_positive(period, "period")
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    out = np.full(c.size, _NAN)
    if c.size >= period:
        hh = _windows(h, period).max(axis=1)
        ll = _windows(l, period).min(axis=1)
        rng = hh - ll
        with np.errstate(invalid="ignore", divide="ignore"):
            out[period - 1:] = np.where(rng > 0, 100.0 * (c[period - 1:] - ll) / rng, 50.0)
    return out


def efficiency_ratio(close, period: int) -> np.ndarray:
    _check_positive(period, "period")
    c = _as_array(close)
    out = np.full(c.size, _NAN)
    if c.size > period:
        path = _windows(np.abs(np.diff(c)), period).sum(axis=1)
        net = np.abs(c[period:] - c[:-period])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[period:] = np.where(path > 0, net / path, 0.0)
    return out


def variance_ratio(returns, window: int, q: int) -> np.ndarray:
    _check_positive(q, "q")
    if window < q + 2:
        raise ValueError("window must be >= q + 2")
    r = _as_array(returns)
    n = r.size
    out = np.full(n, _NAN)
    if n < window:
        return out
    qsum = np.full(n, _NAN)
    qsum[q - 1:] = _windows(r, q).sum(axis=1)
    m = window - q + 1
    var1 = _windows(r, window).std(axis=1, ddof=1) ** 2
    varq = _windows(qsum[q - 1:], m).std(axis=1, ddof=1) ** 2
    # Both arrays have n - window + 1 rows; row k belongs to output index k + window - 1.
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = np.where(var1 > 0, varq / (q * var1), _NAN)
    out[window - 1:] = vr
    return out


def bollinger_pctb(close, window: int, nstd: float = 2.0) -> np.ndarray:
    _check_positive(window, "window")
    c = _as_array(close)
    out = np.full(c.size, _NAN)
    if c.size >= window:
        w = _windows(c, window)
        sd = w.std(axis=1, ddof=0)
        lower = w.mean(axis=1) - nstd * sd
        with np.errstate(invalid="ignore", divide="ignore"):
            out[window - 1:] = np.where(sd == 0, 0.5, (c[window - 1:] - lower) / (2.0 * nstd * sd))
    return out


def triple_barrier(high, low, close, atr, horizon: int, tp_mult: float, sl_mult: float) -> dict:
    h, l, c, a = (_as_array(v) for v in (high, low, close, atr))
    n = c.size
    if not (h.size == l.size == a.size == n):
        raise ValueError("all inputs must have the same length")
    if horizon <= 0 or tp_mult <= 0 or sl_mult <= 0:
        raise ValueError("horizon and barrier multiples must be positive")
    label = np.full(n, _NAN)
    ret = np.full(n, _NAN)
    hold = np.full(n, _NAN)
    end_idx = np.full(n, _NAN)
    for i in range(n - horizon):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        c0 = c[i]
        upper, lower = c0 + tp_mult * a[i], c0 - sl_mult * a[i]
        lab, exit_px, end = 0.0, c[i + horizon], i + horizon
        hu = h[i + 1:i + horizon + 1] >= upper
        hd = l[i + 1:i + horizon + 1] <= lower
        hits = np.flatnonzero(hu | hd)
        if hits.size:
            k = hits[0]
            end = i + 1 + k
            if hu[k] and hd[k]:
                exit_px = c[end]
            elif hu[k]:
                lab, exit_px = 1.0, upper
            else:
                lab, exit_px = -1.0, lower
        label[i] = lab
        ret[i] = np.log(exit_px / c0)
        hold[i] = end - i
        end_idx[i] = end
    return {"label": label, "ret": ret, "hold": hold, "end_idx": end_idx}


def _sign(x: float) -> int:
    if not np.isfinite(x):
        return 0
    return int(x > 0) - int(x < 0)


def backtest(open, high, low, close, atr, signal, sl_mult=1.5, tp_mult=1.5,
             max_hold=24, risk_fraction=0.005, cost=0.0) -> dict:
    o, h, l, c, a, s = (_as_array(v) for v in (open, high, low, close, atr, signal))
    n = c.size
    if not all(v.size == n for v in (o, h, l, a, s)):
        raise ValueError("all inputs must have the same length")
    if sl_mult <= 0 or tp_mult <= 0 or max_hold <= 0 or risk_fraction <= 0 or cost < 0:
        raise ValueError("invalid backtest parameters")

    equity = np.ones(n)
    position = np.zeros(n)
    trades = {k: [] for k in ("entry_idx", "exit_idx", "direction", "entry_px", "exit_px", "r_multiple", "reason")}
    half = 0.5 * cost
    eq_closed = eq_entry = 1.0
    d = 0
    entry_idx = 0
    entry_eff = risk = stop = target = 0.0

    for t in range(n):
        if d == 0 and t > 0:
            sig = _sign(s[t - 1])
            if sig != 0 and np.isfinite(a[t - 1]) and a[t - 1] > 0:
                d, entry_idx, eq_entry = sig, t, eq_closed
                risk = sl_mult * a[t - 1]
                stop = o[t] - d * risk
                target = o[t] + d * tp_mult * a[t - 1]
                entry_eff = o[t] + d * half

        if d != 0:
            exit_px, reason = 0.0, 0
            if t > entry_idx:
                if d * (o[t] - stop) <= 0:
                    exit_px, reason = o[t], 1
                elif d * (o[t] - target) >= 0:
                    exit_px, reason = o[t], 2
            if reason == 0:
                hit_stop = l[t] <= stop if d > 0 else h[t] >= stop
                hit_target = h[t] >= target if d > 0 else l[t] <= target
                if hit_stop:
                    exit_px, reason = stop, 1
                elif hit_target:
                    exit_px, reason = target, 2
                elif t - entry_idx + 1 >= max_hold:
                    exit_px, reason = c[t], 3
                elif _sign(s[t]) == -d:
                    exit_px, reason = c[t], 4
                elif t == n - 1:
                    exit_px, reason = c[t], 5
            if reason:
                exit_eff = exit_px - d * half
                r = d * (exit_eff - entry_eff) / risk
                eq_closed = eq_entry * (1.0 + risk_fraction * r)
                for key, val in zip(trades, (entry_idx, t, d, entry_eff, exit_eff, r, reason)):
                    trades[key].append(val)
                d = 0

        if d != 0:
            mark = c[t] - d * half
            equity[t] = eq_entry * (1.0 + risk_fraction * d * (mark - entry_eff) / risk)
        else:
            equity[t] = eq_closed
        position[t] = d

    return {
        "equity": equity,
        "position": position,
        "trades": {k: np.asarray(v, dtype=np.float64) for k, v in trades.items()},
    }

"""Currency strength: decompose every pair's return into single-currency moves.

A pair return is r(BASE/QUOTE) = s(BASE) - s(QUOTE). Stacking all pairs gives
an over-determined linear system per bar; its minimum-norm least-squares
solution (which sums to zero across currencies) is each currency's strength
move for that bar. Only same-bar returns are used, so the result is causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.DataFrame({p: df["close"] for p, df in data.items()}).sort_index()
    return np.log(closes).diff().fillna(0.0)


def currency_strength(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-bar strength returns, one column per currency (log units)."""
    rets = pair_returns(data)
    pairs = list(rets.columns)
    ccys = sorted({p[:3] for p in pairs} | {p[3:] for p in pairs})
    pos = {c: i for i, c in enumerate(ccys)}
    design = np.zeros((len(pairs), len(ccys)))
    for i, p in enumerate(pairs):
        design[i, pos[p[:3]]] = 1.0
        design[i, pos[p[3:]]] = -1.0
    s = rets.to_numpy() @ np.linalg.pinv(design).T
    return pd.DataFrame(s, index=rets.index, columns=ccys)


def strength_ranks(strength: pd.DataFrame, window: int) -> pd.DataFrame:
    """Cross-sectional percentile (0 weakest .. 1 strongest) of trailing strength."""
    trailing = strength.rolling(window, min_periods=window).sum()
    return trailing.rank(axis=1, pct=True)


def strength_table(strength: pd.DataFrame, bars_per_day: int) -> pd.DataFrame:
    """Cumulative strength (%) over 1 day / 1 week / 1 month and its rank."""
    horizons = {"1D": bars_per_day, "1W": 5 * bars_per_day, "1M": 21 * bars_per_day}
    table = pd.DataFrame({k: strength.iloc[-n:].sum() * 100 for k, n in horizons.items()})
    table["score"] = table.rank(pct=True).mean(axis=1)
    table["rank"] = table["score"].rank(ascending=False, method="min").astype(int)
    table["trend"] = np.where(table["1W"] > 0, "strengthening", "weakening")
    table.index.name = "currency"
    return table.sort_values("rank")

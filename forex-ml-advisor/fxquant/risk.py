"""Position sizing and portfolio-level risk limits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AdvisorConfig, PairSpec

LOT = 100_000  # units of base currency in one standard lot


def conversion_rate(ccy: str, account: str, prices: dict[str, float]) -> float:
    """Value of one unit of `ccy` in the account currency, from the latest prices."""
    if ccy == account:
        return 1.0
    if f"{ccy}{account}" in prices:
        return prices[f"{ccy}{account}"]
    if f"{account}{ccy}" in prices:
        return 1.0 / prices[f"{account}{ccy}"]
    if account != "USD" and ccy != "USD":
        via_usd = conversion_rate(ccy, "USD", prices) * conversion_rate("USD", account, prices)
        if np.isfinite(via_usd):
            return via_usd
    return np.nan


def position_size(equity: float, risk_fraction: float, entry: float, stop: float, spec: PairSpec,
                  prices: dict[str, float], account: str = "USD") -> dict:
    """Units/lots such that hitting the stop loses `risk_fraction` of equity."""
    rate = conversion_rate(spec.quote, account, prices)
    risk_amount = equity * risk_fraction
    stop_dist = abs(entry - stop)
    if not np.isfinite(rate) or stop_dist <= 0:
        return {"units": np.nan, "lots": np.nan, "risk_amount": risk_amount, "pip_value_per_lot": np.nan,
                "stop_pips": stop_dist / spec.pip_size}
    units = risk_amount / (stop_dist * rate)
    return {
        "units": float(units),
        "lots": float(units / LOT),
        "risk_amount": float(risk_amount),
        "pip_value_per_lot": float(LOT * spec.pip_size * rate),
        "stop_pips": float(stop_dist / spec.pip_size),
    }


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full-Kelly bet fraction p - (1 - p) / b with payoff ratio b = avg_win / avg_loss."""
    if not all(np.isfinite([win_rate, avg_win, avg_loss])) or avg_win <= 0 or avg_loss <= 0:
        return 0.0
    return float(win_rate - (1 - win_rate) / (avg_win / avg_loss))


def apply_portfolio_limits(ideas: list[dict], corr: pd.DataFrame, cfg: AdvisorConfig) -> tuple[list[dict], list[dict]]:
    """Greedy selection by score under total-risk, per-currency and correlation limits.

    Each idea needs: pair, direction (+1/-1), risk_fraction, score. Ideas that do
    not fit are returned separately with a `blocked_by` explanation.
    """
    accepted, rejected = [], []
    ccy_risk: dict[str, float] = {}
    total = 0.0
    for idea in sorted(ideas, key=lambda x: x["score"], reverse=True):
        pair, d, risk = idea["pair"], idea["direction"], idea["risk_fraction"]
        base, quote = pair[:3], pair[3:]

        clash = None
        for other in accepted:
            c = corr.at[pair, other["pair"]] if pair in corr.index and other["pair"] in corr.columns else 0.0
            if np.isfinite(c) and c * d * other["direction"] > cfg.correlation_limit:
                clash = other["pair"]
                break
        if clash:
            rejected.append({**idea, "blocked_by": f"same exposure as {clash} (corr {corr.at[pair, clash]:+.2f})"})
            continue
        if len(accepted) >= cfg.max_recommendations:
            rejected.append({**idea, "blocked_by": "max number of open ideas reached"})
            continue

        # Shrink the position until currency and total-risk budgets are respected.
        room = cfg.max_total_risk - total
        for ccy, sign in ((base, d), (quote, -d)):
            cur = ccy_risk.get(ccy, 0.0)
            room = min(room, cfg.max_currency_risk - sign * cur)
        size = min(risk, room)
        if size < 0.25 * risk:
            rejected.append({**idea, "blocked_by": "currency / total risk budget exhausted"})
            continue
        if size < risk:
            idea = {**idea, "risk_fraction": size, "note": "size reduced to respect risk budget"}
        ccy_risk[base] = ccy_risk.get(base, 0.0) + d * size
        ccy_risk[quote] = ccy_risk.get(quote, 0.0) - d * size
        total += size
        accepted.append(idea)
    return accepted, rejected


def currency_exposure(ideas: list[dict]) -> pd.Series:
    """Net risk (fraction of equity) per currency implied by a set of ideas."""
    exp: dict[str, float] = {}
    for idea in ideas:
        d, r = idea["direction"], idea["risk_fraction"]
        exp[idea["pair"][:3]] = exp.get(idea["pair"][:3], 0.0) + d * r
        exp[idea["pair"][3:]] = exp.get(idea["pair"][3:], 0.0) - d * r
    return pd.Series(exp).sort_values()

"""Out-of-sample strategy backtests (execution runs in the C++ core)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core
from .config import ModelConfig, PairSpec, StrategyConfig
from .features import risk_unit

EXIT_REASONS = {1: "stop", 2: "target", 3: "time", 4: "reversal", 5: "end"}


def signals_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    sig = np.where(p > 0.5 + threshold, 1.0, np.where(p < 0.5 - threshold, -1.0, 0.0))
    return np.where(np.isnan(p), 0.0, sig)


def round_trip_cost(spec: PairSpec, slippage_pips: float) -> float:
    """Spread plus slippage on both legs, in price units."""
    return (spec.spread_pips + 2 * slippage_pips) * spec.pip_size


def _daily(series: pd.Series) -> pd.Series:
    return series.resample("24h", origin="epoch", offset="22h").last().dropna()


def performance_metrics(equity: pd.Series, r_multiples: np.ndarray, position: np.ndarray,
                        holds: np.ndarray | None = None) -> dict:
    daily = _daily(equity)
    rets = daily.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    n = len(r_multiples)
    wins, losses = r_multiples[r_multiples > 0], r_multiples[r_multiples < 0]
    sd_r = r_multiples.std(ddof=1) if n > 1 else np.nan
    total = float(equity.iloc[-1] - 1.0)
    return {
        "trades": n,
        "win_rate": float((r_multiples > 0).mean()) if n else np.nan,
        "avg_r": float(r_multiples.mean()) if n else np.nan,
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(-losses.mean()) if len(losses) else np.nan,
        "t_stat": float(r_multiples.mean() / (sd_r / np.sqrt(n))) if n > 1 and sd_r > 0 else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(260)) if len(rets) > 5 and rets.std() > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "exposure": float((position != 0).mean()),
        "avg_hold_bars": float(np.mean(holds)) if holds is not None and len(holds) else np.nan,
    }


def backtest_pair(ohlc: pd.DataFrame, atr: pd.Series, proba: pd.Series, spec: PairSpec,
                  mcfg: ModelConfig, scfg: StrategyConfig) -> dict:
    """Trade the model's out-of-sample probabilities on one pair."""
    window = proba.dropna().index
    if len(window) < 50:
        return {"metrics": {"trades": 0}, "equity": pd.Series(dtype=float), "trades": pd.DataFrame()}
    bars = ohlc.loc[window[0]:window[-1]]
    unit = risk_unit(atr.reindex(bars.index).to_numpy(), mcfg.horizon)
    signal = signals_from_proba(proba.reindex(bars.index).to_numpy(), scfg.entry_threshold)
    res = core.backtest(*(bars[k].to_numpy(float) for k in ("open", "high", "low", "close")), unit, signal,
                        sl_mult=mcfg.barrier_mult, tp_mult=mcfg.barrier_mult, max_hold=mcfg.horizon,
                        risk_fraction=scfg.risk_per_trade, cost=round_trip_cost(spec, scfg.slippage_pips))
    equity = pd.Series(res["equity"], index=bars.index)
    tr = res["trades"]
    decision = tr["entry_idx"].astype(int) - 1  # bar whose close produced the signal
    p_dec = proba.reindex(bars.index).to_numpy()[decision]
    cost_r = round_trip_cost(spec, scfg.slippage_pips) / (mcfg.barrier_mult * unit[decision])
    trades = pd.DataFrame({
        "entry_time": bars.index[tr["entry_idx"].astype(int)],
        "exit_time": bars.index[tr["exit_idx"].astype(int)],
        "direction": tr["direction"].astype(int),
        "entry_px": tr["entry_px"], "exit_px": tr["exit_px"], "r_multiple": tr["r_multiple"],
        "exit_reason": [EXIT_REASONS[int(r)] for r in tr["reason"]],
        "signal_edge": np.abs(p_dec - 0.5), "cost_r": cost_r, "gross_r": tr["r_multiple"] + cost_r,
    })
    holds = tr["exit_idx"] - tr["entry_idx"] + 1
    metrics = performance_metrics(equity, tr["r_multiple"], res["position"], holds)
    return {"metrics": metrics, "equity": equity, "trades": trades}


EDGE_BUCKETS = (0.0, 0.05, 0.08, 0.12, 1.0)


def expectancy_by_edge(trades: pd.DataFrame) -> pd.DataFrame:
    """Realised gross R (before costs) of out-of-sample trades, by signal strength."""
    if trades.empty:
        return pd.DataFrame(columns=["trades", "gross_r", "net_r", "win_rate"])
    bucket = pd.cut(trades["signal_edge"], bins=list(EDGE_BUCKETS), include_lowest=True)
    out = trades.groupby(bucket, observed=True).agg(
        trades=("gross_r", "size"), gross_r=("gross_r", "mean"), net_r=("r_multiple", "mean"),
        win_rate=("r_multiple", lambda r: float((r > 0).mean())))
    return out


def portfolio_metrics(equities: dict[str, pd.Series]) -> tuple[dict, pd.Series]:
    """Equal-capital portfolio of all per-pair strategies (daily rebalanced)."""
    daily = pd.DataFrame({p: _daily(e) for p, e in equities.items() if len(e)})
    if daily.empty:
        return {}, pd.Series(dtype=float)
    rets = daily.pct_change().fillna(0.0).mean(axis=1)
    curve = (1 + rets).cumprod()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    total = float(curve.iloc[-1] - 1)
    return {
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(260)) if rets.std() > 0 else np.nan,
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
        "days": int(len(rets)),
    }, curve

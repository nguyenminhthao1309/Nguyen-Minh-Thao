"""Feature engineering and triple-barrier labels.

All features are causal and scale-free (normalised by volatility or ATR) so a
single pooled model can learn from every pair at once. Higher-timeframe
features only use completed higher-timeframe bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core
from .config import ModelConfig, SEGMENTS, pair_spec
from .regime import classify_regimes
from .sessions import primary_session, session_membership
from .strength import currency_strength, strength_ranks

WINDOWS = {
    "intraday": dict(mom=(1, 4, 24, 120, 480), vol=(24, 120), er=(24, 120), vr=240, z=120,
                     range=120, pct=500, slope=24, warmup=500),
    "daily": dict(mom=(1, 3, 5, 20, 60), vol=(10, 60), er=(10, 40), vr=120, z=40,
                  range=40, pct=250, slope=5, warmup=250),
}

META_COLUMNS = ["time", "pair", "segment", "session", "close", "atr", "label", "label_ret",
                "label_end_time", "target", "vol_regime", "trend_regime"]


def infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    return diffs.mode().iloc[0] if len(diffs) else pd.Timedelta("1h")


def bars_per_day(delta: pd.Timedelta) -> int:
    return max(1, int(round(pd.Timedelta("1D") / delta)))


def _htf_rules(delta: pd.Timedelta) -> list[tuple[str, str, str]]:
    """(resample rule, feature prefix, offset) of the higher timeframes to add."""
    # Fixed-length rules so origin/offset apply: FX days start 22:00 UTC and
    # weeks start Sunday 22:00 UTC (the epoch is a Thursday, +94h).
    if delta < pd.Timedelta("1h"):
        return [("1h", "h1", "0h"), ("4h", "h4", "22h"), ("24h", "d1", "22h")]
    if delta < pd.Timedelta("4h"):
        return [("4h", "h4", "22h"), ("24h", "d1", "22h")]
    if delta < pd.Timedelta("1D"):
        return [("24h", "d1", "22h")]
    return [("168h", "w1", "94h")]


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def htf_features(df: pd.DataFrame, rule: str, prefix: str, offset: str, delta: pd.Timedelta) -> pd.DataFrame:
    """Indicators on a higher timeframe, aligned to base bars without look-ahead.

    A higher-timeframe bar becomes visible at the base bar that completes it
    (bucket start + rule - base delta), never earlier.
    """
    agg = (df.resample(rule, origin="epoch", offset=offset, label="left", closed="left")
             .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    cols = [f"{prefix}_rsi", f"{prefix}_ret", f"{prefix}_ema20_dist", f"{prefix}_ema20_slope", f"{prefix}_adx"]
    if len(agg) < 30:
        return pd.DataFrame(np.nan, index=df.index, columns=cols)
    h, l, c = (agg[k].to_numpy() for k in ("high", "low", "close"))
    a = core.atr(h, l, c, 14)
    e20 = core.ema(c, 20)
    feats = pd.DataFrame({
        cols[0]: (core.rsi(c, 14) - 50.0) / 50.0,
        cols[1]: (c - _shift(c, 1)) / a,
        cols[2]: (c - e20) / a,
        cols[3]: (e20 - _shift(e20, 3)) / a,
        cols[4]: core.adx(h, l, c, 14) / 100.0,
    }, index=agg.index + pd.Timedelta(rule) - delta)
    return feats.reindex(df.index, method="ffill")


def build_features(df: pd.DataFrame, pair: str, ranks: dict[str, pd.DataFrame] | None = None,
                   atr_period: int = 14) -> pd.DataFrame:
    """Feature matrix for one pair (one row per bar)."""
    spec = pair_spec(pair)
    delta = infer_bar_delta(df.index)
    intraday = delta < pd.Timedelta("1D")
    w = WINDOWS["intraday" if intraday else "daily"]

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    logc = np.log(c)
    r = np.r_[0.0, np.diff(logc)]
    vol_fast = core.rolling_std(r, w["vol"][0])
    vol_slow = core.rolling_std(r, w["vol"][1])
    a = core.atr(h, l, c, atr_period)
    a_slow = core.atr(h, l, c, 100)
    macd = core.ema(c, 12) - core.ema(c, 26)
    e50, e200 = core.ema(c, 50), core.ema(c, 200)
    spread = spec.spread_pips * spec.pip_size

    with np.errstate(invalid="ignore", divide="ignore"):
        f = {f"mom_{k}": (logc - _shift(logc, k)) / (vol_slow * np.sqrt(k)) for k in w["mom"]}
        f.update({
            "rsi_14": (core.rsi(c, 14) - 50.0) / 50.0,
            "stoch_14": core.stochastic_k(h, l, c, 14) / 100.0 - 0.5,
            "range_pos": core.stochastic_k(h, l, c, w["range"]) / 100.0 - 0.5,
            "bb_pctb": core.bollinger_pctb(c, 20, 2.0) - 0.5,
            "macd_hist": (macd - core.ema(macd, 9)) / a,
            "ema50_dist": (c - e50) / a,
            "ema200_dist": (c - e200) / a,
            "ema50_slope": (e50 - _shift(e50, w["slope"])) / a,
            "adx_14": core.adx(h, l, c, 14) / 100.0,
            "er_fast": core.efficiency_ratio(c, w["er"][0]),
            "er_slow": core.efficiency_ratio(c, w["er"][1]),
            "variance_ratio": core.variance_ratio(r, w["vr"], 4) - 1.0,
            "zscore": core.rolling_zscore(c, w["z"]),
            "vol_ratio": vol_fast / vol_slow,
            "atr_ratio": a / a_slow,
            "vol_pct": core.rolling_percentile_rank(vol_fast, w["pct"]),
            "bar_range_atr": (h - l) / a,
            "gap_atr": (o - _shift(c, 1)) / a,
            "cost_atr": spread / a,
        })
    feats = pd.DataFrame(f, index=df.index)

    for rule, prefix, offset in _htf_rules(delta):
        feats = feats.join(htf_features(df, rule, prefix, offset, delta))

    if intraday:
        hours = df.index.hour + df.index.minute / 60.0
        feats["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        member = session_membership(df.index)
        for name in ["Sydney", "Tokyo", "London", "New York"]:
            feats[f"in_{name.lower().replace(' ', '_')}"] = member[name].astype(float)
    feats["dow"] = df.index.weekday.astype(float)

    if ranks:
        for label, table in ranks.items():
            aligned = table.reindex(df.index)
            for side, ccy in (("base", spec.base), ("quote", spec.quote)):
                feats[f"{side}_strength_{label}"] = aligned[ccy] if ccy in aligned else np.nan
            feats[f"strength_spread_{label}"] = feats[f"base_strength_{label}"] - feats[f"quote_strength_{label}"]

    for seg in SEGMENTS:
        feats[f"seg_{seg}"] = float(spec.segment == seg)
    feats["atr"] = a  # kept for labeling / sizing, excluded from model inputs
    return feats.replace([np.inf, -np.inf], np.nan)


def risk_unit(atr: np.ndarray, horizon: int) -> np.ndarray:
    """Typical price move over the holding horizon: ATR scaled by sqrt(horizon)."""
    return np.asarray(atr, dtype=float) * np.sqrt(horizon)


def add_labels(df: pd.DataFrame, atr: np.ndarray, cfg: ModelConfig) -> pd.DataFrame:
    """Triple-barrier outcome: does price move +k risk units before -k within the horizon?"""
    unit = risk_unit(atr, cfg.horizon)
    res = core.triple_barrier(df["high"].to_numpy(float), df["low"].to_numpy(float),
                              df["close"].to_numpy(float), unit, cfg.horizon, cfg.barrier_mult, cfg.barrier_mult)
    out = pd.DataFrame({"label": res["label"], "label_ret": res["ret"]}, index=df.index)
    end_idx = res["end_idx"]
    valid = np.isfinite(end_idx)
    end_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    end_time[valid] = df.index[end_idx[valid].astype(int)]
    out["label_end_time"] = end_time
    out["target"] = np.where(valid, (res["ret"] > 0).astype(float), np.nan)
    return out


def build_dataset(data: dict[str, pd.DataFrame], cfg: ModelConfig) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Pooled, time-sorted dataset for all pairs, the feature column list and per-bar currency strength."""
    strength = currency_strength(data)
    delta = infer_bar_delta(next(iter(data.values())).index)
    bpd = bars_per_day(delta)
    ranks = {"day": strength_ranks(strength, bpd), "week": strength_ranks(strength, 5 * bpd)} \
        if len(strength.columns) > 2 else None
    warmup = WINDOWS["intraday" if delta < pd.Timedelta("1D") else "daily"]["warmup"]

    frames = []
    for pair, df in data.items():
        if len(df) < warmup + cfg.horizon + 50:
            print(f"[features] {pair}: not enough history ({len(df)} bars), skipped")
            continue
        feats = build_features(df, pair, ranks, cfg.atr_period)
        labels = add_labels(df, feats["atr"].to_numpy(), cfg)
        frame = feats.join(labels)
        frame["time"] = df.index
        frame["pair"] = pair
        frame["segment"] = pair_spec(pair).segment
        frame["close"] = df["close"].to_numpy()
        frame["session"] = primary_session(df.index).to_numpy() if delta < pd.Timedelta("1D") else "Daily"
        frame = frame.join(classify_regimes(frame))
        frames.append(frame.iloc[warmup:])
    if not frames:
        raise ValueError("no pair has enough history to build features")
    dataset = pd.concat(frames).sort_values(["time", "pair"]).reset_index(drop=True)
    feature_cols = [c for c in dataset.columns if c not in META_COLUMNS]
    return dataset, feature_cols, strength

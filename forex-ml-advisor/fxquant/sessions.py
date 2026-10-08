"""Trading-session analysis: Sydney, Tokyo, London, New York and their overlaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SESSIONS, pair_spec

SESSION_GROUPS = ["Sydney", "Tokyo", "London", "New York", "London/NY overlap", "Off-hours"]


def session_membership(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean table: does the bar starting at each timestamp fall in each session?

    Session hours are defined in local exchange time, so daylight-saving
    transitions in each city are handled automatically.
    """
    out = {}
    for name, (tz, start, end) in SESSIONS.items():
        local = index.tz_convert(tz)
        out[name] = (local.hour >= start) & (local.hour < end) & (local.weekday < 5)
    df = pd.DataFrame(out, index=index)
    df["London/NY overlap"] = df["London"] & df["New York"]
    df["Off-hours"] = ~df[list(SESSIONS)].any(axis=1)
    return df


def primary_session(index: pd.DatetimeIndex) -> pd.Series:
    """Single most relevant session label per bar (overlap > NY > London > Tokyo > Sydney)."""
    m = session_membership(index)
    label = pd.Series("Off-hours", index=index)
    for name in ["Sydney", "Tokyo", "London", "New York", "London/NY overlap"]:
        label[m[name].to_numpy()] = name
    return label


def session_stats(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per pair and session: range, absolute move, share of variance and drift."""
    rows = []
    for pair, df in data.items():
        spec = pair_spec(pair)
        r = np.log(df["close"]).diff()
        rng_pips = (df["high"] - df["low"]) / spec.pip_size
        member = session_membership(df.index)
        total_var = float((r ** 2).sum())
        for name in SESSION_GROUPS:
            mask = member[name].to_numpy() & r.notna().to_numpy()
            if mask.sum() < 30:
                continue
            rs = r[mask]
            drift_bps = rs.mean() * 1e4
            drift_t = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs))) if rs.std() > 0 else 0.0
            rows.append({
                "pair": pair, "segment": spec.segment, "session": name, "bars": int(mask.sum()),
                "avg_range_pips": float(rng_pips[mask].mean()),
                "mean_abs_move_bps": float(rs.abs().mean() * 1e4),
                "variance_share": float((rs ** 2).sum() / total_var) if total_var > 0 else np.nan,
                "drift_bps_per_bar": float(drift_bps),
                "drift_t_stat": float(drift_t),
            })
    return pd.DataFrame(rows)


def segment_session_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """Average |move| per bar (bps) by segment x session."""
    if stats.empty:
        return stats
    table = stats.pivot_table(index="segment", columns="session", values="mean_abs_move_bps", aggfunc="mean")
    return table.reindex(columns=[c for c in SESSION_GROUPS if c in table.columns])


def hourly_profile(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average |return| in bps for each UTC hour, by segment."""
    frames = []
    for pair, df in data.items():
        r = np.log(df["close"]).diff().abs() * 1e4
        frames.append(pd.DataFrame({"hour": df.index.hour, "segment": pair_spec(pair).segment, "abs_bps": r}))
    allr = pd.concat(frames)
    return allr.pivot_table(index="hour", columns="segment", values="abs_bps", aggfunc="mean")


def best_session(stats: pd.DataFrame, pair: str) -> str:
    """Most liquid / active session for a pair (largest average range, overlap preferred on ties)."""
    s = stats[(stats["pair"] == pair) & (stats["session"] != "Off-hours")]
    if s.empty:
        return "London/NY overlap"
    return str(s.sort_values("avg_range_pips", ascending=False).iloc[0]["session"])

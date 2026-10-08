"""Market data: Yahoo Finance, CSV files, or a synthetic FX market.

Every loader returns ``dict[pair, DataFrame]`` with a UTC DatetimeIndex
(bar open time) and float columns ``open, high, low, close``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import UNIVERSE, pair_spec

OHLC = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_ohlc(df: pd.DataFrame, max_jump_mads: float = 25.0) -> pd.DataFrame:
    """Sort, de-duplicate, drop bad ticks and enforce high >= max(open, close)."""
    df = df[OHLC].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df > 0).all(axis=1)]
    if len(df) > 50:
        # Remove isolated spikes: returns far outside the robust dispersion.
        r = np.log(df["close"]).diff()
        mad = (r - r.median()).abs().rolling(500, min_periods=50).median()
        spike = (r.abs() > max_jump_mads * 1.4826 * mad) & (r.shift(-1) * r < 0)
        df = df[~spike.fillna(False)]
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def _to_utc(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Yahoo Finance (requires `pip install yfinance` and internet access)
# ---------------------------------------------------------------------------

def load_yahoo(pairs: list[str], interval: str = "1h", period: str = "730d") -> dict[str, pd.DataFrame]:
    """Download OHLC bars from Yahoo Finance (hourly history is limited to ~730 days)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Live data needs yfinance: pip install yfinance") from exc

    out: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        hist = yf.Ticker(f"{pair}=X").history(period=period, interval=interval, auto_adjust=False)
        if hist.empty:
            print(f"[data] no Yahoo data for {pair}, skipping")
            continue
        hist.columns = [c.lower() for c in hist.columns]
        hist.index = _to_utc(hist.index)
        out[pair] = drop_unfinished_bar(clean_ohlc(hist))
    return out


def drop_unfinished_bar(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Live feeds include the bar that is still forming; it must not be treated as closed."""
    if len(df) < 3:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    bar = df.index.to_series().diff().mode().iloc[0]
    return df[df.index + bar <= now]


# ---------------------------------------------------------------------------
# CSV files: one file per pair named like EURUSD.csv (MT4/MT5, Dukascopy, ...)
# ---------------------------------------------------------------------------

_TIME_COLUMNS = ("timestamp", "datetime", "date_time", "gmt time", "local time", "time", "date")


def read_csv_ohlc(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower().lstrip("<").rstrip(">") for c in raw.columns]
    if "date" in raw.columns and "time" in raw.columns:
        stamps = raw["date"].astype(str) + " " + raw["time"].astype(str)
    else:
        col = next((c for c in _TIME_COLUMNS if c in raw.columns), raw.columns[0])
        stamps = raw[col]
    aliases = {"o": "open", "h": "high", "l": "low", "c": "close", "bid": "close"}
    raw = raw.rename(columns={k: v for k, v in aliases.items() if k in raw.columns and v not in raw.columns})
    missing = [c for c in OHLC if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    stamps = pd.Series(stamps).astype(str).str.replace(".", "-", n=2, regex=False)
    df = raw[OHLC].copy()
    df.index = _to_utc(pd.to_datetime(stamps, utc=False, format="mixed", dayfirst=False))
    return clean_ohlc(df)


def load_csv_dir(directory: str | Path, pairs: list[str] | None = None) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    out: dict[str, pd.DataFrame] = {}
    for file in sorted(directory.glob("*.csv")):
        pair = "".join(ch for ch in file.stem.upper() if ch.isalpha())[:6]
        if len(pair) != 6 or (pairs and pair not in pairs):
            continue
        out[pair] = read_csv_ohlc(file)
    if not out:
        raise FileNotFoundError(f"no <PAIR>.csv files found in {directory}")
    return out


# ---------------------------------------------------------------------------
# Synthetic market
# ---------------------------------------------------------------------------
# Each currency's log value against an abstract world numeraire follows
#   d s_c = mu_c dt + beta_c . factors + sigma_c(t) dW_c + jumps
# Pairs are s_base - s_quote, so all cross rates are arbitrage-free. Volatility
# has a global calm/stress regime, per-currency stochastic vol and intraday
# session seasonality. mu_c is a slow mean-reverting drift, i.e. a weak
# momentum effect the models can learn. It exists to exercise the pipeline;
# results on synthetic data say nothing about real markets.

# currency: (start value in USD, idiosyncratic annual vol, risk-on beta, europe beta, carry drift/yr)
_CCY = {
    "USD": (1.0, 0.050, -0.2, 0.0, 0.00),
    "EUR": (1.08, 0.045, 0.0, 0.5, 0.00),
    "GBP": (1.27, 0.055, 0.1, 0.3, 0.00),
    "JPY": (1 / 150, 0.060, -0.6, 0.0, 0.00),
    "CHF": (1 / 0.88, 0.050, -0.4, 0.4, 0.00),
    "AUD": (0.66, 0.060, 0.6, 0.0, 0.00),
    "NZD": (0.60, 0.065, 0.6, 0.0, 0.00),
    "CAD": (1 / 1.36, 0.045, 0.3, 0.0, 0.00),
    "MXN": (1 / 17.5, 0.090, 0.8, 0.0, -0.02),
    "ZAR": (1 / 18.5, 0.110, 1.0, 0.0, -0.04),
    "TRY": (1 / 32.0, 0.100, 0.6, 0.0, -0.25),
    "SGD": (1 / 1.34, 0.025, 0.2, 0.0, 0.00),
    "NOK": (1 / 10.6, 0.070, 0.4, 0.4, 0.00),
    "SEK": (1 / 10.4, 0.070, 0.3, 0.5, 0.00),
    "PLN": (1 / 4.0, 0.060, 0.4, 0.6, 0.00),
}
_ASIA = {"JPY", "AUD", "NZD", "SGD"}
_HOURS_PER_YEAR = 52 * 120


def fx_trading_hours(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Hourly bar index of the FX week (Sunday 22:00 to Friday 22:00 UTC)."""
    idx = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    wd, hr = idx.weekday, idx.hour
    closed = (wd == 5) | ((wd == 6) & (hr < 22)) | ((wd == 4) & (hr >= 22))
    return idx[~closed]


def _seasonality(hours: np.ndarray, ccy: str) -> np.ndarray:
    profile = np.ones(24)
    profile[0:7] = 1.0 if ccy in _ASIA else 0.6
    profile[7:12] = 1.2
    profile[12:17] = 1.5
    profile[17:21] = 0.9
    profile[21:24] = 0.5
    return profile[hours]


def generate_synthetic(pairs: list[str] | None = None, years: float = 2.0, end: str | pd.Timestamp | None = None,
                       seed: int = 42, momentum: float = 0.06, substeps: int = 4) -> dict[str, pd.DataFrame]:
    """Simulate an internally consistent hourly FX market.

    `momentum` sets the strength of the persistent drift relative to hourly
    volatility (0 gives driftless random walks with nothing to predict).
    """
    pairs = pairs or list(UNIVERSE)
    for p in pairs:
        if p[:3] not in _CCY or p[3:] not in _CCY:
            raise ValueError(f"synthetic market does not include currencies of {p}")
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    index = fx_trading_hours(end_ts - pd.Timedelta(days=int(365 * years)), end_ts)
    n, k = len(index), substeps
    ccys = list(_CCY)
    m = len(ccys)
    rng = np.random.default_rng(seed)

    start_val = np.log(np.array([_CCY[c][0] for c in ccys]))
    idio = np.array([_CCY[c][1] for c in ccys]) / np.sqrt(_HOURS_PER_YEAR)
    beta_risk = np.array([_CCY[c][2] for c in ccys])
    beta_eur = np.array([_CCY[c][3] for c in ccys])
    carry = np.array([_CCY[c][4] for c in ccys]) / _HOURS_PER_YEAR

    # Global calm/stress volatility regime (Markov chain).
    stress = np.zeros(n, dtype=bool)
    u = rng.random(n)
    for t in range(1, n):
        stress[t] = (u[t] < 1 / 480) if not stress[t - 1] else (u[t] > 1 / 168)
    regime_mult = np.where(stress, 1.5, 0.8)

    # Per-currency stochastic log-volatility, AR(1).
    log_vol = np.zeros((n, m))
    shocks = rng.normal(0.0, 0.03, (n, m))
    for t in range(1, n):
        log_vol[t] = 0.995 * log_vol[t - 1] + shocks[t]

    # Slow mean-reverting drift (weak momentum), half-life ~ 10 trading days.
    phi = np.exp(-np.log(2) / 240)
    mu_sd = momentum * idio
    mu = np.zeros((n, m))
    mu_shocks = rng.normal(0.0, 1.0, (n, m)) * mu_sd * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        mu[t] = phi * mu[t - 1] + mu_shocks[t]

    hours = index.hour.to_numpy()
    season = np.column_stack([_seasonality(hours, c) for c in ccys])
    sigma = idio * season * regime_mult[:, None] * np.exp(log_vol)  # (n, m) hourly vol

    f_risk = rng.normal(0.0, 0.05 / np.sqrt(_HOURS_PER_YEAR), (n, k)) * regime_mult[:, None]
    f_eur = rng.normal(0.0, 0.03 / np.sqrt(_HOURS_PER_YEAR), (n, k))
    eps = rng.normal(0.0, 1.0, (n, k, m)) * (sigma / np.sqrt(k))[:, None, :]
    steps = (eps + f_risk[:, :, None] * beta_risk + f_eur[:, :, None] * beta_eur
             + ((mu + carry) / k)[:, None, :])

    # Event jumps (data releases) and weekend gaps, applied on the first sub-step.
    jumps = (rng.random((n, m)) < 1 / 600) * rng.normal(0.0, 6.0, (n, m)) * sigma
    gap = np.r_[False, np.diff(index.asi8) > 3_600_000_000_000]
    jumps[gap] += rng.normal(0.0, 3.0, (gap.sum(), m)) * sigma[gap]
    steps[:, 0, :] += jumps

    path = start_val + np.cumsum(steps.reshape(n * k, m), axis=0)  # (n*k, m)
    pos = {c: i for i, c in enumerate(ccys)}
    out: dict[str, pd.DataFrame] = {}
    for p in pairs:
        lp = (path[:, pos[p[:3]]] - path[:, pos[p[3:]]]).reshape(n, k)
        first = start_val[pos[p[:3]]] - start_val[pos[p[3:]]]
        close = lp[:, -1]
        open_ = np.r_[first, close[:-1]]
        high = np.maximum(open_, lp.max(axis=1))
        low = np.minimum(open_, lp.min(axis=1))
        df = pd.DataFrame(np.exp(np.column_stack([open_, high, low, close])), index=index, columns=OHLC)
        decimals = 3 if p.endswith("JPY") else 5
        if pair_spec(p).segment == "exotic":
            decimals = 4
        out[p] = df.round(decimals)
    return out


def load_data(source: str, pairs: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    if source == "synthetic":
        return generate_synthetic(pairs, years=kwargs.get("years", 2.0), seed=kwargs.get("seed", 42),
                                  end=kwargs.get("end"), momentum=kwargs.get("momentum", 0.06))
    if source == "yahoo":
        return load_yahoo(pairs, interval=kwargs.get("interval", "1h"), period=kwargs.get("period", "730d"))
    if source == "csv":
        return load_csv_dir(kwargs["csv_dir"], pairs)
    raise ValueError(f"unknown data source: {source}")

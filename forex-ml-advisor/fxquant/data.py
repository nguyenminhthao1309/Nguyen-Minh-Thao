"""Market data: Yahoo Finance, CSV files, or a synthetic FX market.

Every loader returns ``dict[pair, DataFrame]`` with a UTC DatetimeIndex
(bar open time) and float columns ``open, high, low, close``.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UNIVERSE, pair_spec

OHLC = ["open", "high", "low", "close"]


class DataError(Exception):
    """A data problem the user can fix (shown as a short message, not a traceback)."""


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
    except ImportError as exc:  # optional dependency
        raise DataError("Live data needs the yfinance package. Run the setup again, or install it with: "
                        "python -m pip install yfinance") from exc

    import logging

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # its retries/delisting notes are noise here
    out: dict[str, pd.DataFrame] = {}
    misses = 0
    for pair in pairs:
        try:
            hist = yf.Ticker(f"{pair}=X").history(period=period, interval=interval, auto_adjust=False)
            reason = "no data returned" if hist is None or hist.empty else ""
        except Exception as exc:  # network down, firewall, rate limit, Yahoo API change ...
            hist, reason = None, exc.__class__.__name__
        if reason:
            misses += 1
            print(f"[data] {pair}: could not download from Yahoo Finance ({reason})")
            if not out and misses >= 3:
                print("[data] giving up: the first pairs all failed, so Yahoo Finance is probably unreachable")
                break
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


def resolve_timezone(text: str | None):
    """'UTC', an IANA name such as 'Europe/Athens', or a fixed offset such as 'UTC+2' / '+02:00'."""
    if text is None or str(text).strip().upper() in ("", "UTC", "GMT", "Z"):
        return "UTC"
    m = re.fullmatch(r"(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?", str(text).strip(), flags=re.I)
    if m:
        minutes = int(m.group(2)) * 60 + int(m.group(3) or 0)
        return dt.timezone(dt.timedelta(minutes=minutes if m.group(1) == "+" else -minutes))
    try:
        pd.Timestamp("2026-01-01").tz_localize(str(text).strip())
    except Exception as exc:
        raise DataError(f"Unknown time zone '{text}'. Use UTC, an offset like UTC+2, or a name like "
                        "Europe/Athens") from exc
    return str(text).strip()


def parse_timestamps(stamps: pd.Series, timezone: str | None = None) -> pd.DatetimeIndex:
    """Parse broker time stamps into UTC, deciding day/month order from the data itself.

    Year-first dates (2026-03-02, 2026.03.02) are unambiguous. For day/month
    dates the column is scanned: a first field above 12 means day-first, a
    second field above 12 means month-first; otherwise dotted dates
    (02.03.2026, Dukascopy/European) are read day-first and slashed dates
    (03/02/2026) month-first. Stamps without a zone are taken to be in
    `timezone` (default UTC); stamps with an explicit offset are converted.
    """
    s = pd.Series(stamps).astype(str).str.strip().reset_index(drop=True)
    head = s.str.extract(r"^(\d{1,4})([./-])(\d{1,2})[./-](\d{1,4})")
    if head.notna().all(axis=None) and len(s):
        rest = s.str.replace(r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}", "", regex=True)
        if (head[0].str.len() == 4).all():  # year first
            year, month, day = head[0], head[2], head[3]
        else:
            first, second = head[0].astype(int), head[2].astype(int)
            if first.max() > 12:
                day_first = True
            elif second.max() > 12:
                day_first = False
            else:
                day_first = head[1].iloc[0] == "."
            year = head[3]
            day, month = (head[0], head[2]) if day_first else (head[2], head[0])
        s = year + "-" + month.str.zfill(2) + "-" + day.str.zfill(2) + rest
    tz = resolve_timezone(timezone)
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(s, format="mixed"))
    except (ValueError, TypeError):  # mixed UTC offsets in one column
        return pd.DatetimeIndex(pd.to_datetime(s, format="mixed", utc=True))
    if parsed.tz is not None:
        return parsed.tz_convert("UTC")
    local = parsed.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return local.tz_convert("UTC")


def _sniff_csv(path: Path) -> tuple[str, str, bool]:
    """(encoding, separator, has_header) of a price file."""
    head = path.read_bytes()[:4096]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif head.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "latin-1"
    text = head.decode(encoding, errors="ignore").lstrip("\ufeff")
    first = text.splitlines()[0] if text.strip() else ""
    if "\t" in first:
        sep = "\t"
    elif ";" in first and "," not in first:
        sep = ";"
    else:
        sep = ","
    has_header = not first.strip()[:1].isdigit()  # MT4 History Center exports have no header row
    return encoding, sep, has_header


def read_csv_ohlc(path: str | Path, timezone: str | None = None) -> pd.DataFrame:
    path = Path(path)
    encoding, sep, has_header = _sniff_csv(path)
    raw = pd.read_csv(path, sep=sep, encoding=encoding, header=0 if has_header else None)
    if not has_header:
        two_columns = raw.shape[1] >= 6 and raw.iloc[:, 1].astype(str).str.contains(":").all()
        names = ["date", "time"] + OHLC if two_columns else ["timestamp"] + OHLC
        names += [f"extra{i}" for i in range(raw.shape[1] - len(names))]
        raw.columns = names[:raw.shape[1]]
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
        raise DataError(f"{path.name}: missing columns {missing} (found {list(raw.columns)[:8]}). "
                        "See data/README.md for the supported layouts.")
    df = raw[OHLC].apply(pd.to_numeric, errors="coerce")
    df.index = parse_timestamps(stamps, timezone)
    df = df[df.index.notna()]
    return clean_ohlc(df)


def load_csv_dir(directory: str | Path, pairs: list[str] | None = None,
                 timezone: str | None = None) -> dict[str, pd.DataFrame]:
    """Load every <PAIR>.csv in `directory` (only `pairs` if given). Unreadable files are skipped."""
    directory = Path(directory)
    if not directory.is_dir():
        raise DataError(f"The folder '{directory}' does not exist. Create it and put price files such as "
                        f"{directory / 'EURUSD.csv'} in it (see data/README.md).")
    files = {}
    for file in sorted(directory.glob("*.csv")):
        pair = "".join(ch for ch in file.stem.upper() if ch.isalpha())[:6]
        if len(pair) == 6:
            files.setdefault(pair, file)
    if not files:
        raise DataError(f"No price files found in '{directory}'. Put one CSV per pair there, e.g. "
                        f"{directory / 'EURUSD.csv'} (see data/README.md), or create sample files with: "
                        f"python -m fxquant save-data --source synthetic --out {directory}")
    wanted = list(files) if pairs is None else [p for p in pairs if p in files]
    if pairs is not None:
        absent = [p for p in pairs if p not in files]
        if absent and len(absent) < len(pairs):
            print(f"[data] no file for {', '.join(absent)} in {directory}, skipping")
        if not wanted:
            raise DataError(f"None of the requested pairs has a file in '{directory}'. "
                            f"Files found: {', '.join(sorted(files))}")
    out: dict[str, pd.DataFrame] = {}
    for pair in wanted:
        try:
            df = read_csv_ohlc(files[pair], timezone)
        except DataError as exc:
            print(f"[data] skipping {files[pair].name}: {exc}")
            continue
        except Exception as exc:
            print(f"[data] skipping {files[pair].name}: could not read it ({exc.__class__.__name__}: {exc})")
            continue
        if df.empty:
            print(f"[data] skipping {files[pair].name}: no valid rows")
            continue
        out[pair] = df
    if not out:
        raise DataError(f"No file in '{directory}' could be read. See data/README.md for the supported layouts.")
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
            raise DataError(f"The synthetic market has no {p}. It covers these currencies: {', '.join(_CCY)}")
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
        return load_csv_dir(kwargs["csv_dir"], pairs, timezone=kwargs.get("timezone"))
    raise DataError(f"unknown data source: {source}")

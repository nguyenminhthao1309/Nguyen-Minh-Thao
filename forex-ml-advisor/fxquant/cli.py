"""Command line interface: `fxquant analyze`, `fxquant save-data` and `fxquant bench`."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

from .config import UNIVERSE, Config, SEGMENTS, pair_spec


class UserError(Exception):
    """A problem the user can fix; printed as a short message instead of a traceback."""


def normalise_pair(text: str) -> str:
    """'eur/usd', 'EUR_USD', 'EURUSD=X' -> 'EURUSD'."""
    pair = re.sub(r"=X$", "", text.strip().upper())
    pair = re.sub(r"[^A-Z]", "", pair)
    if len(pair) != 6:
        raise UserError(f"'{text.strip()}' is not a currency pair. Use six letters such as EURUSD or GBPJPY.")
    return pair


def parse_segments(text: str) -> list[str]:
    segments = []
    for raw in text.split(","):
        seg = raw.strip().lower()
        seg = {"majors": "major", "crosses": "cross", "minor": "cross", "minors": "cross",
               "exotics": "exotic"}.get(seg, seg)
        if not seg:
            continue
        if seg not in SEGMENTS:
            raise UserError(f"Unknown segment '{raw.strip()}'. Use any of: {', '.join(SEGMENTS)}.")
        segments.append(seg)
    return segments


def _selection(args) -> tuple[list[str] | None, list[str] | None]:
    """(pairs, segments) requested on the command line; None means no restriction.

    For CSV files the pair list is not limited to the built-in universe: by
    default every <PAIR>.csv in the folder is used.
    """
    if args.pairs:
        pairs = list(dict.fromkeys(normalise_pair(p) for p in args.pairs.split(",") if p.strip()))
        if not pairs:
            raise UserError("--pairs is empty.")
        return pairs, None
    segments = parse_segments(args.segments) if args.segments else None
    if getattr(args, "source", "") == "csv":
        return None, segments
    return [p for p, spec in UNIVERSE.items() if segments is None or spec.segment in segments], None


def _load(args) -> dict:
    from .data import DataError, load_data, resolve_timezone

    pairs, segments = _selection(args)
    if args.source == "csv":
        try:
            resolve_timezone(getattr(args, "csv_timezone", None))
        except DataError as exc:
            raise UserError(str(exc)) from exc
    verb = {"synthetic": "generating", "yahoo": "downloading", "csv": "loading"}[args.source]
    what = f"{len(pairs)} pairs" if pairs else "price files"
    where = f" from {args.csv_dir}" if args.source == "csv" else f" ({args.source})"
    print(f"[cli] {verb} {what}{where} ...")
    try:
        data = load_data(args.source, pairs, csv_dir=getattr(args, "csv_dir", "data"),
                         timezone=getattr(args, "csv_timezone", None), interval=args.interval,
                         period=args.period, years=args.years, seed=args.seed, momentum=args.momentum)
    except DataError as exc:
        raise UserError(str(exc)) from exc
    if segments:
        data = {p: df for p, df in data.items() if pair_spec(p).segment in segments}
    if not data:
        if args.source == "yahoo":
            raise UserError("No data loaded: Yahoo Finance could not be reached. Check your internet "
                            "connection; Yahoo also rate-limits, so wait a few minutes or use fewer pairs.")
        raise UserError(f"No data loaded for the requested pairs/segments from {args.source}.")
    return data


def cmd_analyze(args) -> None:
    from .advisor import ForexAdvisor
    from .report import console_summary, save_report

    from .features import infer_bar_delta

    if args.equity <= 0 or not 0 < args.risk <= 5:
        raise UserError("--equity must be positive and --risk between 0 and 5 (percent of equity).")
    if not re.fullmatch(r"[A-Za-z]{3}", args.account_currency.strip()):
        raise UserError(f"Account currency '{args.account_currency}' should be a 3-letter code such as USD or EUR.")
    data = _load(args)
    cfg = Config(pairs=list(data))
    cfg.advisor.account_equity = args.equity
    cfg.advisor.account_currency = args.account_currency.strip().upper()
    cfg.advisor.risk_per_trade = args.risk / 100
    daily = infer_bar_delta(next(iter(data.values())).index) >= np.timedelta64(1, "D")
    cfg.model.horizon = args.horizon or (5 if daily else 24)
    if daily:
        cfg.model.train_stride = 1  # daily labels barely overlap, keep every row
    if args.fast:
        cfg.model.retrain_every = "90D"
        cfg.model.train_stride *= 2

    result = ForexAdvisor(cfg).run(data)
    paths = save_report(result, args.out)
    print()
    print(console_summary(result))
    if "report" in paths:
        print(f"\nReport: {paths['report']}\nIdeas:  {paths.get('ideas', '-')}")
    if "latest" in paths:
        print(f"Latest report is always at {paths['latest']} (in VS Code: open it, then Ctrl+Shift+V / Cmd+Shift+V)")


def cmd_save_data(args) -> None:
    """Save market data as one CSV per pair, merging with files already there."""
    import pandas as pd

    from .data import read_csv_ohlc

    data = _load(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for pair, df in data.items():
        path = out / f"{pair}.csv"
        if path.exists():  # keep older history so the archive grows beyond the feed's limit
            try:
                old = read_csv_ohlc(path)
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            except Exception as exc:
                print(f"  could not merge with the existing {path.name} ({exc}); overwriting it")
        try:
            df.rename_axis("timestamp").to_csv(path)
        except PermissionError:
            print(f"  could not write {path} - is it open in another program (e.g. Excel)? skipped")
            continue
        saved += 1
        print(f"  {path}  {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    if not saved:
        raise UserError("Nothing was saved.")
    print(f"Saved {saved} pairs. Analyse them with: python -m fxquant analyze --source csv --csv-dir {out}")


def cmd_bench(args) -> None:
    from . import _reference as py
    from . import core

    if core.BACKEND != "c++":
        print("The C++ core is not built on this computer, so there is nothing to compare: fxquant is using "
              "its NumPy implementation (same results, slower). To enable C++, install a compiler and run "
              "the setup again (see README, 'Install the tools').")
        return
    from . import _fxcore as cpp

    rng = np.random.default_rng(0)
    n = args.bars
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 5e-4, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 5e-4, n)))
    a = cpp.atr(h, l, c, 14)
    sig = rng.choice([-1.0, 0.0, 1.0], n)
    jobs = {
        "atr": lambda m: m.atr(h, l, c, 14),
        "adx": lambda m: m.adx(h, l, c, 14),
        "rsi": lambda m: m.rsi(c, 14),
        "variance_ratio": lambda m: m.variance_ratio(np.diff(np.log(c)), 240, 4),
        "triple_barrier": lambda m: m.triple_barrier(h, l, c, a * 5, 24, 1.0, 1.0),
        "backtest": lambda m: m.backtest(o, h, l, c, a * 5, sig, 1.0, 1.0, 24, 0.005, 1e-4),
    }
    print(f"{'function':16s} {'python (s)':>11s} {'c++ (s)':>9s} {'speed-up':>9s}   ({n:,} bars)")
    for name, job in jobs.items():
        t = time.perf_counter(); job(py); t_py = time.perf_counter() - t
        t = time.perf_counter(); job(cpp); t_cpp = time.perf_counter() - t
        print(f"{name:16s} {t_py:11.3f} {t_cpp:9.4f} {t_py / max(t_cpp, 1e-9):8.0f}x")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fxquant", description="Machine-learning forex analysis and trade advice")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse the market and produce trade ideas")
    a.add_argument("--source", choices=["synthetic", "yahoo", "csv"], default="synthetic")
    a.add_argument("--csv-dir", default="data", help="folder with one <PAIR>.csv per pair (for --source csv)")
    a.add_argument("--csv-timezone", default="UTC",
                   help="time zone of CSV time stamps without one: UTC (default), an offset like UTC+2, "
                        "or a name like Europe/Athens (typical MetaTrader server time)")
    a.add_argument("--pairs", help="comma-separated list, e.g. EURUSD,GBPUSD,USDJPY")
    a.add_argument("--segments", help="comma-separated subset of: major,cross,exotic")
    a.add_argument("--interval", default="1h", help="Yahoo bar size (1h recommended)")
    a.add_argument("--period", default="730d", help="Yahoo history length (max 730d for 1h bars)")
    a.add_argument("--years", type=float, default=2.0, help="synthetic history length")
    a.add_argument("--seed", type=int, default=42, help="synthetic random seed")
    a.add_argument("--momentum", type=float, default=0.06, help="synthetic signal strength (0 = pure noise)")
    a.add_argument("--horizon", type=int, default=None, help="holding horizon in bars (default 24 hourly / 5 daily)")
    a.add_argument("--equity", type=float, default=10_000.0, help="account equity")
    a.add_argument("--account-currency", default="USD")
    a.add_argument("--risk", type=float, default=0.5, help="risk per trade, percent of equity")
    a.add_argument("--fast", action="store_true", help="fewer refits for a quicker run")
    a.add_argument("--out", default="reports", help="output folder")
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("save-data", help="save market data to CSV files (merges with existing files)")
    d.add_argument("--source", choices=["synthetic", "yahoo"], default="yahoo")
    d.add_argument("--pairs", help="comma-separated list, e.g. EURUSD,GBPUSD,USDJPY")
    d.add_argument("--segments", help="comma-separated subset of: major,cross,exotic")
    d.add_argument("--interval", default="1h")
    d.add_argument("--period", default="730d")
    d.add_argument("--years", type=float, default=2.0)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--momentum", type=float, default=0.06)
    d.add_argument("--out", default="data", help="output folder")
    d.set_defaults(func=cmd_save_data)

    b = sub.add_parser("bench", help="compare the C++ core with the NumPy fallback")
    b.add_argument("--bars", type=int, default=50_000)
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except UserError as exc:
        print(f"\n{exc}", file=sys.stderr)
        if "debugpy" in sys.modules:  # VS Code F5: end quietly instead of pausing on an expected error
            return
        sys.exit(1)


if __name__ == "__main__":
    main()

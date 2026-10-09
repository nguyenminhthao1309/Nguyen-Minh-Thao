"""Command line interface: `fxquant analyze`, `fxquant save-data` and `fxquant bench`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import UNIVERSE, Config, SEGMENTS


def _pairs(args) -> list[str]:
    if args.pairs:
        return [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    segments = [s.strip() for s in args.segments.split(",")] if args.segments else list(SEGMENTS)
    return [p for p, spec in UNIVERSE.items() if spec.segment in segments]


def cmd_analyze(args) -> None:
    from .advisor import ForexAdvisor
    from .data import load_data
    from .report import console_summary, save_report

    cfg = Config(pairs=_pairs(args))
    cfg.advisor.account_equity = args.equity
    cfg.advisor.account_currency = args.account_currency.upper()
    cfg.advisor.risk_per_trade = args.risk / 100
    cfg.model.horizon = args.horizon
    if args.fast:
        cfg.model.retrain_every, cfg.model.train_stride = "90D", 8

    print(f"[cli] loading {len(cfg.pairs)} pairs from {args.source} ...")
    data = load_data(args.source, cfg.pairs, csv_dir=args.csv_dir, interval=args.interval, period=args.period,
                     years=args.years, seed=args.seed, momentum=args.momentum)
    if not data:
        raise SystemExit("No data loaded. For --source yahoo check your internet connection; "
                         "for --source csv check the folder and file names (e.g. data/EURUSD.csv).")
    result = ForexAdvisor(cfg).run(data)
    paths = save_report(result, args.out)
    print()
    print(console_summary(result))
    print(f"\nReport: {paths['report']}\nIdeas:  {paths['ideas']}")
    print(f"Latest report is always at {paths['latest']} (in VS Code: open it and press Ctrl+Shift+V)")


def cmd_save_data(args) -> None:
    """Save market data as one CSV per pair, merging with files already there."""
    import pandas as pd

    from .data import load_data, read_csv_ohlc

    pairs = _pairs(args)
    print(f"[cli] downloading {len(pairs)} pairs from {args.source} ...")
    data = load_data(args.source, pairs, interval=args.interval, period=args.period,
                     years=args.years, seed=args.seed, momentum=args.momentum)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pair, df in data.items():
        path = out / f"{pair}.csv"
        if path.exists():  # keep older history so the archive grows beyond the feed's limit
            old = read_csv_ohlc(path)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.rename_axis("timestamp").to_csv(path)
        print(f"  {path}  {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    print(f"Saved {len(data)} pairs. Analyse them with: python -m fxquant analyze --source csv --csv-dir {out}")


def cmd_bench(args) -> None:
    from . import _reference as py
    from . import core

    if core.BACKEND != "c++":
        raise SystemExit("The C++ core is not built, so there is nothing to compare. Install a C++ compiler "
                         "and run the setup again (or `pip install -e .`). See README, 'C++ compiler'.")
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
    a.add_argument("--pairs", help="comma-separated list, e.g. EURUSD,GBPUSD,USDJPY")
    a.add_argument("--segments", help="comma-separated subset of: major,cross,exotic")
    a.add_argument("--interval", default="1h", help="Yahoo bar size (1h recommended)")
    a.add_argument("--period", default="730d", help="Yahoo history length (max 730d for 1h bars)")
    a.add_argument("--years", type=float, default=2.0, help="synthetic history length")
    a.add_argument("--seed", type=int, default=42, help="synthetic random seed")
    a.add_argument("--momentum", type=float, default=0.06, help="synthetic signal strength (0 = pure noise)")
    a.add_argument("--horizon", type=int, default=24, help="holding horizon in bars")
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
    args.func(args)


if __name__ == "__main__":
    main()

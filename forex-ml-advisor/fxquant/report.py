"""Markdown / JSON / CSV reporting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .advisor import AnalysisResult

DISCLAIMER = (
    "> **Not financial advice.** This report is produced by a statistical model for research and "
    "education. Forex is leveraged and most retail accounts lose money. Out-of-sample backtests do not "
    "guarantee future results, model edges in FX are small and unstable, and costs, slippage, swaps and "
    "gaps can differ from the assumptions. Size positions so that you can afford every stop being hit."
)


def _fmt(v, spec: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if spec == "%":
        return f"{100 * v:.1f}%"
    if spec:
        return format(v, spec)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _duration(td: pd.Timedelta) -> str:
    minutes = int(td.total_seconds() // 60)
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}min"


def md_table(df: pd.DataFrame, formats: dict[str, str] | None = None, index: bool = True) -> str:
    """GitHub-flavoured markdown table without third-party dependencies."""
    if df is None or df.empty:
        return "_no data_\n"
    formats = formats or {}
    data = df.reset_index() if index else df
    cols = [str(c) for c in data.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], formats.get(str(c), "")) for c in data.columns) + " |")
    return "\n".join(lines) + "\n"


def _price_fmt(pair: str) -> str:
    return ".3f" if pair.endswith("JPY") else ".5f" if pair[3:] in {"USD", "GBP", "EUR", "CHF", "CAD", "AUD", "NZD"} else ".4f"


def render_markdown(res: AnalysisResult) -> str:
    cfg = res.config
    out: list[str] = []
    add = out.append
    add(f"# Forex ML Market Analysis - {res.data_end:%Y-%m-%d %H:%M} UTC\n")
    add(DISCLAIMER + "\n")
    add(f"Data: **{len(res.pair_table)} pairs**, {res.data_start:%Y-%m-%d} -> {res.data_end:%Y-%m-%d}, "
        f"bar size {_duration(res.bar_size)}, numeric core: **{res.backend}**. "
        f"Account: {cfg.advisor.account_equity:,.0f} {cfg.advisor.account_currency}, "
        f"base risk {cfg.advisor.risk_per_trade:.2%} per idea, max total risk {cfg.advisor.max_total_risk:.1%}.\n")
    for w in res.warnings:
        add(f"> **Warning:** {w}\n")

    # 1. Advice ----------------------------------------------------------
    add("## 1. Trade ideas\n")
    if res.recommendations:
        rows = []
        for r in res.recommendations:
            pf = _price_fmt(r["pair"])
            rows.append({"Pair": r["pair"], "Action": r["action"], "Conviction": r["conviction"],
                         "Entry": format(r["entry"], pf), "Stop": format(r["stop"], pf),
                         "Target": format(r["target"], pf), "Stop (pips)": f"{r['stop_pips']:.0f}",
                         "Risk": f"{r['risk_fraction']:.2%}", "Size (lots)": _fmt(r["lots"], ".2f"),
                         "Exp. R": f"{r['expected_r']:+.2f}", "Best session": r["best_session"]})
        add(md_table(pd.DataFrame(rows), index=False))
        for r in res.recommendations:
            add(f"\n**{r['action']} {r['pair']}** ({r['conviction']} conviction) - {r['holding']}.")
            for line in r["rationale"]:
                add(f"- {line}")
            if r.get("note"):
                add(f"- Note: {r['note']}")
        add("\nExecution: enter near the quoted price, avoid the 21:00-23:00 UTC rollover (wide spreads), "
            "and do not widen stops. Re-run the analysis before acting if prices have moved more than "
            "a quarter of the stop distance.\n")
    else:
        add("**No trade passes every filter today.** Standing aside is a valid position: the model only "
            "recommends trades where its out-of-sample record on that pair is positive and statistically "
            "meaningful and the expected value clears costs.\n")
    if len(res.exposure):
        add("Net currency exposure of the ideas (risk as % of equity, + = long the currency):\n")
        add(md_table(res.exposure.to_frame("net_risk").T, {c: "%" for c in res.exposure.index}, index=False))

    # 2. Sections ---------------------------------------------------------
    add("\n## 2. Market sections\n")
    add("### 2.1 Segments: majors, crosses, exotics\n")
    add(md_table(res.segment_table, {"avg_ann_vol_%": ".1f", "trending_share": "%", "high_vol_share": "%",
                                     "oos_auc": ".3f", "avg_r": "+.3f", "strategy_sharpe": ".2f",
                                     "strategy_max_dd": "%"}))
    if not res.session_summary.empty:
        add("\n### 2.2 Trading sessions\n")
        add("Average absolute move per hourly bar (basis points) by segment and session "
            "(session hours are DST-aware local exchange hours):\n")
        add(md_table(res.session_summary.round(2)))
        best = (res.session_stats[res.session_stats["session"] != "Off-hours"]
                .sort_values("avg_range_pips", ascending=False).groupby("pair").head(1)
                .set_index("pair")[["session", "avg_range_pips", "variance_share"]].sort_index())
        add("\nMost active session per pair (highest average bar range):\n")
        add(md_table(best, {"avg_range_pips": ".1f", "variance_share": "%"}))
        drift = res.session_stats[res.session_stats["drift_t_stat"].abs() >= 2.5]
        if len(drift):
            add("\nSession drifts with |t| >= 2.5 (treat with caution, many are tested):\n")
            add(md_table(drift[["pair", "session", "drift_bps_per_bar", "drift_t_stat"]], index=False,
                         formats={"drift_bps_per_bar": "+.2f", "drift_t_stat": "+.2f"}))
    add("\n### 2.3 Currency strength\n")
    add("Cumulative strength (%) from the least-squares decomposition of all pairs; rank 1 = strongest.\n")
    add(md_table(res.strength_table, {"1D": "+.2f", "1W": "+.2f", "1M": "+.2f", "score": ".2f"}))
    add("\n### 2.4 Pair dashboard: regimes, timeframes and model view\n")
    cols = ["segment", "price", "chg_1d_%", "chg_1w_%", "ann_vol_%", "atr_pips", "vol_regime", "trend_regime",
            "trend_h1", "trend_h4", "trend_d1", "p_up", "p_up_calibrated", "oos_auc", "bt_trades", "bt_avg_r",
            "bt_t_stat", "advice"]
    add(md_table(res.pair_table[cols].sort_values(["segment", "pair"]),
                 {"price": ".5g", "chg_1d_%": "+.2f", "chg_1w_%": "+.2f", "ann_vol_%": ".1f", "atr_pips": ".1f",
                  "p_up": ".3f", "p_up_calibrated": ".3f", "oos_auc": ".3f", "bt_avg_r": "+.3f",
                  "bt_t_stat": "+.2f"}))

    # 3. Model diagnostics -------------------------------------------------
    m = res.model_overall
    add("\n## 3. Model diagnostics (walk-forward, out-of-sample)\n")
    if "auc" in m:
        add(f"- Rows scored: {m['n']:,}; AUC **{m['auc']:.3f}** (0.5 = coin flip); accuracy {m['accuracy']:.1%}; "
            f"hit rate on traded signals {m['signal_hit_rate']:.1%} ({m['signal_share']:.0%} of bars signalled)")
        add(f"- Log loss {m['log_loss']:.4f} vs {m['log_loss_baseline']:.4f} for always predicting the base rate "
            "(raw scores; see the calibrated probability in the dashboard)")
    if res.portfolio:
        p = res.portfolio
        add(f"- Equal-weight strategy across all pairs: return {p['total_return']:+.2%}, Sharpe {p['sharpe']:.2f}, "
            f"max drawdown {p['max_drawdown']:.2%} over {p['days']} days (risk {cfg.strategy.risk_per_trade:.2%}/trade, "
            "spreads + slippage included)")
    add(f"- Final model component weights (validated skill): {res.final_weights}\n")
    for name, title in (("segment", "By segment"), ("session", "By session"), ("vol_regime", "By volatility regime"),
                        ("trend_regime", "By trend regime")):
        t = res.model_by.get(name)
        if t is not None and not t.empty:
            add(f"**{title}**\n")
            add(md_table(t[["n", "auc", "accuracy", "signal_hit_rate"]],
                         {"auc": ".3f", "accuracy": "%", "signal_hit_rate": "%"}))
    add("**Realised expectancy by signal strength** (all pairs, R-multiples):\n")
    add(md_table(res.expectancy, {"gross_r": "+.3f", "net_r": "+.3f", "win_rate": "%"}))
    add("**Calibration of raw scores**\n")
    add(md_table(res.calibration, {"mean_pred": ".3f", "realised_up": ".3f"}))
    if len(res.importance):
        add("**Top features** (drop in OOS AUC when shuffled):\n")
        add(md_table(res.importance.head(15).rename_axis("feature").to_frame("auc_drop"), {"auc_drop": "+.4f"}))
    if not res.folds.empty:
        add("**Walk-forward folds**\n")
        add(md_table(res.folds.assign(test_start=res.folds["test_start"].dt.strftime("%Y-%m-%d"),
                                      test_end=res.folds["test_end"].dt.strftime("%Y-%m-%d")),
                     {"auc": ".3f"}, index=False))

    # 4. Watchlist ---------------------------------------------------------
    add("\n## 4. Watchlist: signals that did not make the cut\n")
    if res.watchlist:
        rows = [{"Pair": w["pair"], "Signal": w["action"], "Strength": f"{w['edge']:.1%}",
                 "Exp. R": _fmt(w["expected_r"], "+.2f"),
                 "Why not": w.get("blocked_by") or "; ".join(w["reasons"])} for w in res.watchlist]
        add(md_table(pd.DataFrame(rows), index=False))
    else:
        add("_none_\n")

    add("\n## 5. Method\n")
    intraday = res.bar_size < pd.Timedelta("1D")
    context = "H4/D1 context, session flags" if intraday else "weekly context"
    add("Features (momentum, RSI, ADX, efficiency ratio, variance ratio, Bollinger, volatility regime, "
        f"{context}, currency-strength ranks) are computed with the {_core_name(res.backend)}. "
        "Labels are triple-barrier "
        f"outcomes (+/-{cfg.model.barrier_mult:g} x ATR x sqrt({cfg.model.horizon}) within {cfg.model.horizon} bars). "
        "A stack of a trend factor, gradient boosting and logistic regression is retrained every "
        f"{cfg.model.retrain_every} with purged walk-forward validation. Every number in sections 2-4 is "
        "out-of-sample. Trade ideas must pass: signal strength, positive and significant pair track record, "
        "positive pair AUC, expected value after costs, and portfolio limits (total risk, per-currency risk, "
        "correlation).\n")
    add("\n_Run time: " + ", ".join(f"{k} {v:.1f}s" for k, v in res.timings.items()) + "_\n")
    return "\n".join(out)


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def _core_name(backend: str) -> str:
    return "C++ core" if backend == "c++" else "NumPy implementation of the core (C++ core not built)"


def _safe_write(path: Path, text: str | None = None, frame: pd.DataFrame | None = None) -> bool:
    """Write a file; if it is locked (e.g. open in Excel on Windows) warn and carry on."""
    try:
        if frame is not None:
            frame.to_csv(path)
        else:
            path.write_text(text, encoding="utf-8")
        return True
    except PermissionError:
        print(f"[report] could not write {path} - is it open in another program (e.g. Excel)? skipped")
        return False


def save_report(res: AnalysisResult, out_dir: str | Path) -> dict[str, Path]:
    """Write the report, trade ideas and tables.

    Time-stamped copies use the run time (local clock), so every run keeps its
    own copy; latest_report.md / latest_trade_ideas.json always hold the newest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    n = 1
    while (out / f"fx_report_{stamp}.md").exists():  # two runs in the same second
        n += 1
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{n}"
    markdown = render_markdown(res)
    ideas = [{k: _jsonable(v) for k, v in r.items()} for r in res.recommendations]
    ideas_json = json.dumps({"generated_at": res.generated_at.isoformat(),
                             "data_end": res.data_end.isoformat(), "ideas": ideas}, indent=2)
    files = {"report": (f"fx_report_{stamp}.md", markdown), "latest": ("latest_report.md", markdown),
             "ideas": (f"trade_ideas_{stamp}.json", ideas_json), "latest_ideas": ("latest_trade_ideas.json", ideas_json)}
    paths: dict[str, Path] = {}
    for key, (name, text) in files.items():
        if _safe_write(out / name, text):
            paths[key] = out / name
    tables = {"pairs": res.pair_table, "segments": res.segment_table, "sessions": res.session_stats,
              "strength": res.strength_table, "folds": res.folds}
    for name, df in tables.items():
        if df is not None and not df.empty and _safe_write(out / f"{name}_{stamp}.csv", frame=df):
            paths[name] = out / f"{name}_{stamp}.csv"
    return paths


def strongest_and_weakest(ranked: list[str], k: int = 3) -> tuple[list[str], list[str]]:
    """Top and bottom of a ranking without overlap (k shrinks for short lists)."""
    k = min(k, len(ranked) // 2)
    return ranked[:k], (ranked[-k:][::-1] if k else [])


def console_summary(res: AnalysisResult) -> str:
    lines = [f"Forex ML analysis ({res.data_end:%Y-%m-%d %H:%M} UTC, core: {res.backend})"]
    m = res.model_overall
    if "auc" in m:
        lines.append(f"  OOS AUC {m['auc']:.3f} | signal hit rate {m['signal_hit_rate']:.1%} | "
                     f"strategy Sharpe {res.portfolio.get('sharpe', float('nan')):.2f}")
    top, bottom = strongest_and_weakest(res.strength_table.index.tolist())
    if top:
        lines.append(f"  Strongest currencies: {', '.join(top)} | weakest: {', '.join(bottom)}")
    for w in res.warnings:
        lines.append(f"  WARNING: {w}")
    if res.recommendations:
        lines.append("  Trade ideas:")
        for r in res.recommendations:
            pf = _price_fmt(r["pair"])
            lines.append(f"    {r['action']:4s} {r['pair']}  entry {r['entry']:{pf}}  stop {r['stop']:{pf}}  "
                         f"target {r['target']:{pf}}  risk {r['risk_fraction']:.2%} ({_fmt(r['lots'], '.2f')} lots)  "
                         f"exp {r['expected_r']:+.2f}R  [{r['conviction']}]")
    else:
        lines.append("  No trade passes every filter today - stand aside.")
    lines.append("  Not financial advice. See the full report for rationale and risks.")
    return "\n".join(lines)

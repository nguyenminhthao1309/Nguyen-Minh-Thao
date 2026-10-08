"""End-to-end pipeline: analyse every market section, then produce trade advice."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import core
from .backtest import backtest_pair, expectancy_by_edge, portfolio_metrics, round_trip_cost
from .config import Config, SEGMENTS, pair_spec
from .features import bars_per_day, build_dataset, infer_bar_delta
from .model import (StackedModel, calibration_table, fit_final, permutation_importance, score, score_by,
                    walk_forward)
from .risk import apply_portfolio_limits, currency_exposure, kelly_fraction, position_size
from .sessions import best_session, hourly_profile, segment_session_summary, session_stats
from .strength import strength_table


@dataclass
class AnalysisResult:
    generated_at: pd.Timestamp
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    bar_size: pd.Timedelta
    backend: str
    config: Config
    pair_table: pd.DataFrame
    segment_table: pd.DataFrame
    session_stats: pd.DataFrame
    session_summary: pd.DataFrame
    hourly_profile: pd.DataFrame
    strength_table: pd.DataFrame
    model_overall: dict
    model_by: dict[str, pd.DataFrame]
    calibration: pd.DataFrame
    importance: pd.Series
    folds: pd.DataFrame
    final_weights: dict
    portfolio: dict
    portfolio_curve: pd.Series
    expectancy: pd.DataFrame
    recommendations: list[dict]
    watchlist: list[dict]
    exposure: pd.Series
    timings: dict = field(default_factory=dict)


def _arrow(x: float) -> str:
    if not np.isfinite(x):
        return "?"
    return "up" if x > 0 else "down"


class ForexAdvisor:
    def __init__(self, config: Config | None = None, log=print):
        self.config = config or Config()
        self.log = log

    # ------------------------------------------------------------------
    def run(self, data: dict[str, pd.DataFrame]) -> AnalysisResult:
        cfg = self.config
        timings: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = now - clock
            clock = now

        data = {p: df for p, df in data.items() if len(df)}
        delta = infer_bar_delta(next(iter(data.values())).index)
        bpd = bars_per_day(delta)
        intraday = delta < pd.Timedelta("1D")
        self.log(f"[advisor] {len(data)} pairs, bar size {delta}, numeric backend: {core.BACKEND}")

        dataset, feature_cols, strength = build_dataset(data, cfg.model)
        lap("features + labels")
        self.log(f"[advisor] dataset: {len(dataset):,} rows x {len(feature_cols)} features")

        wf = walk_forward(dataset, feature_cols, cfg.model, self.log)
        dataset["p_oos"] = wf.oos_proba
        lap("walk-forward training")

        final = fit_final(dataset, feature_cols, cfg.model)
        latest = dataset.groupby("pair").tail(1).set_index("pair")
        latest["p_up"] = final.predict_proba(latest[feature_cols].to_numpy(np.float32))
        latest["p_up_calibrated"] = wf.calibrator.transform(latest["p_up"].to_numpy())
        lap("final model")

        backtests = {}
        for pair, g in dataset.groupby("pair"):
            g = g.set_index("time")
            backtests[pair] = backtest_pair(data[pair], g["atr"], g["p_oos"], pair_spec(pair), cfg.model, cfg.strategy)
        port, curve = portfolio_metrics({p: b["equity"] for p, b in backtests.items()})
        all_trades = pd.concat([b["trades"].assign(pair=p) for p, b in backtests.items() if len(b["trades"])],
                               ignore_index=True) if any(len(b["trades"]) for b in backtests.values()) else pd.DataFrame()
        expectancy = expectancy_by_edge(all_trades)
        lap("backtests")

        oos = dataset[dataset["p_oos"].notna()]
        thr = cfg.strategy.entry_threshold
        overall = score(oos["target"].to_numpy(), oos["p_oos"].to_numpy(), thr)
        model_by = {g: score_by(oos, "p_oos", g, thr) for g in ("segment", "session", "vol_regime", "trend_regime", "pair")}
        calib = calibration_table(oos["target"].to_numpy(), oos["p_oos"].to_numpy())
        importance = pd.Series(dtype=float)
        if wf.last_model is not None and wf.last_test_mask is not None and wf.last_test_mask.any():
            last = dataset[wf.last_test_mask]
            importance = permutation_importance(wf.last_model, last[feature_cols].to_numpy(np.float32),
                                                last["target"].to_numpy(), feature_cols)
        lap("diagnostics")

        sess = session_stats(data) if intraday else pd.DataFrame()
        sess_summary = segment_session_summary(sess) if intraday else pd.DataFrame()
        hourly = hourly_profile(data) if intraday else pd.DataFrame()
        strength_tbl = strength_table(strength, bpd)

        pair_table = self._pair_table(data, latest, backtests, model_by["pair"], bpd)
        segment_table = self._segment_table(pair_table, model_by["segment"], backtests)
        lap("section analytics")

        prices = {p: float(df["close"].iloc[-1]) for p, df in data.items()}
        corr = self._recent_correlation(data)
        ideas, passed_on = [], []
        for pair, row in latest.iterrows():
            idea = self._evaluate(pair, row, backtests[pair]["metrics"], model_by["pair"], expectancy, sess, bpd)
            (ideas if idea["eligible"] else passed_on).append(idea)
        recs, blocked = apply_portfolio_limits(ideas, corr, cfg.advisor)
        recs = [self._finalise(r, prices) for r in recs]
        pair_table["advice"] = "no trade"
        for r in recs:
            pair_table.loc[r["pair"], "advice"] = r["action"]
        watchlist = blocked + sorted([i for i in passed_on if i["edge"] >= cfg.advisor.min_edge / 2],
                                     key=lambda x: -x["edge"])
        lap("advice")

        return AnalysisResult(
            generated_at=pd.Timestamp.now(tz="UTC"), data_start=dataset["time"].min(),
            data_end=dataset["time"].max(), bar_size=delta, backend=core.BACKEND, config=cfg,
            pair_table=pair_table, segment_table=segment_table, session_stats=sess, session_summary=sess_summary,
            hourly_profile=hourly, strength_table=strength_tbl, model_overall=overall, model_by=model_by,
            calibration=calib, importance=importance, folds=wf.folds,
            final_weights={k: round(float(w), 3) for k, w in zip(StackedModel.COMPONENTS, final.weights)},
            portfolio=port, portfolio_curve=curve, expectancy=expectancy, recommendations=recs, watchlist=watchlist,
            exposure=currency_exposure(recs), timings=timings)

    # ------------------------------------------------------------------
    def _pair_table(self, data, latest, backtests, by_pair, bpd) -> pd.DataFrame:
        rows = []
        for pair, row in latest.iterrows():
            spec = pair_spec(pair)
            close = data[pair]["close"]
            r = np.log(close).diff().iloc[-21 * bpd:]
            m = backtests[pair]["metrics"]
            rows.append({
                "pair": pair, "segment": spec.segment, "price": row["close"],
                "chg_1d_%": 100 * (close.iloc[-1] / close.iloc[-1 - bpd] - 1) if len(close) > bpd else np.nan,
                "chg_1w_%": 100 * (close.iloc[-1] / close.iloc[-1 - 5 * bpd] - 1) if len(close) > 5 * bpd else np.nan,
                "ann_vol_%": 100 * r.std() * np.sqrt(260 * bpd),
                "atr_pips": row["atr"] / spec.pip_size,
                "vol_regime": row["vol_regime"], "trend_regime": row["trend_regime"],
                "trend_h1": _arrow(row.get("ema50_slope", np.nan)),
                "trend_h4": _arrow(row.get("h4_ema20_slope", np.nan)),
                "trend_d1": _arrow(row.get("d1_ema20_slope", np.nan)),
                "base_strength": row.get("base_strength_week", np.nan),
                "quote_strength": row.get("quote_strength_week", np.nan),
                "p_up": row["p_up"], "p_up_calibrated": row["p_up_calibrated"],
                "oos_auc": by_pair["auc"].get(pair, np.nan) if not by_pair.empty else np.nan,
                "bt_trades": m.get("trades", 0), "bt_win_rate": m.get("win_rate", np.nan),
                "bt_avg_r": m.get("avg_r", np.nan), "bt_t_stat": m.get("t_stat", np.nan),
                "bt_sharpe": m.get("sharpe", np.nan), "bt_max_dd": m.get("max_drawdown", np.nan),
            })
        return pd.DataFrame(rows).set_index("pair")

    def _segment_table(self, pair_table, by_segment, backtests) -> pd.DataFrame:
        rows = []
        for seg in SEGMENTS:
            pt = pair_table[pair_table["segment"] == seg]
            if pt.empty:
                continue
            port, _ = portfolio_metrics({p: backtests[p]["equity"] for p in pt.index})
            rows.append({
                "segment": seg, "pairs": len(pt), "avg_ann_vol_%": pt["ann_vol_%"].mean(),
                "trending_share": pt["trend_regime"].isin(["uptrend", "downtrend"]).mean(),
                "high_vol_share": (pt["vol_regime"] == "high vol").mean(),
                "oos_auc": by_segment["auc"].get(seg, np.nan) if not by_segment.empty else np.nan,
                "trades": int(pt["bt_trades"].sum()), "avg_r": pt["bt_avg_r"].mean(),
                "strategy_sharpe": port.get("sharpe", np.nan), "strategy_max_dd": port.get("max_drawdown", np.nan),
            })
        return pd.DataFrame(rows).set_index("segment") if rows else pd.DataFrame()

    @staticmethod
    def _recent_correlation(data, days: int = 60) -> pd.DataFrame:
        closes = pd.DataFrame({p: df["close"] for p, df in data.items()})
        daily = closes.resample("24h", origin="epoch", offset="22h").last().dropna(how="all")
        return np.log(daily).diff().iloc[-days:].corr()

    # ------------------------------------------------------------------
    @staticmethod
    def _expected_gross_r(edge: float, expectancy: pd.DataFrame, min_trades: int = 30) -> float:
        """Average realised gross R of past OOS trades whose signal strength matched `edge`."""
        if expectancy.empty:
            return np.nan
        for interval, row in expectancy.iloc[::-1].iterrows():
            if edge >= interval.left and row["trades"] >= min_trades:
                return float(row["gross_r"])
        return np.nan

    def _evaluate(self, pair, row, m, by_pair, expectancy, sess, bpd) -> dict:
        cfg = self.config
        spec = pair_spec(pair)
        horizon, mult = cfg.model.horizon, cfg.model.barrier_mult
        p = float(row["p_up"])
        d = 1 if p >= 0.5 else -1
        p_dir = max(p, 1 - p)
        edge = p_dir - 0.5

        stop_dist = mult * float(row["atr"]) * np.sqrt(horizon)
        entry = float(row["close"])
        cost_r = round_trip_cost(spec, cfg.strategy.slippage_pips) / stop_dist if stop_dist > 0 else np.inf
        # Empirical expectancy: what past out-of-sample signals of this strength
        # actually earned before costs, minus this pair's own cost today.
        exp_r = self._expected_gross_r(edge, expectancy) - cost_r
        p_cal = float(row.get("p_up_calibrated", np.nan))

        trades, avg_r, t_stat = m.get("trades", 0), m.get("avg_r", np.nan), m.get("t_stat", np.nan)
        pair_auc = by_pair["auc"].get(pair, np.nan) if not by_pair.empty else np.nan
        reasons = []
        if edge < cfg.advisor.min_edge:
            reasons.append(f"model edge {edge:.1%} is below the {cfg.advisor.min_edge:.0%} threshold")
        if trades < cfg.advisor.min_oos_trades:
            reasons.append(f"only {trades} out-of-sample trades (need {cfg.advisor.min_oos_trades})")
        if not (np.isfinite(avg_r) and avg_r > 0):
            reasons.append(f"strategy did not make money out-of-sample on this pair (avg {avg_r:+.2f}R)")
        elif not (np.isfinite(t_stat) and t_stat >= 1.0):
            reasons.append(f"out-of-sample result not statistically meaningful (t = {t_stat:.2f})")
        if not (np.isfinite(pair_auc) and pair_auc > 0.5):
            reasons.append(f"model not predictive out-of-sample on this pair (AUC {pair_auc:.3f})")
        if not np.isfinite(exp_r):
            reasons.append("no out-of-sample trades with a comparable signal strength")
        elif exp_r < cfg.advisor.min_expected_r:
            reasons.append(f"expected value {exp_r:+.2f}R after costs is below {cfg.advisor.min_expected_r:.2f}R")

        strength_spread = float(row.get("strength_spread_week", np.nan))
        strength_align = np.clip(2 * strength_spread * d, -1, 1) if np.isfinite(strength_spread) else 0.0
        slopes = [row.get(k, np.nan) for k in ("ema50_slope", "h4_ema20_slope", "d1_ema20_slope")]
        tf_agree = sum(1 for s in slopes if np.isfinite(s) and np.sign(s) == d)
        align = 0.5 * strength_align + 0.5 * (2 * tf_agree / 3 - 1)

        t_eff = t_stat if np.isfinite(t_stat) else 0.0
        if exp_r >= 0.15 and t_eff >= 2 and align >= 0.33:
            conviction = "High"
        elif exp_r >= 0.07 and t_eff >= 1.5:
            conviction = "Medium"
        else:
            conviction = "Low"
        kelly = kelly_fraction(m.get("win_rate", np.nan), m.get("avg_win_r", np.nan), m.get("avg_loss_r", np.nan))
        risk = cfg.advisor.risk_per_trade * {"High": 1.0, "Medium": 0.75, "Low": 0.5}[conviction]
        if kelly > 0:
            risk = min(risk, cfg.advisor.kelly_fraction * kelly)
        if row["vol_regime"] == "high vol":
            risk *= 0.5

        session = best_session(sess, pair) if not sess.empty else "n/a"
        base_rank, quote_rank = row.get("base_strength_week", np.nan), row.get("quote_strength_week", np.nan)
        rationale = [
            f"Model score P(up) = {p:.1%} -> {('long' if d > 0 else 'short')} signal, strength {edge:.1%} "
            f"(calibrated probability {p_cal:.1%})",
            f"Out-of-sample on {pair}: AUC {pair_auc:.3f}, {trades} trades, avg {avg_r:+.2f}R (t = {t_stat:.2f}), "
            f"win rate {m.get('win_rate', np.nan):.0%}",
            f"Expected value {exp_r:+.2f}R per trade (past OOS signals of this strength, after {cost_r:.2f}R costs)",
            f"Regime: {row['trend_regime']}, {row['vol_regime']}",
            f"Timeframes agreeing with the trade: {tf_agree}/3 (H1, H4, D1)",
        ]
        if np.isfinite(base_rank) and np.isfinite(quote_rank):
            rationale.append(f"Weekly strength percentile: {spec.base} {base_rank:.0%} vs {spec.quote} {quote_rank:.0%} "
                             f"({'supports' if strength_align > 0.2 else 'contradicts' if strength_align < -0.2 else 'neutral to'} the trade)")
        return {
            "pair": pair, "segment": spec.segment, "direction": d, "action": "BUY" if d > 0 else "SELL",
            "p_up": p, "p_up_calibrated": p_cal, "edge": edge, "expected_r": exp_r, "cost_r": cost_r, "conviction": conviction,
            "alignment": align, "risk_fraction": risk, "kelly": kelly,
            "score": exp_r * (0.5 + 0.5 * np.clip(t_eff / 3, 0, 1)) * (1 + 0.25 * align),
            "entry": entry, "stop": entry - d * stop_dist, "target": entry + d * stop_dist,
            "holding": f"up to {horizon} bars (~{horizon / bpd:.1f} days); exit at stop/target or if the signal flips",
            "best_session": session, "rationale": rationale, "eligible": not reasons, "reasons": reasons,
        }

    def _finalise(self, idea: dict, prices: dict) -> dict:
        a = self.config.advisor
        size = position_size(a.account_equity, idea["risk_fraction"], idea["entry"], idea["stop"],
                             pair_spec(idea["pair"]), prices, a.account_currency)
        return {**idea, **size}

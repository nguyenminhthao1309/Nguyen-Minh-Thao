"""Market regime classification (causal, rule-based on rolling statistics).

Volatility regime: the current fast volatility's percentile within its own
trailing history (low < 33% < normal < 67% < high).
Trend regime: Kaufman efficiency ratio and ADX decide trending vs ranging;
the slope of the 50-bar EMA gives the direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ER_TREND = 0.20   # a random walk scores ~1/sqrt(n) ~ 0.09 over 120 bars
ADX_TREND = 0.25  # ADX 25, rescaled to [0, 1]


def classify_regimes(features: pd.DataFrame) -> pd.DataFrame:
    vol_pct = features["vol_pct"].to_numpy()
    vol = np.select([vol_pct < 1 / 3, vol_pct > 2 / 3], ["low vol", "high vol"], "normal vol")
    vol = np.where(np.isnan(vol_pct), "unknown", vol)

    er, adx, slope = (features[k].to_numpy() for k in ("er_slow", "adx_14", "ema50_slope"))
    trending = (er >= ER_TREND) | (adx >= ADX_TREND + 0.05)
    trend = np.where(trending, np.where(slope >= 0, "uptrend", "downtrend"), "range")
    trend = np.where(np.isnan(er) | np.isnan(adx), "unknown", trend)
    return pd.DataFrame({"vol_regime": vol, "trend_regime": trend}, index=features.index)

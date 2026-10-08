"""Numeric core: the compiled C++ extension when available, NumPy otherwise.

Set the environment variable FXQUANT_BACKEND=python to force the fallback.
"""

from __future__ import annotations

import os

from . import _reference

BACKEND = "python"
_impl = _reference

if os.environ.get("FXQUANT_BACKEND", "").lower() != "python":
    try:
        from . import _fxcore as _impl  # type: ignore[no-redef]

        BACKEND = "c++"
    except ImportError:  # extension not built: pip install -e .
        pass

ema = _impl.ema
sma = _impl.sma
rolling_std = _impl.rolling_std
rolling_zscore = _impl.rolling_zscore
rolling_percentile_rank = _impl.rolling_percentile_rank
rsi = _impl.rsi
true_range = _impl.true_range
atr = _impl.atr
adx = _impl.adx
stochastic_k = _impl.stochastic_k
efficiency_ratio = _impl.efficiency_ratio
variance_ratio = _impl.variance_ratio
bollinger_pctb = _impl.bollinger_pctb
triple_barrier = _impl.triple_barrier
backtest = _impl.backtest

__all__ = [
    "BACKEND", "ema", "sma", "rolling_std", "rolling_zscore", "rolling_percentile_rank", "rsi",
    "true_range", "atr", "adx", "stochastic_k", "efficiency_ratio", "variance_ratio",
    "bollinger_pctb", "triple_barrier", "backtest",
]

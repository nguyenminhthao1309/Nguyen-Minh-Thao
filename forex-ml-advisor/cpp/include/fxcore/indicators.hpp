// Technical indicators used as ML features.
//
// Every function is strictly causal: out[i] depends only on inputs[0..i].
// Warm-up positions are NaN. The Python fallback in fxquant/_reference.py
// implements the exact same algorithms, and tests/test_core_parity.py checks
// that both backends agree.
#pragma once

#include <vector>

namespace fxcore {

using Vec = std::vector<double>;

// Exponential moving average, alpha = 2 / (span + 1), seeded with the first
// finite value (pandas ewm(span, adjust=False) semantics).
Vec ema(const Vec& x, int span);

// Simple moving average over `window` values.
Vec sma(const Vec& x, int window);

// Rolling standard deviation (two-pass per window, numerically exact on
// price levels). ddof = 1 gives the sample std, ddof = 0 the population std.
Vec rolling_std(const Vec& x, int window, int ddof);

// (x - rolling mean) / rolling std (ddof = 1). 0 when the std is zero.
Vec rolling_zscore(const Vec& x, int window);

// Fraction of the trailing `window` values (current one included) that are
// <= the current value. Used for causal volatility regime percentiles.
Vec rolling_percentile_rank(const Vec& x, int window);

// Wilder's RSI.
Vec rsi(const Vec& close, int period);

// True range (first bar uses high - low).
Vec true_range(const Vec& high, const Vec& low, const Vec& close);

// Wilder's Average True Range.
Vec atr(const Vec& high, const Vec& low, const Vec& close, int period);

// Wilder's Average Directional Index.
Vec adx(const Vec& high, const Vec& low, const Vec& close, int period);

// Stochastic %K in [0, 100].
Vec stochastic_k(const Vec& high, const Vec& low, const Vec& close, int period);

// Kaufman efficiency ratio: |net move| / path length over `period` bars.
// 1 = perfectly trending, 0 = pure noise.
Vec efficiency_ratio(const Vec& close, int period);

// Lo-MacKinlay variance ratio of returns over a rolling window.
// VR > 1 suggests trending (positive autocorrelation), VR < 1 mean reversion.
Vec variance_ratio(const Vec& returns, int window, int q);

// Bollinger %B: position of close inside the (window, nstd) bands.
Vec bollinger_pctb(const Vec& close, int window, double nstd);

}  // namespace fxcore

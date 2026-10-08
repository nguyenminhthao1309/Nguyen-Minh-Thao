#include "fxcore/indicators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fxcore {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void require_positive(int value, const char* name) {
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
}

void require_same_size(const Vec& a, const Vec& b, const Vec& c) {
    if (a.size() != b.size() || a.size() != c.size())
        throw std::invalid_argument("high, low and close must have the same length");
}

double window_mean(const Vec& x, std::size_t end, int window) {
    double s = 0.0;
    for (std::size_t j = end + 1 - window; j <= end; ++j) s += x[j];
    return s / window;
}

double window_std(const Vec& x, std::size_t end, int window, int ddof) {
    const double m = window_mean(x, end, window);
    double ss = 0.0;
    for (std::size_t j = end + 1 - window; j <= end; ++j) {
        const double d = x[j] - m;
        ss += d * d;
    }
    return std::sqrt(ss / (window - ddof));
}

double rsi_value(double avg_gain, double avg_loss) {
    if (avg_loss == 0.0) return avg_gain == 0.0 ? 50.0 : 100.0;
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
}

}  // namespace

Vec ema(const Vec& x, int span) {
    require_positive(span, "span");
    const std::size_t n = x.size();
    Vec out(n, kNaN);
    const double alpha = 2.0 / (span + 1.0);
    std::size_t s = 0;
    while (s < n && !std::isfinite(x[s])) ++s;
    if (s == n) return out;
    out[s] = x[s];
    for (std::size_t i = s + 1; i < n; ++i)
        out[i] = std::isfinite(x[i]) ? alpha * x[i] + (1.0 - alpha) * out[i - 1] : out[i - 1];
    return out;
}

Vec sma(const Vec& x, int window) {
    require_positive(window, "window");
    const std::size_t n = x.size();
    Vec out(n, kNaN);
    for (std::size_t i = window - 1; i < n; ++i) out[i] = window_mean(x, i, window);
    return out;
}

Vec rolling_std(const Vec& x, int window, int ddof) {
    require_positive(window, "window");
    if (window - ddof <= 0) throw std::invalid_argument("window must exceed ddof");
    const std::size_t n = x.size();
    Vec out(n, kNaN);
    for (std::size_t i = window - 1; i < n; ++i) out[i] = window_std(x, i, window, ddof);
    return out;
}

Vec rolling_zscore(const Vec& x, int window) {
    if (window < 2) throw std::invalid_argument("window must be >= 2");
    const std::size_t n = x.size();
    Vec out(n, kNaN);
    for (std::size_t i = window - 1; i < n; ++i) {
        const double sd = window_std(x, i, window, 1);
        if (!std::isfinite(sd)) continue;
        out[i] = sd > 0.0 ? (x[i] - window_mean(x, i, window)) / sd : 0.0;
    }
    return out;
}

Vec rolling_percentile_rank(const Vec& x, int window) {
    require_positive(window, "window");
    const std::size_t n = x.size();
    Vec out(n, kNaN);
    for (std::size_t i = window - 1; i < n; ++i) {
        if (!std::isfinite(x[i])) continue;
        int below = 0, valid = 0;
        for (std::size_t j = i + 1 - window; j <= i; ++j) {
            if (!std::isfinite(x[j])) continue;
            ++valid;
            if (x[j] <= x[i]) ++below;
        }
        out[i] = static_cast<double>(below) / valid;
    }
    return out;
}

Vec rsi(const Vec& close, int period) {
    require_positive(period, "period");
    const std::size_t n = close.size();
    Vec out(n, kNaN);
    if (n <= static_cast<std::size_t>(period)) return out;
    double gain = 0.0, loss = 0.0;
    for (int i = 1; i <= period; ++i) {
        const double d = close[i] - close[i - 1];
        gain += std::max(d, 0.0);
        loss += std::max(-d, 0.0);
    }
    double avg_gain = gain / period, avg_loss = loss / period;
    out[period] = rsi_value(avg_gain, avg_loss);
    for (std::size_t i = period + 1; i < n; ++i) {
        const double d = close[i] - close[i - 1];
        avg_gain = (avg_gain * (period - 1) + std::max(d, 0.0)) / period;
        avg_loss = (avg_loss * (period - 1) + std::max(-d, 0.0)) / period;
        out[i] = rsi_value(avg_gain, avg_loss);
    }
    return out;
}

Vec true_range(const Vec& high, const Vec& low, const Vec& close) {
    require_same_size(high, low, close);
    const std::size_t n = close.size();
    Vec tr(n, kNaN);
    if (n == 0) return tr;
    tr[0] = high[0] - low[0];
    for (std::size_t i = 1; i < n; ++i) {
        tr[i] = std::max({high[i] - low[i], std::fabs(high[i] - close[i - 1]),
                          std::fabs(low[i] - close[i - 1])});
    }
    return tr;
}

Vec atr(const Vec& high, const Vec& low, const Vec& close, int period) {
    require_positive(period, "period");
    const Vec tr = true_range(high, low, close);
    const std::size_t n = tr.size();
    Vec out(n, kNaN);
    if (n < static_cast<std::size_t>(period)) return out;
    out[period - 1] = window_mean(tr, period - 1, period);
    for (std::size_t i = period; i < n; ++i) out[i] = (out[i - 1] * (period - 1) + tr[i]) / period;
    return out;
}

Vec adx(const Vec& high, const Vec& low, const Vec& close, int period) {
    require_positive(period, "period");
    const Vec tr = true_range(high, low, close);
    const std::size_t n = tr.size();
    Vec out(n, kNaN);
    if (n < static_cast<std::size_t>(2 * period)) return out;

    Vec pdm(n, 0.0), mdm(n, 0.0);
    for (std::size_t i = 1; i < n; ++i) {
        const double up = high[i] - high[i - 1];
        const double down = low[i - 1] - low[i];
        pdm[i] = (up > down && up > 0.0) ? up : 0.0;
        mdm[i] = (down > up && down > 0.0) ? down : 0.0;
    }

    double s_tr = 0.0, s_pdm = 0.0, s_mdm = 0.0;
    for (int i = 1; i <= period; ++i) {
        s_tr += tr[i];
        s_pdm += pdm[i];
        s_mdm += mdm[i];
    }
    auto dx = [](double str, double spdm, double smdm) {
        if (str <= 0.0) return 0.0;
        const double pdi = 100.0 * spdm / str;
        const double mdi = 100.0 * smdm / str;
        const double denom = pdi + mdi;
        return denom == 0.0 ? 0.0 : 100.0 * std::fabs(pdi - mdi) / denom;
    };

    Vec dxv(n, kNaN);
    dxv[period] = dx(s_tr, s_pdm, s_mdm);
    for (std::size_t i = period + 1; i < n; ++i) {
        s_tr = s_tr - s_tr / period + tr[i];
        s_pdm = s_pdm - s_pdm / period + pdm[i];
        s_mdm = s_mdm - s_mdm / period + mdm[i];
        dxv[i] = dx(s_tr, s_pdm, s_mdm);
    }

    double first = 0.0;
    for (int i = period; i < 2 * period; ++i) first += dxv[i];
    out[2 * period - 1] = first / period;
    for (std::size_t i = 2 * period; i < n; ++i) out[i] = (out[i - 1] * (period - 1) + dxv[i]) / period;
    return out;
}

Vec stochastic_k(const Vec& high, const Vec& low, const Vec& close, int period) {
    require_positive(period, "period");
    require_same_size(high, low, close);
    const std::size_t n = close.size();
    Vec out(n, kNaN);
    for (std::size_t i = period - 1; i < n; ++i) {
        double hh = high[i], ll = low[i];
        for (std::size_t j = i + 1 - period; j < i; ++j) {
            hh = std::max(hh, high[j]);
            ll = std::min(ll, low[j]);
        }
        const double range = hh - ll;
        out[i] = range > 0.0 ? 100.0 * (close[i] - ll) / range : 50.0;
    }
    return out;
}

Vec efficiency_ratio(const Vec& close, int period) {
    require_positive(period, "period");
    const std::size_t n = close.size();
    Vec out(n, kNaN);
    for (std::size_t i = period; i < n; ++i) {
        double path = 0.0;
        for (std::size_t j = i + 1 - period; j <= i; ++j) path += std::fabs(close[j] - close[j - 1]);
        out[i] = path > 0.0 ? std::fabs(close[i] - close[i - period]) / path : 0.0;
    }
    return out;
}

Vec variance_ratio(const Vec& returns, int window, int q) {
    require_positive(q, "q");
    if (window < q + 2) throw std::invalid_argument("window must be >= q + 2");
    const std::size_t n = returns.size();
    Vec out(n, kNaN);
    Vec qsum(n, kNaN);
    for (std::size_t i = q - 1; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = i + 1 - q; j <= i; ++j) s += returns[j];
        qsum[i] = s;
    }
    const int m = window - q + 1;  // number of overlapping q-sums in the window
    for (std::size_t i = window - 1; i < n; ++i) {
        const double var1 = window_std(returns, i, window, 1);
        const double varq = window_std(qsum, i, m, 1);
        if (!std::isfinite(var1) || !std::isfinite(varq) || var1 == 0.0) continue;
        out[i] = (varq * varq) / (q * var1 * var1);
    }
    return out;
}

Vec bollinger_pctb(const Vec& close, int window, double nstd) {
    require_positive(window, "window");
    const std::size_t n = close.size();
    Vec out(n, kNaN);
    for (std::size_t i = window - 1; i < n; ++i) {
        const double sd = window_std(close, i, window, 0);
        if (sd == 0.0) {
            out[i] = 0.5;
            continue;
        }
        const double lower = window_mean(close, i, window) - nstd * sd;
        out[i] = (close[i] - lower) / (2.0 * nstd * sd);
    }
    return out;
}

}  // namespace fxcore

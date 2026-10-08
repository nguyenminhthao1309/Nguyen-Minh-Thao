#include "fxcore/backtest.hpp"

#include <cmath>
#include <stdexcept>

namespace fxcore {

namespace {

int sign_of(double x) {
    if (!std::isfinite(x)) return 0;
    return (x > 0.0) - (x < 0.0);
}

}  // namespace

BacktestResult backtest(const std::vector<double>& open, const std::vector<double>& high,
                        const std::vector<double>& low, const std::vector<double>& close,
                        const std::vector<double>& atr, const std::vector<double>& signal,
                        const BacktestParams& p) {
    const std::size_t n = close.size();
    if (open.size() != n || high.size() != n || low.size() != n || atr.size() != n || signal.size() != n)
        throw std::invalid_argument("all inputs must have the same length");
    if (p.sl_mult <= 0.0 || p.tp_mult <= 0.0 || p.max_hold <= 0 || p.risk_fraction <= 0.0 || p.cost < 0.0)
        throw std::invalid_argument("invalid backtest parameters");

    BacktestResult res{std::vector<double>(n, 1.0), std::vector<double>(n, 0.0), {}};
    const double half_cost = 0.5 * p.cost;

    double equity_closed = 1.0, equity_at_entry = 1.0;
    int dir = 0;
    std::size_t entry_idx = 0;
    double entry_eff = 0.0, risk_dist = 0.0, stop = 0.0, target = 0.0;

    for (std::size_t t = 0; t < n; ++t) {
        if (dir == 0 && t > 0) {
            const int s = sign_of(signal[t - 1]);
            if (s != 0 && std::isfinite(atr[t - 1]) && atr[t - 1] > 0.0) {
                dir = s;
                entry_idx = t;
                equity_at_entry = equity_closed;
                risk_dist = p.sl_mult * atr[t - 1];
                stop = open[t] - dir * risk_dist;
                target = open[t] + dir * p.tp_mult * atr[t - 1];
                entry_eff = open[t] + dir * half_cost;
            }
        }

        if (dir != 0) {
            double exit_px = 0.0;
            int reason = 0;
            if (t > entry_idx) {  // gaps through a level fill at the open
                if (dir * (open[t] - stop) <= 0.0) {
                    exit_px = open[t];
                    reason = kStop;
                } else if (dir * (open[t] - target) >= 0.0) {
                    exit_px = open[t];
                    reason = kTarget;
                }
            }
            if (reason == 0) {
                const bool hit_stop = dir > 0 ? low[t] <= stop : high[t] >= stop;
                const bool hit_target = dir > 0 ? high[t] >= target : low[t] <= target;
                if (hit_stop) {
                    exit_px = stop;
                    reason = kStop;
                } else if (hit_target) {
                    exit_px = target;
                    reason = kTarget;
                } else if (t - entry_idx + 1 >= static_cast<std::size_t>(p.max_hold)) {
                    exit_px = close[t];
                    reason = kTime;
                } else if (sign_of(signal[t]) == -dir) {
                    exit_px = close[t];
                    reason = kReversal;
                } else if (t == n - 1) {
                    exit_px = close[t];
                    reason = kEndOfData;
                }
            }
            if (reason != 0) {
                const double exit_eff = exit_px - dir * half_cost;
                const double r = dir * (exit_eff - entry_eff) / risk_dist;
                equity_closed = equity_at_entry * (1.0 + p.risk_fraction * r);
                res.trades.push_back(Trade{static_cast<int>(entry_idx), static_cast<int>(t), dir, entry_eff,
                                           exit_eff, r, reason});
                dir = 0;
            }
        }

        if (dir != 0) {
            const double mark = close[t] - dir * half_cost;
            res.equity[t] = equity_at_entry * (1.0 + p.risk_fraction * dir * (mark - entry_eff) / risk_dist);
        } else {
            res.equity[t] = equity_closed;
        }
        res.position[t] = dir;
    }
    return res;
}

}  // namespace fxcore

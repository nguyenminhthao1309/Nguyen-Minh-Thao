#include "fxcore/labeling.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace fxcore {

BarrierLabels triple_barrier(const std::vector<double>& high, const std::vector<double>& low,
                             const std::vector<double>& close, const std::vector<double>& atr,
                             int horizon, double tp_mult, double sl_mult) {
    const std::size_t n = close.size();
    if (high.size() != n || low.size() != n || atr.size() != n)
        throw std::invalid_argument("all inputs must have the same length");
    if (horizon <= 0 || tp_mult <= 0.0 || sl_mult <= 0.0)
        throw std::invalid_argument("horizon and barrier multiples must be positive");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    BarrierLabels out{std::vector<double>(n, nan), std::vector<double>(n, nan),
                      std::vector<double>(n, nan), std::vector<double>(n, nan)};

    for (std::size_t i = 0; i + horizon < n; ++i) {
        if (!std::isfinite(atr[i]) || atr[i] <= 0.0) continue;
        const double c0 = close[i];
        const double upper = c0 + tp_mult * atr[i];
        const double lower = c0 - sl_mult * atr[i];

        double label = 0.0;
        double exit_px = close[i + horizon];
        std::size_t end = i + horizon;
        for (std::size_t j = i + 1; j <= i + horizon; ++j) {
            const bool hit_up = high[j] >= upper;
            const bool hit_dn = low[j] <= lower;
            if (hit_up && hit_dn) {  // order inside the bar is unknown
                exit_px = close[j];
                end = j;
                break;
            }
            if (hit_up) {
                label = 1.0;
                exit_px = upper;
                end = j;
                break;
            }
            if (hit_dn) {
                label = -1.0;
                exit_px = lower;
                end = j;
                break;
            }
        }
        out.label[i] = label;
        out.ret[i] = std::log(exit_px / c0);
        out.hold[i] = static_cast<double>(end - i);
        out.end_idx[i] = static_cast<double>(end);
    }
    return out;
}

}  // namespace fxcore

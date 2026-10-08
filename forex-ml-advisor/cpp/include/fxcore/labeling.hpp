// Triple-barrier labeling (Lopez de Prado, "Advances in Financial ML", ch. 3).
//
// For each bar i, an upper barrier close[i] + tp_mult * atr[i], a lower
// barrier close[i] - sl_mult * atr[i] and a time barrier i + horizon are set.
// The label records which barrier the future path touches first, using the
// bar highs and lows. Bars without a full future window are invalid (NaN),
// so the training set never contains partially observed outcomes.
#pragma once

#include <vector>

namespace fxcore {

struct BarrierLabels {
    std::vector<double> label;    // +1 upper first, -1 lower first, 0 time barrier or both in one bar, NaN invalid
    std::vector<double> ret;      // log return realised at the barrier, NaN invalid
    std::vector<double> hold;     // bars until the barrier was touched, NaN invalid
    std::vector<double> end_idx;  // index of the bar that resolved the label, NaN invalid
};

BarrierLabels triple_barrier(const std::vector<double>& high, const std::vector<double>& low,
                             const std::vector<double>& close, const std::vector<double>& atr,
                             int horizon, double tp_mult, double sl_mult);

}  // namespace fxcore

// Bar-by-bar backtester with ATR stops/targets and transaction costs.
//
// Execution model (no look-ahead):
//   * signal[t] is decided at the close of bar t and executed at open[t + 1].
//   * Stop and target are set from atr[t] (known at decision time).
//   * Inside a bar, if both stop and target are touched the stop is assumed
//     to be hit first (conservative). Gaps through a level fill at the open.
//   * Positions close at the stop, the target, after max_hold bars, when the
//     signal flips to the opposite side, or at the end of the data.
//   * Half of the round-trip cost is paid on entry and half on exit.
// P&L is tracked in R-multiples (profit / initial stop distance) and equity
// compounds as equity *= 1 + risk_fraction * R, which makes results
// independent of pair price scale and account currency.
#pragma once

#include <vector>

namespace fxcore {

struct BacktestParams {
    double sl_mult = 1.5;
    double tp_mult = 1.5;
    int max_hold = 24;
    double risk_fraction = 0.005;
    double cost = 0.0;  // round-trip cost in price units (spread + slippage)
};

enum ExitReason : int { kStop = 1, kTarget = 2, kTime = 3, kReversal = 4, kEndOfData = 5 };

struct Trade {
    int entry_idx;
    int exit_idx;
    int direction;
    double entry_px;  // effective price including half the cost
    double exit_px;   // effective price including half the cost
    double r_multiple;
    int reason;
};

struct BacktestResult {
    std::vector<double> equity;    // mark-to-market equity at each close, starts at 1.0
    std::vector<double> position;  // position held at each close: -1, 0, +1
    std::vector<Trade> trades;
};

BacktestResult backtest(const std::vector<double>& open, const std::vector<double>& high,
                        const std::vector<double>& low, const std::vector<double>& close,
                        const std::vector<double>& atr, const std::vector<double>& signal,
                        const BacktestParams& params);

}  // namespace fxcore

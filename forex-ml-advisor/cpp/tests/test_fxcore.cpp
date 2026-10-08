// Standalone C++ unit tests (no third-party framework). Run via CTest.
#include <cmath>
#include <cstdio>
#include <vector>

#include "fxcore/backtest.hpp"
#include "fxcore/indicators.hpp"
#include "fxcore/labeling.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

void test_ema_sma() {
    const fxcore::Vec x{1, 2, 3, 4, 5};
    const auto e = fxcore::ema(x, 3);  // alpha = 0.5
    check(near(e[0], 1.0) && near(e[1], 1.5) && near(e[2], 2.25), "ema recursion");
    const auto s = fxcore::sma(x, 3);
    check(std::isnan(s[1]) && near(s[2], 2.0) && near(s[4], 4.0), "sma window");
}

void test_rsi_extremes() {
    fxcore::Vec up(30), down(30);
    for (int i = 0; i < 30; ++i) {
        up[i] = 1.0 + 0.01 * i;
        down[i] = 2.0 - 0.01 * i;
    }
    check(near(fxcore::rsi(up, 14).back(), 100.0), "rsi of a rising series is 100");
    check(near(fxcore::rsi(down, 14).back(), 0.0), "rsi of a falling series is 0");
}

void test_atr_constant_range() {
    const fxcore::Vec h(50, 1.2), l(50, 1.1), c(50, 1.15);
    const auto a = fxcore::atr(h, l, c, 14);
    check(std::isnan(a[12]) && near(a[13], 0.1) && near(a[49], 0.1), "atr of a constant range");
}

void test_efficiency_ratio() {
    fxcore::Vec trend(40), zigzag(40);
    for (int i = 0; i < 40; ++i) {
        trend[i] = 1.0 + 0.001 * i;
        zigzag[i] = 1.0 + ((i % 2) ? 0.001 : 0.0);
    }
    check(near(fxcore::efficiency_ratio(trend, 10).back(), 1.0), "efficiency ratio of a straight line is 1");
    check(near(fxcore::efficiency_ratio(zigzag, 10).back(), 0.0), "efficiency ratio of a zigzag is 0");
}

void test_triple_barrier() {
    // Price rises steadily: the upper barrier must be hit first.
    const int n = 30;
    fxcore::Vec c(n), h(n), l(n), a(n, 0.01);
    for (int i = 0; i < n; ++i) {
        c[i] = 1.0 + 0.005 * i;
        h[i] = c[i] + 0.001;
        l[i] = c[i] - 0.001;
    }
    const auto r = fxcore::triple_barrier(h, l, c, a, 10, 1.0, 1.0);
    check(near(r.label[0], 1.0), "rising path labels +1");
    check(near(r.hold[0], 2.0), "barrier reached after two bars");
    check(std::isnan(r.label[n - 1]) && std::isnan(r.label[n - 10]), "incomplete windows are invalid");
}

void test_backtest_stop_and_target() {
    // Bar 0 signals long; entry at open[1] = 1.0, atr = 0.01, stop 0.99, target 1.01.
    const fxcore::Vec o{1.0, 1.0, 1.0, 1.0};
    const fxcore::Vec h{1.0, 1.005, 1.02, 1.0};
    const fxcore::Vec l{1.0, 0.995, 0.999, 1.0};
    const fxcore::Vec c{1.0, 1.0, 1.015, 1.0};
    const fxcore::Vec a(4, 0.01), s{1, 0, 0, 0};
    fxcore::BacktestParams p;
    p.sl_mult = 1.0;
    p.tp_mult = 1.0;
    p.max_hold = 10;
    p.risk_fraction = 0.01;
    p.cost = 0.0;
    const auto r = fxcore::backtest(o, h, l, c, a, s, p);
    check(r.trades.size() == 1, "one trade");
    if (!r.trades.empty()) {
        check(r.trades[0].reason == fxcore::kTarget, "target exit");
        check(near(r.trades[0].r_multiple, 1.0), "+1R");
        check(near(r.equity.back(), 1.01), "equity grows by risk_fraction * R");
    }

    // Same path, but a cost of 0.002 makes the trade worth 0.8R.
    p.cost = 0.002;
    const auto rc = fxcore::backtest(o, h, l, c, a, s, p);
    check(!rc.trades.empty() && near(rc.trades[0].r_multiple, 0.8), "costs reduce R");

    // Both levels inside one bar: the stop is assumed first.
    const fxcore::Vec h2{1.0, 1.02, 1.0, 1.0}, l2{1.0, 0.98, 1.0, 1.0};
    p.cost = 0.0;
    const auto r2 = fxcore::backtest(o, h2, l2, c, a, s, p);
    check(!r2.trades.empty() && r2.trades[0].reason == fxcore::kStop && near(r2.trades[0].r_multiple, -1.0),
          "ambiguous bar resolves to the stop");
}

}  // namespace

int main() {
    test_ema_sma();
    test_rsi_extremes();
    test_atr_constant_range();
    test_efficiency_ratio();
    test_triple_barrier();
    test_backtest_stop_and_target();
    if (failures == 0) std::printf("All fxcore tests passed\n");
    return failures == 0 ? 0 : 1;
}

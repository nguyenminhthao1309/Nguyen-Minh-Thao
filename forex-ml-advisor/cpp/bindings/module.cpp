// pybind11 bindings exposing the C++ core as fxquant._fxcore.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>

#include "fxcore/backtest.hpp"
#include "fxcore/indicators.hpp"
#include "fxcore/labeling.hpp"

namespace py = pybind11;
using fxcore::Vec;

namespace {

using InArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

Vec to_vec(const InArray& a) {
    if (a.ndim() != 1) throw std::invalid_argument("expected a 1-D array");
    Vec v(static_cast<std::size_t>(a.shape(0)));
    if (!v.empty()) std::memcpy(v.data(), a.data(), v.size() * sizeof(double));
    return v;
}

py::array_t<double> to_array(const Vec& v) {
    py::array_t<double> a(static_cast<py::ssize_t>(v.size()));
    if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
}

template <typename F>
auto unary(F f) {
    return [f](const InArray& x, int arg) { return to_array(f(to_vec(x), arg)); };
}

template <typename F>
auto hlc(F f) {
    return [f](const InArray& h, const InArray& l, const InArray& c, int period) {
        return to_array(f(to_vec(h), to_vec(l), to_vec(c), period));
    };
}

}  // namespace

PYBIND11_MODULE(_fxcore, m) {
    m.doc() = "High-performance forex indicator, labeling and backtesting core";

    m.def("ema", unary(fxcore::ema), py::arg("x"), py::arg("span"));
    m.def("sma", unary(fxcore::sma), py::arg("x"), py::arg("window"));
    m.def("rolling_zscore", unary(fxcore::rolling_zscore), py::arg("x"), py::arg("window"));
    m.def("rolling_percentile_rank", unary(fxcore::rolling_percentile_rank), py::arg("x"), py::arg("window"));
    m.def("rsi", unary(fxcore::rsi), py::arg("close"), py::arg("period"));
    m.def("efficiency_ratio", unary(fxcore::efficiency_ratio), py::arg("close"), py::arg("period"));
    m.def("atr", hlc(fxcore::atr), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("period"));
    m.def("adx", hlc(fxcore::adx), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("period"));
    m.def("stochastic_k", hlc(fxcore::stochastic_k), py::arg("high"), py::arg("low"), py::arg("close"),
          py::arg("period"));

    m.def(
        "rolling_std",
        [](const InArray& x, int window, int ddof) { return to_array(fxcore::rolling_std(to_vec(x), window, ddof)); },
        py::arg("x"), py::arg("window"), py::arg("ddof") = 1);
    m.def(
        "true_range",
        [](const InArray& h, const InArray& l, const InArray& c) {
            return to_array(fxcore::true_range(to_vec(h), to_vec(l), to_vec(c)));
        },
        py::arg("high"), py::arg("low"), py::arg("close"));
    m.def(
        "variance_ratio",
        [](const InArray& r, int window, int q) { return to_array(fxcore::variance_ratio(to_vec(r), window, q)); },
        py::arg("returns"), py::arg("window"), py::arg("q"));
    m.def(
        "bollinger_pctb",
        [](const InArray& c, int window, double nstd) {
            return to_array(fxcore::bollinger_pctb(to_vec(c), window, nstd));
        },
        py::arg("close"), py::arg("window"), py::arg("nstd") = 2.0);

    m.def(
        "triple_barrier",
        [](const InArray& h, const InArray& l, const InArray& c, const InArray& a, int horizon, double tp_mult,
           double sl_mult) {
            const Vec hv = to_vec(h), lv = to_vec(l), cv = to_vec(c), av = to_vec(a);
            fxcore::BarrierLabels r;
            {
                py::gil_scoped_release release;
                r = fxcore::triple_barrier(hv, lv, cv, av, horizon, tp_mult, sl_mult);
            }
            py::dict out;
            out["label"] = to_array(r.label);
            out["ret"] = to_array(r.ret);
            out["hold"] = to_array(r.hold);
            out["end_idx"] = to_array(r.end_idx);
            return out;
        },
        py::arg("high"), py::arg("low"), py::arg("close"), py::arg("atr"), py::arg("horizon"),
        py::arg("tp_mult"), py::arg("sl_mult"));

    m.def(
        "backtest",
        [](const InArray& o, const InArray& h, const InArray& l, const InArray& c, const InArray& a,
           const InArray& s, double sl_mult, double tp_mult, int max_hold, double risk_fraction, double cost) {
            fxcore::BacktestParams p{sl_mult, tp_mult, max_hold, risk_fraction, cost};
            const auto r = fxcore::backtest(to_vec(o), to_vec(h), to_vec(l), to_vec(c), to_vec(a), to_vec(s), p);

            const std::size_t k = r.trades.size();
            Vec entry_idx(k), exit_idx(k), direction(k), entry_px(k), exit_px(k), r_mult(k), reason(k);
            for (std::size_t i = 0; i < k; ++i) {
                const auto& t = r.trades[i];
                entry_idx[i] = t.entry_idx;
                exit_idx[i] = t.exit_idx;
                direction[i] = t.direction;
                entry_px[i] = t.entry_px;
                exit_px[i] = t.exit_px;
                r_mult[i] = t.r_multiple;
                reason[i] = t.reason;
            }
            py::dict trades;
            trades["entry_idx"] = to_array(entry_idx);
            trades["exit_idx"] = to_array(exit_idx);
            trades["direction"] = to_array(direction);
            trades["entry_px"] = to_array(entry_px);
            trades["exit_px"] = to_array(exit_px);
            trades["r_multiple"] = to_array(r_mult);
            trades["reason"] = to_array(reason);

            py::dict out;
            out["equity"] = to_array(r.equity);
            out["position"] = to_array(r.position);
            out["trades"] = trades;
            return out;
        },
        py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("atr"), py::arg("signal"),
        py::arg("sl_mult") = 1.5, py::arg("tp_mult") = 1.5, py::arg("max_hold") = 24,
        py::arg("risk_fraction") = 0.005, py::arg("cost") = 0.0);
}

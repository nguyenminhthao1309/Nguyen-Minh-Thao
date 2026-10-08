"""The C++ core and the NumPy reference must produce the same numbers."""

import numpy as np
import pytest

from fxquant import _reference as ref

cpp = pytest.importorskip("fxquant._fxcore", reason="C++ extension not built")


@pytest.fixture(scope="module")
def ohlc():
    rng = np.random.default_rng(7)
    n = 3000
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0007, (2, n)))
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    return open_, high, low, close


def assert_same(a, b):
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name,args", [
    ("ema", (20,)), ("sma", (50,)), ("rolling_zscore", (100,)),
    ("rolling_percentile_rank", (250,)), ("rsi", (14,)), ("efficiency_ratio", (24,)),
])
def test_close_only(ohlc, name, args):
    close = ohlc[3]
    assert_same(getattr(cpp, name)(close, *args), getattr(ref, name)(close, *args))


@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_std(ohlc, ddof):
    close = ohlc[3]
    assert_same(cpp.rolling_std(close, 30, ddof), ref.rolling_std(close, 30, ddof))


@pytest.mark.parametrize("name", ["atr", "adx", "stochastic_k"])
def test_hlc(ohlc, name):
    _, h, l, c = ohlc
    assert_same(getattr(cpp, name)(h, l, c, 14), getattr(ref, name)(h, l, c, 14))


def test_true_range_and_bollinger(ohlc):
    _, h, l, c = ohlc
    assert_same(cpp.true_range(h, l, c), ref.true_range(h, l, c))
    assert_same(cpp.bollinger_pctb(c, 20, 2.0), ref.bollinger_pctb(c, 20, 2.0))


def test_variance_ratio(ohlc):
    r = np.diff(np.log(ohlc[3]))
    assert_same(cpp.variance_ratio(r, 240, 4), ref.variance_ratio(r, 240, 4))


def test_ema_with_leading_nans():
    x = np.r_[np.nan, np.nan, np.arange(10.0)]
    assert_same(cpp.ema(x, 3), ref.ema(x, 3))


def test_triple_barrier(ohlc):
    _, h, l, c = ohlc
    a = ref.atr(h, l, c, 14)
    out_c = cpp.triple_barrier(h, l, c, a, 24, 1.5, 1.0)
    out_r = ref.triple_barrier(h, l, c, a, 24, 1.5, 1.0)
    for key in ("label", "ret", "hold", "end_idx"):
        assert_same(out_c[key], out_r[key])


@pytest.mark.parametrize("cost", [0.0, 0.0002])
def test_backtest(ohlc, cost):
    o, h, l, c = ohlc
    a = ref.atr(h, l, c, 14)
    rng = np.random.default_rng(3)
    signal = rng.choice([-1.0, 0.0, 0.0, 1.0], size=c.size)
    kw = dict(sl_mult=1.5, tp_mult=2.0, max_hold=24, risk_fraction=0.01, cost=cost)
    rc = cpp.backtest(o, h, l, c, a, signal, **kw)
    rr = ref.backtest(o, h, l, c, a, signal, **kw)
    assert_same(rc["equity"], rr["equity"])
    assert_same(rc["position"], rr["position"])
    assert len(rc["trades"]["r_multiple"]) > 50
    for key in rc["trades"]:
        assert_same(rc["trades"][key], rr["trades"][key])

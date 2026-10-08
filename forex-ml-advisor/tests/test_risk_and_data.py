"""Position sizing, portfolio limits and data loading."""

import numpy as np
import pandas as pd
import pytest

from fxquant.config import AdvisorConfig, pair_spec
from fxquant.data import fx_trading_hours, read_csv_ohlc
from fxquant.risk import apply_portfolio_limits, conversion_rate, kelly_fraction, position_size

PRICES = {"EURUSD": 1.10, "USDJPY": 150.0, "GBPUSD": 1.25, "EURGBP": 0.88}


def test_position_size_quote_usd():
    # 50 pip stop, 0.5% of 10,000 = 50 USD, pip value 10 USD/lot -> 0.10 lots
    s = position_size(10_000, 0.005, 1.1000, 1.0950, pair_spec("EURUSD"), PRICES)
    assert s["lots"] == pytest.approx(0.10)
    assert s["stop_pips"] == pytest.approx(50)


def test_position_size_jpy_quote():
    # 100 pip stop on USDJPY at 150: pip value = 1000 JPY = 6.67 USD per lot -> 0.075 lots
    s = position_size(10_000, 0.005, 150.00, 149.00, pair_spec("USDJPY"), PRICES)
    assert s["lots"] == pytest.approx(50 / (100 * 100_000 * 0.01 / 150))


def test_conversion_through_usd():
    assert conversion_rate("GBP", "USD", PRICES) == pytest.approx(1.25)
    assert conversion_rate("JPY", "USD", PRICES) == pytest.approx(1 / 150)
    assert conversion_rate("JPY", "EUR", PRICES) == pytest.approx(1 / 150 / 1.10)


def test_kelly():
    assert kelly_fraction(0.55, 1.0, 1.0) == pytest.approx(0.10)
    assert kelly_fraction(0.45, 1.0, 1.0) < 0


def _idea(pair, d, score, risk=0.005):
    return {"pair": pair, "direction": d, "score": score, "risk_fraction": risk}


def test_portfolio_blocks_duplicate_exposure():
    corr = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=["EURUSD", "GBPUSD"], columns=["EURUSD", "GBPUSD"])
    acc, rej = apply_portfolio_limits([_idea("EURUSD", 1, 2.0), _idea("GBPUSD", 1, 1.0)], corr, AdvisorConfig())
    assert [a["pair"] for a in acc] == ["EURUSD"]
    assert "same exposure" in rej[0]["blocked_by"]
    # Opposite directions on correlated pairs is a spread, not a duplicate
    acc, _ = apply_portfolio_limits([_idea("EURUSD", 1, 2.0), _idea("GBPUSD", -1, 1.0)], corr, AdvisorConfig())
    assert len(acc) == 2


def test_portfolio_caps_single_currency_risk():
    cfg = AdvisorConfig(max_currency_risk=0.01, risk_per_trade=0.005)
    pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]  # all short USD
    corr = pd.DataFrame(np.eye(4), index=pairs, columns=pairs)
    acc, rej = apply_portfolio_limits([_idea(p, 1, 4 - i) for i, p in enumerate(pairs)], corr, cfg)
    usd = sum(a["risk_fraction"] for a in acc)
    assert usd <= cfg.max_currency_risk + 1e-12
    assert len(rej) == 2


def test_fx_trading_hours_skip_weekend():
    idx = fx_trading_hours(pd.Timestamp("2026-03-06 20:00", tz="UTC"), pd.Timestamp("2026-03-08 23:00", tz="UTC"))
    assert list(idx.strftime("%a %H")) == ["Fri 20", "Fri 21", "Sun 22", "Sun 23"]


def test_read_mt4_style_csv(tmp_path):
    f = tmp_path / "EURUSD.csv"
    f.write_text("<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>\n"
                 "2026.03.02,00:00,1.1000,1.1010,1.0990,1.1005,100\n"
                 "2026.03.02,01:00,1.1005,1.1020,1.1000,1.1015,120\n")
    df = read_csv_ohlc(f)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert str(df.index.tz) == "UTC" and df.index[1] == pd.Timestamp("2026-03-02 01:00", tz="UTC")


def test_yahoo_loader_with_mocked_feed(monkeypatch):
    import sys
    import types

    from fxquant import data as data_mod

    idx = pd.date_range("2026-03-02 00:00", periods=6, freq="h", tz="Europe/London")
    frame = pd.DataFrame({"Open": 1.1, "High": 1.102, "Low": 1.098, "Close": 1.101, "Volume": 0}, index=idx)

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "EURUSD=X"

        def history(self, **kwargs):
            return frame

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    out = data_mod.load_yahoo(["EURUSD"])
    assert list(out["EURUSD"].columns) == ["open", "high", "low", "close"]
    assert str(out["EURUSD"].index.tz) == "UTC"


def test_unfinished_bar_is_dropped():
    from fxquant.data import drop_unfinished_bar

    idx = pd.date_range("2026-03-02 00:00", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    trimmed = drop_unfinished_bar(df, now=pd.Timestamp("2026-03-02 04:30", tz="UTC"))
    assert trimmed.index[-1] == pd.Timestamp("2026-03-02 03:00", tz="UTC")

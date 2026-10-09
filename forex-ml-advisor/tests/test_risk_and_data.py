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


@pytest.mark.parametrize("stamps,expected", [
    (["2026.03.02 00:00", "2026.03.13 01:00"], ["2026-03-02 00:00", "2026-03-13 01:00"]),        # MetaTrader
    (["02.03.2026 00:00:00.000", "03.03.2026 01:00:00.000"], ["2026-03-02 00:00", "2026-03-03 01:00"]),  # Dukascopy
    (["02.03.2026 00:00", "13.03.2026 01:00"], ["2026-03-02 00:00", "2026-03-13 01:00"]),        # day > 12
    (["03/02/2026 00:00", "03/13/2026 01:00"], ["2026-03-02 00:00", "2026-03-13 01:00"]),        # US month-first
    (["2026-03-02 00:00:00+00:00", "2026-03-02 03:00:00+02:00"], ["2026-03-02 00:00", "2026-03-02 01:00"]),
])
def test_timestamp_formats(stamps, expected):
    from fxquant.data import parse_timestamps

    got = parse_timestamps(pd.Series(stamps))
    assert list(got) == [pd.Timestamp(e, tz="UTC") for e in expected]


def _bars(n=30):
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="h")
    close = 1.1 + np.cumsum(np.full(n, 1e-4))
    return idx, close


def test_metatrader5_tab_separated(tmp_path):
    idx, close = _bars()
    lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    lines += [f"{t:%Y.%m.%d}\t{t:%H:%M:%S}\t{c:.5f}\t{c + 2e-4:.5f}\t{c - 2e-4:.5f}\t{c:.5f}\t10\t0\t5"
              for t, c in zip(idx, close)]
    f = tmp_path / "EURUSD.csv"
    f.write_text("\n".join(lines) + "\n", encoding="utf-16")  # MT5 can write UTF-16
    df = read_csv_ohlc(f)
    assert len(df) == 30 and df.index[0] == pd.Timestamp("2026-03-02 00:00", tz="UTC")


def test_metatrader4_without_header(tmp_path):
    idx, close = _bars()
    f = tmp_path / "EURUSD.csv"
    f.write_text("".join(f"{t:%Y.%m.%d},{t:%H:%M},{c:.5f},{c + 2e-4:.5f},{c - 2e-4:.5f},{c:.5f},100\n"
                         for t, c in zip(idx, close)))
    df = read_csv_ohlc(f)
    assert len(df) == 30 and df["close"].iloc[-1] == pytest.approx(close[-1], abs=1e-5)


@pytest.mark.parametrize("tz,winter_hour,summer_hour", [("UTC+2", 22, 22), ("Europe/Athens", 22, 21), ("UTC", 0, 0)])
def test_csv_timezone(tz, winter_hour, summer_hour):
    from fxquant.data import parse_timestamps

    got = parse_timestamps(pd.Series(["2026-01-15 00:00", "2026-07-15 00:00"]), timezone=tz)
    assert [got[0].hour, got[1].hour] == [winter_hour, summer_hour]


def test_bad_csv_file_is_skipped_not_fatal(tmp_path, capsys):
    from fxquant.data import load_csv_dir

    idx, close = _bars()
    pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                 index=pd.Index(idx, name="timestamp")).to_csv(tmp_path / "EURUSD.csv")
    (tmp_path / "GBPUSD.csv").write_text("this,is,not\nprice,data,at all\n")
    out = load_csv_dir(tmp_path)
    assert list(out) == ["EURUSD"]
    assert "skipping GBPUSD.csv" in capsys.readouterr().out

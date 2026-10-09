"""Command-line behaviour: friendly messages instead of tracebacks, input validation."""

import sys
import types

import pandas as pd
import pytest

from fxquant import cli


def run_cli(argv):
    """Run the CLI; return (exit code, stdout+stderr)."""
    try:
        cli.main(argv)
        return 0
    except SystemExit as exc:
        return exc.code


def test_yahoo_offline_gives_short_message_and_stops_early(monkeypatch, capsys):
    calls = []

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
            calls.append(symbol)

        def history(self, **kwargs):
            if len(calls) <= 2:
                raise ConnectionError("curl: (7) CONNECT tunnel failed")
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    assert run_cli(["analyze", "--source", "yahoo"]) == 1
    out = capsys.readouterr()
    assert "Yahoo Finance could not be reached" in out.err
    assert "Traceback" not in out.out + out.err
    assert len(calls) == 3  # gave up after three misses instead of trying all 28 pairs


def test_save_data_reports_failure(monkeypatch, capsys, tmp_path):
    class Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    assert run_cli(["save-data", "--source", "yahoo", "--pairs", "EURUSD", "--out", str(tmp_path)]) == 1
    assert "could not be reached" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.csv"))


def test_empty_csv_folder(capsys, tmp_path):
    assert run_cli(["analyze", "--source", "csv", "--csv-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "No price files found" in err and "data/README.md" in err


def test_missing_csv_folder(capsys, tmp_path):
    assert run_cli(["analyze", "--source", "csv", "--csv-dir", str(tmp_path / "nope")]) == 1
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("argv,message", [
    (["analyze", "--pairs", "EURUSDX"], "is not a currency pair"),
    (["analyze", "--segments", "major,foo"], "Unknown segment 'foo'"),
    (["analyze", "--account-currency", "dollars"], "3-letter code"),
    (["analyze", "--risk", "50"], "--risk between"),
    (["analyze", "--pairs", "EURXYZ"], "synthetic market has no EURXYZ"),
    (["analyze", "--source", "csv", "--csv-timezone", "Mars/Base"], "Unknown time zone"),
])
def test_bad_input_is_explained(capsys, argv, message):
    assert run_cli(argv) == 1
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


@pytest.mark.parametrize("text,expected", [("eurusd", "EURUSD"), ("EUR/USD", "EURUSD"), ("eur_usd", "EURUSD"),
                                           ("EURUSD=X", "EURUSD"), (" gbpjpy ", "GBPJPY")])
def test_pair_spellings(text, expected):
    assert cli.normalise_pair(text) == expected


def test_segment_spellings():
    assert cli.parse_segments("Majors, crosses,EXOTIC") == ["major", "cross", "exotic"]


def test_under_vscode_debugger_errors_end_quietly(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "debugpy", types.ModuleType("debugpy"))
    assert run_cli(["analyze", "--pairs", "EURUSDX"]) == 0  # no SystemExit, so F5 does not pause
    assert "is not a currency pair" in capsys.readouterr().err


def test_bench_without_cpp_is_not_an_error(monkeypatch, capsys):
    from fxquant import core

    monkeypatch.setattr(core, "BACKEND", "python")
    assert run_cli(["bench"]) == 0
    assert "nothing to compare" in capsys.readouterr().out


def test_save_data_merges_history(tmp_path, capsys):
    out = str(tmp_path)
    assert run_cli(["save-data", "--source", "synthetic", "--pairs", "EURUSD", "--years", "0.3", "--out", out]) == 0
    first = len(pd.read_csv(tmp_path / "EURUSD.csv"))
    assert run_cli(["save-data", "--source", "synthetic", "--pairs", "EURUSD", "--years", "0.6", "--seed", "5",
                    "--out", out]) == 0
    assert len(pd.read_csv(tmp_path / "EURUSD.csv")) > first


def test_csv_segments_and_unlisted_pairs(tmp_path, capsys):
    from fxquant.data import generate_synthetic

    for pair, df in generate_synthetic(["EURUSD", "EURGBP", "USDNOK"], years=0.2, seed=1).items():
        name = "EURNOK" if pair == "USDNOK" else pair  # a pair outside the built-in list
        df.rename_axis("timestamp").to_csv(tmp_path / f"{name}.csv")
    from fxquant.data import load_csv_dir

    assert set(load_csv_dir(tmp_path)) == {"EURUSD", "EURGBP", "EURNOK"}
    assert set(load_csv_dir(tmp_path, ["EURUSD", "GBPUSD"])) == {"EURUSD"}
    assert "no file for GBPUSD" in capsys.readouterr().out

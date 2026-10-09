"""One-click entry point for the forex ML analysis.

In VS Code: open this file and click the Run button (top right), or open Run and
Debug, choose "5. run.py - uses the SETTINGS at the top of run.py" and press
Ctrl+F5 (Cmd+F5 on Mac). Change the SETTINGS below to choose the data, the pairs,
your account size and risk. The full report is written to reports/latest_report.md
(open it and press Ctrl+Shift+V, or Cmd+Shift+V on Mac).
"""

# ----------------------------------------------------------------------------
# SETTINGS - edit these
# ----------------------------------------------------------------------------
SETTINGS = {
    # Where the prices come from:
    #   "synthetic" = built-in simulated market, works offline (demo only, not real prices)
    #   "yahoo"     = live hourly prices from Yahoo Finance (needs internet)
    #   "csv"       = your own files in the csv_dir folder, one per pair, e.g. data/EURUSD.csv
    "source": "synthetic",
    "csv_dir": "data",
    # Time zone of the times in your CSV files: "UTC" (default), an offset such as "UTC+2",
    # or a name. Most MetaTrader brokers use "Europe/Athens" (UTC+2 in winter, UTC+3 in summer).
    "csv_timezone": "UTC",

    # Which market sections to analyse: "major", "cross", "exotic"
    "segments": ["major", "cross", "exotic"],
    # Or list exact pairs (this overrides segments), e.g. ["EURUSD", "GBPUSD", "USDJPY"]
    "pairs": [],

    # Your account, used for position sizes in lots. The account currency must appear in at
    # least one analysed pair (USD, EUR, GBP, JPY, CHF, AUD, NZD, CAD, ... are fine).
    "equity": 10_000,
    "account_currency": "USD",
    "risk_percent": 0.5,  # % of the account you lose if a stop is hit

    # True = fewer model refits: about 2x faster, slightly less thorough
    "fast": False,

    # Output folder for reports
    "out": "reports",
}
# ----------------------------------------------------------------------------

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def check_environment(settings: dict) -> bool:
    missing = [m for m in ("numpy", "pandas", "sklearn") if importlib.util.find_spec(m) is None]
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
        print("Run the setup first: in VS Code use Terminal > Run Task > '1. Setup ...',")
        print("or double-click setup_windows.bat (Windows) / run 'bash setup_mac_linux.sh' (Mac/Linux).")
        print("Then select the .venv interpreter: Ctrl+Shift+P > 'Python: Select Interpreter'.")
        print(f"(Python currently running this file: {sys.executable})")
        return False
    if settings["source"] == "yahoo" and importlib.util.find_spec("yfinance") is None:
        print("Live data needs the yfinance package. Run the setup again, or install it with:")
        call = "& " if os.name == "nt" else ""  # PowerShell needs & before a quoted program path
        print(f'  {call}"{sys.executable}" -m pip install yfinance')
        return False
    return True


def build_arguments(settings: dict) -> list[str]:
    argv = ["analyze", "--source", settings["source"], "--csv-dir", settings["csv_dir"],
            "--csv-timezone", settings.get("csv_timezone", "UTC"),
            "--equity", str(settings["equity"]), "--account-currency", settings["account_currency"],
            "--risk", str(settings["risk_percent"]), "--out", settings["out"]]
    if settings["pairs"]:
        argv += ["--pairs", ",".join(settings["pairs"])]
    else:
        argv += ["--segments", ",".join(settings["segments"])]
    if settings["fast"]:
        argv.append("--fast")
    return argv


def main() -> None:
    os.chdir(HERE)  # reports/ and data/ are relative to this folder
    sys.path.insert(0, HERE)
    if not check_environment(SETTINGS):
        if "debugpy" not in sys.modules:  # under the VS Code debugger, end quietly instead of pausing
            sys.exit(1)
        return
    from fxquant.cli import main as fxquant_cli

    fxquant_cli(build_arguments(SETTINGS))


if __name__ == "__main__":
    main()

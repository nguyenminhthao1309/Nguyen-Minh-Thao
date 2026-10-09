# fxquant: machine-learning forex analysis and trade advisor (Python + C++)

`fxquant` analyses every section of the FX market and turns the analysis into concrete,
risk-managed trade ideas: direction, entry, stop, target, position size, and the reasons for each.

* **Market segments**: 7 majors, 14 crosses, 7 exotics (`fxquant/config.py`).
* **Trading sessions**: Sydney, Tokyo, London, New York, the London/NY overlap and off-hours, using daylight-saving-aware local hours.
* **Currencies**: each currency's strength, separated out from all pairs at once by least squares.
* **Regimes**: volatility (low/normal/high) and trend (uptrend/downtrend/range).
* **Timeframes**: H1 signals with H4 and D1 context, using completed higher-timeframe bars only.
* **Pairs**: dashboard with model view, out-of-sample track record and advice.

The speed-critical parts are written in C++17 and exposed to Python with pybind11: indicators,
triple-barrier labelling and the bar-by-bar backtester. Machine learning, analytics and reporting are
in Python. A NumPy implementation of the same core is included as a fallback, and tests check that it
matches the C++ to 1e-9.

> **Not financial advice.** This is a research tool. FX is leveraged, edges are small and unstable, and
> most retail traders lose money. Read [Limitations](#limitations) before using any output.

---

## Run it in Visual Studio Code

### 1. Install the tools (once)

| tool | where | notes |
|---|---|---|
| **Python 3.10 or newer** | https://www.python.org/downloads/ | Windows: tick **"Add python.exe to PATH"** in the installer |
| **Visual Studio Code** | https://code.visualstudio.com/ | |
| C++ compiler (optional) | Windows: [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/), "Desktop development with C++" workload. macOS: `xcode-select --install`. Linux: `sudo apt install g++` | Without a compiler everything still works on the NumPy fallback, just slower |

### 2. Open the folder

1. Save the `forex-ml-advisor` folder anywhere, for example in Documents. Unzip it first if you
   downloaded the `.zip`.
2. In VS Code choose **File > Open Folder...** and select the `forex-ml-advisor` folder itself (the one
   that contains `run.py`), not a folder above it. Otherwise VS Code will not find the run buttons.
3. Click **Install** when VS Code offers the recommended extensions (Python, Python Debugger, C/C++).

### 3. Set up (once, about 2-5 minutes)

* **Terminal > Run Task... > "1. Setup: create .venv, install packages, build C++ core"**
* Or double-click `setup_windows.bat` (Windows), or run `bash setup_mac_linux.sh` (macOS/Linux).

The setup creates a private Python environment in `.venv`, installs the packages, and builds the C++
core. At the end it prints `Numeric core in use: c++`, or `python` if no compiler was found.

Then tell VS Code to use that environment: press **Ctrl+Shift+P** (Cmd+Shift+P on Mac), run
**Python: Select Interpreter**, and pick the entry that shows `.venv`.

### 4. Run

Open **Run and Debug** (Ctrl+Shift+D), choose a configuration from the drop-down, and press **F5**:

| configuration | what it does |
|---|---|
| 1. Demo - synthetic market (offline) | full analysis on the built-in simulated market (no internet needed) |
| 2. Live - Yahoo Finance data | real hourly prices for all 28 pairs (needs internet) |
| 3. My data - CSV files in the data folder | your own price files, see `data/README.md` |
| 4. Quick run - majors only | 7 pairs, fewer refits; about 15 seconds |
| 5. run.py | uses the SETTINGS at the top of `run.py` (source, pairs, account size, risk) |
| 6. Save live Yahoo data to the data folder | downloads prices to CSV and merges them with older files |
| 7. Benchmark - C++ core vs Python | speed comparison |
| 8. Run all tests | the test suite |

You can also open `run.py`, edit `SETTINGS`, and click the **Run** button at the top right.

### 5. Read the result

The summary appears in the terminal. The full report is written to **`reports/latest_report.md`**:
open it and press **Ctrl+Shift+V** to see it formatted. Each run also keeps a time-stamped copy, a JSON
file of the trade ideas, and CSV tables.

### Troubleshooting

| message | fix |
|---|---|
| `No module named 'numpy'` (or pandas / sklearn) | VS Code is using another Python. Run the setup, then **Python: Select Interpreter** > `.venv` |
| `Python 3.10 or newer was not found` | install Python from python.org (Windows: tick "Add python.exe to PATH"), restart VS Code |
| `running scripts is disabled on this system` (PowerShell) | run once in the terminal: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Numeric core in use: python` | no C++ compiler was found. Results are identical, only slower. Install one (table above) and run the setup again |
| `No data loaded` with Yahoo | check your internet connection; Yahoo sometimes rate-limits, so wait a few minutes or pick fewer pairs |
| `could not create .venv` on Linux | `sudo apt install python3-venv` |

### What is in the folder

```
forex-ml-advisor/
  run.py                  one-click entry point (edit SETTINGS at the top)
  setup_windows.bat       one-click setup for Windows
  setup_mac_linux.sh      one-click setup for macOS / Linux
  .vscode/                run configurations, tasks, test and editor settings
  fxquant/                the Python package (ML, analytics, reports)
  cpp/                    the C++ core (indicators, labelling, backtester) + C++ unit tests
  tests/                  Python test suite
  data/                   put your own CSV price files here
  reports/                created on the first run; reports are written here
```

---

## Command line (any terminal)

```bash
cd forex-ml-advisor
pip install -e .            # installs fxquant and builds the C++ core if a compiler is available
pip install yfinance        # optional: live data

# Offline demo on a synthetic market (no internet needed)
fxquant analyze --source synthetic

# Live hourly data from Yahoo Finance, all 28 pairs
fxquant analyze --source yahoo --equity 25000 --risk 0.5

# Only majors and crosses, your own CSV files (EURUSD.csv, GBPJPY.csv, ...)
fxquant analyze --source csv --csv-dir ./data --segments major,cross

# Save live data to data/*.csv (re-run later to extend the history)
fxquant save-data --source yahoo --out data

# C++ vs Python speed comparison
fxquant bench
```

`python -m fxquant ...` works as well. Each run writes these files to `reports/`:

| file | contents |
|---|---|
| `latest_report.md`, `fx_report_<time>.md` | full report: trade ideas, every market section, model diagnostics, watchlist |
| `trade_ideas_<time>.json` | machine-readable ideas (entry/stop/target/lots/risk/rationale) |
| `pairs_*.csv`, `segments_*.csv`, `sessions_*.csv`, `strength_*.csv`, `folds_*.csv` | raw tables |

A full run (28 pairs, 2 years of hourly bars, monthly walk-forward refits) takes about 1–2 minutes;
`--fast` refits every 90 days instead.

### Building the C++ core separately

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build   # C++ unit tests
cmake -S . -B build -DFXCORE_BUILD_PYTHON=ON && cmake --build build       # module via CMake
```

The C++ core is optional: if it cannot be compiled, installation still succeeds and `fxquant` uses its
NumPy implementation automatically. Set `FXQUANT_BACKEND=python` to force the fallback.

---

## Architecture

```
            data.py  (Yahoo / CSV / synthetic)            C++ core (cpp/, via pybind11)
                 |                                     +-------------------------------+
                 v                                     | indicators: EMA, RSI, ATR,    |
 features.py  --calls-->  core.py  ------------------->| ADX, stochastic, efficiency   |
   |  sessions.py  strength.py  regime.py              | ratio, variance ratio, %B,    |
   v                                                   | rolling z-score / percentile  |
 model.py  purged walk-forward, validated stack        | triple-barrier labelling      |
   v                                                   | event-driven backtester       |
 backtest.py  OOS strategy per pair ------------------>+-------------------------------+
   v
 advisor.py  section analytics + gating + sizing (risk.py)
   v
 report.py  Markdown / JSON / CSV
```

| module | role |
|---|---|
| `cpp/src/indicators.cpp` | causal technical indicators (O(n) recursions, exact two-pass rolling stats) |
| `cpp/src/labeling.cpp` | triple-barrier labels: which of +k / −k barriers is touched first within the horizon |
| `cpp/src/backtest.cpp` | next-bar-open execution, stop/target with intrabar high/low, gap fills, costs, R-multiple equity |
| `fxquant/features.py` | ~50 scale-free features: multi-horizon momentum, oscillators, trend quality, volatility regime, H4/D1 context, session flags, currency-strength ranks, segment |
| `fxquant/model.py` | stacked model and purged walk-forward validation (see below) |
| `fxquant/sessions.py` | DST-aware session membership and per-session range, variance share and drift |
| `fxquant/strength.py` | least-squares currency-strength decomposition |
| `fxquant/regime.py` | causal volatility and trend regime labels |
| `fxquant/risk.py` | position size in lots (any account currency), Kelly cap, portfolio limits |
| `fxquant/advisor.py` | runs the pipeline and turns predictions into trade ideas |

## Method

**Labels.** For each bar, the triple-barrier method asks: within the next `horizon` bars (default 24 H1
bars, about one day), does price move up by `ATR × √horizon` before it moves down by the same amount? The
target is the sign of the realised barrier return. Bars without a fully observed future are never used
for training.

**Model.** The model is trained on all pairs pooled together, because the features are normalised by
volatility. It is a stack of three components:

1. **Trend factor**: the average of standardised momentum and trend features, with a single fitted coefficient. It is very hard to overfit.
2. **Gradient-boosted trees**: heavily regularised; capture non-linear interactions such as regime × momentum.
3. **Logistic regression**: L2-regularised, on all features.

Each refit holds out the most recent 25% of its training window, with purging. It weights each component
by its validated skill (AUC above 0.5) and calibrates the blend. If no component shows skill, the model
outputs 0.5, i.e. no signal. This matters in FX: free-form models easily learn period-specific patterns
that reverse out of sample.

**Validation.** Purged walk-forward, refit every 30 days. A training sample is used only if its label was
resolved before the test block starts, so overlapping labels cannot leak future information. Every model
statistic, backtest, expectancy figure and per-pair gate in the report is out of sample.

**From prediction to advice.** A pair becomes a trade idea only if all of the following hold:

* signal strength |P(up) − 0.5| ≥ 4% (the same threshold the backtest traded);
* the pair's out-of-sample strategy made money, with t-stat ≥ 1 and at least 20 trades;
* the pair's out-of-sample AUC is above 0.5;
* **expected value** ≥ +0.03R after this pair's spread and slippage. The expected value is the realised
  gross R of past out-of-sample trades with similar signal strength, minus today's cost; it is not a
  model-implied number;
* portfolio limits: total open risk ≤ 3%, net risk per currency ≤ 1% (so you are not short USD five
  times), and no two ideas with return correlation above 0.8 in the same direction.

**Sizing.** Base risk is 0.5% of equity at the stop, scaled by conviction (High / Medium / Low). It is
capped at a quarter of the Kelly fraction estimated from the pair's out-of-sample win rate and payoff,
and halved in high-volatility regimes. Lots are computed from the stop distance and the quote-to-account
currency conversion.

## Testing

```bash
pytest                          # 43 tests, about 1 minute
FXQUANT_BACKEND=python pytest   # same tests on the NumPy fallback
```

The tests cover:

* **C++/Python parity**: every indicator, labeller and backtester output agrees to 1e-9.
* **No look-ahead**: features computed on truncated history equal the same rows computed on the full
  history, including daily features and currency strength.
* **Signal recovery**: on a synthetic market with planted momentum the pipeline finds the edge (AUC > 0.53).
* **No false edge**: on pure random walks it does not (AUC ≈ 0.5, at most 2 ideas pass the gates).
* **Risk maths**: pip values, cross-currency conversion, Kelly, per-currency and correlation limits.
* **Sessions**: London opens 07:00 UTC in summer and 08:00 UTC in winter.
* **Data loading**: MetaTrader, Dukascopy, ISO and US time-stamp formats; live-feed handling.

## Synthetic market

`--source synthetic` simulates a consistent 15-currency market. Every pair is the ratio of two currency
paths, so cross rates never allow arbitrage. It includes risk-on/risk-off and European factor loadings,
calm and stress volatility regimes, stochastic volatility, intraday session seasonality, data-release
jumps, weekend gaps and carry drift in high-yield currencies. It also contains a **planted, slowly
mean-reverting drift** (`--momentum`, default 0.06) that gives the model something to find. Results on
synthetic data demonstrate the machinery; they say **nothing** about real markets. Use `--momentum 0`
to see how the system behaves when there is nothing to predict.

## Limitations

* **Small edges.** Realistic out-of-sample AUCs in FX are around 0.50–0.55. Expect long flat or losing
  stretches, and expect many days with no trade idea; that is the system working as intended.
* **Data.** Yahoo hourly FX history is limited to about 730 days, and its OHLC data is mid-price with
  occasional bad ticks (the loader removes isolated spikes). For serious use, load broker or Dukascopy
  data through CSV, with as much history as possible.
* **Costs.** Spreads in `config.py` are typical ECN values; set them to your broker's. Swap/rollover
  (carry) is not modelled, and it matters for multi-day holds in exotics such as USDTRY or USDZAR.
* **Multiple testing.** 28 pairs × several sections means some pairs pass the gates by luck. The
  t-stat and AUC gates reduce this but do not remove it.
* **No fundamentals or news.** The model only sees prices. Central-bank decisions, data releases and
  geopolitical events can overwhelm any statistical signal. Check the economic calendar before trading.
* **Paper trade first.** Run the system daily for weeks on a demo account before risking money.

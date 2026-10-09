# Your market data

Put one CSV file per currency pair in this folder, named after the pair, then run the analysis with
`--source csv` (VS Code: Run and Debug > "3. My data - CSV files in the data folder").

```
data/EURUSD.csv
data/GBPUSD.csv
data/USDJPY.csv
...
```

Every `<PAIR>.csv` in the folder is used, including pairs outside the built-in list (e.g. `EURNOK.csv`).

## Layouts that are recognised

Comma-, semicolon- and tab-separated files, with or without a header row:

```
timestamp,open,high,low,close
2026-03-02 00:00:00+00:00,1.10000,1.10100,1.09900,1.10050
```

```
<DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>     (MetaTrader 5 "Export Bars", tab-separated)
2026.03.02	00:00:00	1.10000	1.10100	1.09900	1.10050	1234	0	5
```

```
2026.03.02,00:00,1.10000,1.10100,1.09900,1.10050,100          (MetaTrader 4 History Center, no header)
```

```
Gmt time,Open,High,Low,Close,Volume                           (Dukascopy, day-first dates)
02.03.2026 00:00:00.000,1.10000,1.10100,1.09900,1.10050,1234
```

Day/month order is detected from the data (a day above 12 settles it). Dotted dates such as
`02.03.2026` are read day-first, slashed US dates such as `03/02/2026` month-first.

## Time zone

Time stamps without a time zone are read as **UTC**. MetaTrader files are usually in the broker's
*server time*, most often UTC+2 in winter and UTC+3 in summer. Tell the program with
`--csv-timezone Europe/Athens` (or `"csv_timezone": "Europe/Athens"` in `run.py`), or with a fixed offset
such as `--csv-timezone UTC+2`. Getting this right matters for the session analysis (London, New York, ...).

## How much history

Hourly bars are recommended because the session analysis needs intraday data. The model needs enough
history to train and then test itself out of sample:

* hourly bars: about **a year** for a handful of pairs (a few months is enough with 20+ pairs), 2 years recommended;
* daily bars: about **4 years or more** for a handful of pairs.

With too little history the program says `Not enough history to validate the model` and recommends no trades.

## Filling the folder automatically

Run "6. Save live Yahoo data to the data folder (CSV)", or `python -m fxquant save-data --source yahoo --out data`.
Running it again later merges the new bars into the existing files, so your history keeps growing beyond
Yahoo's 730-day limit for hourly data. For a quick offline test,
`python -m fxquant save-data --source synthetic --out data` writes simulated prices.

The files in this folder are ignored by git (see `.gitignore`).

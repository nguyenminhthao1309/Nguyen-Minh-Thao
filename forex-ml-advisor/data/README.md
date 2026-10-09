# Your market data

Put one CSV file per currency pair in this folder, named after the pair, then run the analysis with
`--source csv` (VS Code: F5 > "3. My data - CSV files in the data folder").

```
data/EURUSD.csv
data/GBPUSD.csv
data/USDJPY.csv
...
```

Each file needs a time column and open/high/low/close columns. Hourly bars are recommended because
the session analysis needs intraday data; daily bars also work. Times are read as UTC unless they include
a time zone. All of these layouts are recognised:

```
timestamp,open,high,low,close
2026-03-02 00:00:00+00:00,1.10000,1.10100,1.09900,1.10050
```

```
<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>          (MetaTrader export)
2026.03.02,00:00,1.10000,1.10100,1.09900,1.10050,100
```

```
Gmt time,Open,High,Low,Close,Volume                    (Dukascopy export)
```

To fill this folder automatically with live Yahoo Finance data, run F5 > "6. Save live Yahoo data to the
data folder (CSV)", or `python -m fxquant save-data --source yahoo --out data`. Running it again later
merges the new bars into the existing files, so your history keeps growing beyond Yahoo's 730-day
limit for hourly data.

The files in this folder are ignored by git (see `.gitignore`).

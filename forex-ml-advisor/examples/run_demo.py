"""Use fxquant from Python instead of the CLI.

    python examples/run_demo.py            # synthetic market (offline)
    python examples/run_demo.py yahoo      # live Yahoo Finance data (needs yfinance + internet)
"""

import sys

from fxquant import Config, ForexAdvisor
from fxquant.data import load_data
from fxquant.report import console_summary, save_report

source = sys.argv[1] if len(sys.argv) > 1 else "synthetic"

cfg = Config()                              # every pair in the universe (majors, crosses, exotics)
cfg.advisor.account_equity = 25_000         # your account size
cfg.advisor.risk_per_trade = 0.005          # 0.5% of equity at the stop
cfg.model.retrain_every = "60D"             # fewer refits = faster run

data = load_data(source, cfg.pairs)
result = ForexAdvisor(cfg).run(data)

print(console_summary(result))
print("\nSegment view:\n", result.segment_table.round(3))
print("\nCurrency strength:\n", result.strength_table.round(2))
print("\nModel by session:\n", result.model_by["session"][["n", "auc", "signal_hit_rate"]].round(3))
paths = save_report(result, "reports")
print(f"\nFull report written to {paths['report']}")

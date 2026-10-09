"""Market universe, trading sessions and tunable parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Market universe: every pair belongs to one market section (segment).
# Spreads are typical retail ECN round-trip costs in pips, used by the
# backtester and the expected-value calculation. Adjust them to your broker.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSpec:
    symbol: str
    segment: str  # "major" | "cross" | "exotic"
    spread_pips: float

    @property
    def base(self) -> str:
        return self.symbol[:3]

    @property
    def quote(self) -> str:
        return self.symbol[3:]

    @property
    def pip_size(self) -> float:
        return 0.01 if self.quote == "JPY" else 0.0001


_UNIVERSE_SPEC = {
    "major": {
        "EURUSD": 0.8, "GBPUSD": 1.2, "USDJPY": 1.0, "USDCHF": 1.5,
        "AUDUSD": 1.0, "USDCAD": 1.5, "NZDUSD": 1.5,
    },
    "cross": {
        "EURGBP": 1.2, "EURJPY": 1.5, "GBPJPY": 2.5, "EURCHF": 1.8, "AUDJPY": 1.8,
        "EURAUD": 2.0, "GBPCHF": 2.8, "CADJPY": 2.0, "AUDNZD": 2.5, "EURCAD": 2.2,
        "GBPAUD": 3.0, "NZDJPY": 2.2, "CHFJPY": 2.5, "AUDCAD": 2.0,
    },
    "exotic": {
        "USDMXN": 45.0, "USDZAR": 90.0, "USDTRY": 150.0, "USDSGD": 3.0,
        "USDNOK": 30.0, "USDSEK": 30.0, "USDPLN": 25.0,
    },
}

UNIVERSE: dict[str, PairSpec] = {
    sym: PairSpec(sym, seg, spread)
    for seg, pairs in _UNIVERSE_SPEC.items()
    for sym, spread in pairs.items()
}

SEGMENTS = ("major", "cross", "exotic")

# Trading sessions in local exchange time; converted to UTC per bar so that
# daylight-saving shifts are handled correctly.
SESSIONS: dict[str, tuple[str, int, int]] = {
    "Sydney": ("Australia/Sydney", 7, 16),
    "Tokyo": ("Asia/Tokyo", 9, 18),
    "London": ("Europe/London", 8, 17),
    "New York": ("America/New_York", 8, 17),
}


def pair_spec(symbol: str) -> PairSpec:
    """Spec for any 6-letter pair; unknown pairs default to the exotic segment."""
    symbol = symbol.upper()
    if symbol in UNIVERSE:
        return UNIVERSE[symbol]
    return PairSpec(symbol, "exotic", 30.0)


@dataclass
class ModelConfig:
    horizon: int = 24              # bars until the time barrier (24 x H1 = one day)
    barrier_mult: float = 1.0      # take-profit / stop distance in units of ATR * sqrt(horizon)
    atr_period: int = 14
    initial_train_frac: float = 0.4
    retrain_every: str = "30D"     # walk-forward refit frequency
    train_stride: int = 4          # subsample overlapping labels
    min_train_rows: int = 2000
    random_state: int = 42
    n_threads: int = 1             # CPU threads for model fitting (1 = fastest and reproducible here)


@dataclass
class StrategyConfig:
    entry_threshold: float = 0.04  # trade when |P(up) - 0.5| exceeds this
    risk_per_trade: float = 0.005  # equity fraction risked per backtest trade
    slippage_pips: float = 0.3


@dataclass
class AdvisorConfig:
    account_equity: float = 10_000.0
    account_currency: str = "USD"
    risk_per_trade: float = 0.005      # 0.5% of equity at the stop
    max_total_risk: float = 0.03       # sum of open risk across all ideas
    max_currency_risk: float = 0.01    # net risk per single currency
    max_recommendations: int = 6
    min_edge: float = 0.04             # minimum |P(up) - 0.5| (same scale as entry_threshold)
    min_expected_r: float = 0.03       # minimum expected R-multiple after costs
    min_oos_trades: int = 20
    correlation_limit: float = 0.8
    kelly_fraction: float = 0.25


@dataclass
class Config:
    pairs: list[str] = field(default_factory=lambda: list(UNIVERSE))
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    advisor: AdvisorConfig = field(default_factory=AdvisorConfig)

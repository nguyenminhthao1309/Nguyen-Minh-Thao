"""End-to-end checks: the system finds a planted signal and does not invent one."""

import numpy as np
import pytest

from fxquant.advisor import ForexAdvisor
from fxquant.config import Config
from fxquant.data import generate_synthetic
from fxquant.report import render_markdown

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "USDMXN"]


def _run(momentum: float, seed: int = 11):
    data = generate_synthetic(PAIRS, years=1.5, end="2026-09-30", seed=seed, momentum=momentum)
    cfg = Config(pairs=PAIRS)
    cfg.model.retrain_every = "60D"
    return ForexAdvisor(cfg, log=lambda *a: None).run(data)


@pytest.fixture(scope="module")
def strong():
    return _run(momentum=0.10)


@pytest.fixture(scope="module")
def null():
    return _run(momentum=0.0)


def test_recovers_planted_momentum(strong):
    assert strong.model_overall["auc"] > 0.53
    assert strong.portfolio["sharpe"] > 0


def test_no_edge_on_random_walks(null):
    assert abs(null.model_overall["auc"] - 0.5) < 0.03
    # Gates should keep false positives rare on pure noise
    assert len(null.recommendations) <= 2


def test_result_structure(strong):
    assert set(strong.pair_table.index) == set(PAIRS)
    assert set(strong.segment_table.index) <= {"major", "cross", "exotic"}
    assert not strong.session_summary.empty and not strong.strength_table.empty
    for r in strong.recommendations:
        assert r["action"] in ("BUY", "SELL")
        risk_side = (r["entry"] - r["stop"]) * r["direction"]
        assert risk_side > 0 and np.isclose(r["target"] - r["entry"], risk_side * r["direction"])
        assert 0 < r["risk_fraction"] <= strong.config.advisor.risk_per_trade
        assert r["lots"] > 0 and r["expected_r"] >= strong.config.advisor.min_expected_r
    assert strong.exposure.abs().max() <= strong.config.advisor.max_currency_risk + 1e-12 if len(strong.exposure) else True
    text = render_markdown(strong)
    assert "Not financial advice" in text and "## 2. Market sections" in text

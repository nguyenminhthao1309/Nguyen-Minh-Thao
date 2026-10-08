"""Features must never use information from the future."""

import numpy as np
import pandas as pd
import pytest

from fxquant.config import ModelConfig
from fxquant.data import generate_synthetic
from fxquant.features import add_labels, build_features, htf_features, infer_bar_delta
from fxquant.strength import currency_strength, strength_ranks


@pytest.fixture(scope="module")
def market():
    return generate_synthetic(["EURUSD", "GBPUSD", "USDJPY", "EURJPY"], years=0.5, end="2026-06-30", seed=3)


@pytest.mark.parametrize("cut_frac", [0.6, 0.85])
def test_features_have_no_lookahead(market, cut_frac):
    df = market["EURUSD"]
    cut = df.index[int(len(df) * cut_frac)]
    full = build_features(df, "EURUSD")
    partial = build_features(df.loc[:cut], "EURUSD")
    pd.testing.assert_frame_equal(full.loc[:cut], partial, check_exact=False, rtol=1e-9, atol=1e-12)


def test_strength_features_have_no_lookahead(market):
    cut = market["EURUSD"].index[2000]
    full = strength_ranks(currency_strength(market), 24)
    partial = strength_ranks(currency_strength({p: d.loc[:cut] for p, d in market.items()}), 24)
    pd.testing.assert_frame_equal(full.loc[:cut], partial)


def test_daily_features_appear_only_after_the_day_closes(market):
    df = market["EURUSD"]
    d1 = htf_features(df, "24h", "d1", "22h", infer_bar_delta(df.index))
    changes = d1["d1_rsi"].dropna().diff().fillna(1).ne(0)
    hours = d1["d1_rsi"].dropna().index[changes.to_numpy()].hour
    # A new daily value becomes visible on the last H1 bar of the FX day (21:00 UTC bar, closes 22:00)
    assert set(hours) <= {21}


def test_triple_barrier_labels(market):
    df = market["EURUSD"]
    feats = build_features(df, "EURUSD")
    cfg = ModelConfig(horizon=24)
    labels = add_labels(df, feats["atr"].to_numpy(), cfg)
    valid = labels["target"].notna()
    assert valid.iloc[-cfg.horizon:].sum() == 0  # unresolved futures are never labelled
    assert (labels.loc[valid, "label_end_time"] > labels.index[valid]).all()
    assert set(labels.loc[valid, "label"].unique()) <= {-1.0, 0.0, 1.0}
    # The target is the sign of the realised barrier return
    np.testing.assert_array_equal(labels.loc[valid, "target"], (labels.loc[valid, "label_ret"] > 0).astype(float))

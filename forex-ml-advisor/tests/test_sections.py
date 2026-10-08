"""Market-section analytics: currency strength and trading sessions."""

import numpy as np
import pandas as pd

from fxquant.sessions import session_membership
from fxquant.strength import currency_strength


def test_strength_recovers_single_currency_moves():
    idx = pd.date_range("2026-01-05", periods=200, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    ccys = ["USD", "EUR", "JPY", "GBP"]
    s = pd.DataFrame(rng.normal(0, 1e-3, (200, 4)), index=idx, columns=ccys)
    s = s.sub(s.mean(axis=1), axis=0)  # the decomposition is defined up to a common shift
    logv = s.cumsum()
    data = {}
    for pair in ["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "EURGBP", "GBPJPY"]:
        close = np.exp(logv[pair[:3]] - logv[pair[3:]])
        data[pair] = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    est = currency_strength(data)
    np.testing.assert_allclose(est[ccys].iloc[1:], s[ccys].iloc[1:], atol=1e-12)


def test_sessions_follow_daylight_saving():
    winter = pd.DatetimeIndex(["2026-01-14 07:00", "2026-01-14 08:00"], tz="UTC")
    summer = pd.DatetimeIndex(["2026-07-14 07:00", "2026-07-14 06:00"], tz="UTC")
    w = session_membership(winter)["London"].tolist()
    s = session_membership(summer)["London"].tolist()
    assert w == [False, True]  # London opens 08:00 UTC in winter (GMT)
    assert s == [True, False]  # and 07:00 UTC in summer (BST)


def test_overlap_and_off_hours():
    # Tue 14:00 UTC: London and New York both open. Fri 21:00 UTC: New York has
    # closed and it is already Saturday in Sydney/Tokyo, so nothing is open.
    idx = pd.DatetimeIndex(["2026-03-03 14:00", "2026-07-10 21:00"], tz="UTC")
    m = session_membership(idx)
    assert m.loc[idx[0], "London/NY overlap"]
    assert m.loc[idx[1], "Off-hours"]
    weekday = session_membership(pd.DatetimeIndex(["2026-03-03 21:00"], tz="UTC"))
    assert not weekday["Off-hours"].iloc[0]  # Sydney opens before New York closes

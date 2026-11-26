import numpy as np
import pandas as pd

from dhs import rebalance as rb
from dhs import sim
from dhs.real import forward_path


def test_forward_path_fills_missing_days_and_settles_on_the_last_trading_day():
    idx = pd.bdate_range("2024-05-31", "2024-06-28")
    nifty = pd.Series(np.linspace(22000, 23000, len(idx)), index=idx)
    have = idx[:-3]  # no slice on the last two trading days before settlement
    rows = pd.DataFrame({"date": have, "F": nifty.loc[have] * 1.002})
    expiry = pd.Timestamp("2024-06-29")  # a Saturday: settles on Friday 28 June
    days, F, sd, settle, filled = forward_path(rows, nifty, idx[0], expiry)
    assert sd == pd.Timestamp("2024-06-28") and settle == nifty.iloc[-1]
    assert days[-1] == pd.Timestamp("2024-06-27") and filled == 2
    assert np.allclose(F[-2:], nifty.loc[idx[-3:-1]] * 1.002)


def test_whalley_wilmott_band_widens_with_gamma_and_leland_raises_the_vol():
    F, T, r = 25000.0, 20 / 365, 0.065
    t = np.array([0.0, 1 / 365, 2 / 365])

    def band(K):
        g = float(sim.gamma(F, K, T, r, 0.15))
        return (1.5 * np.exp(-r * T) * rb.RATE_SELL * F * g * g / 1.0) ** (1 / 3)

    assert band(25000.0) > band(28000.0)  # more gamma at the money, wider band
    path = np.array([F, F * 1.01, F * 1.02])
    daily = rb.hedge_rule(path, t, F * 1.02, T, r, 0.15, "call", path[-1], "time", 1)
    leland = rb.hedge_rule(path, t, F * 1.02, T, r, 0.15, "call", path[-1], "leland", 0)
    assert not np.isclose(daily[0], leland[0])  # a different (raised-vol) delta was used

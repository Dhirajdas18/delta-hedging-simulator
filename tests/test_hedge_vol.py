import numpy as np
import pandas as pd

from dhs import sim
from dhs.real import hedge_path, trailing_vols


def test_hedge_vol_changes_only_the_delta_not_the_premium():
    T, steps, r = 30 / 365, 30, 0.065
    p = sim.simulate_paths(100.0, 0.13, T, steps, 2, seed=5)
    t = np.linspace(0, T, steps + 1)
    for i in range(2):
        a, _, prem_a = hedge_path(p[i], t, 100.0, T, r, 0.15, "call", p[i, -1])
        b, gb, prem_b = hedge_path(p[i], t, 100.0, T, r, 0.15, "call", p[i, -1], sigma_hedge=0.13)
        out = sim.hedge_pnl(p[i:i + 1], 100.0, T, r, 0.13, 0.15, "call")
        assert prem_a == prem_b and np.isclose(b, out["pnl"][0], atol=1e-9)
        assert not np.isclose(a, b)


def test_trailing_vols_use_only_data_up_to_the_start():
    idx = pd.bdate_range("2024-01-01", periods=60)
    s = pd.Series(100 * np.exp(np.cumsum(np.r_[0, np.full(59, 0.01)])), index=idx)
    v = trailing_vols(s, idx[40])
    assert np.isclose(v["hist20"], 0.01 * np.sqrt(252))
    s2 = s.copy()
    s2.iloc[41:] *= 2  # later prices must not matter
    assert trailing_vols(s2, idx[40]) == v

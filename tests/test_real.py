import numpy as np

from dhs import sim
from dhs.real import hedge_path


def test_real_path_hedge_matches_the_simulator_on_an_even_grid():
    T, steps, r, v = 30 / 365, 30, 0.065, 0.15
    paths = sim.simulate_paths(100.0, 0.13, T, steps, 3, seed=3)
    t = np.linspace(0, T, steps + 1)
    for kind, K in (("call", 100.0), ("put", 95.0)):
        out = sim.hedge_pnl(paths, K, T, r, v, v, kind)
        for i in range(3):
            pnl, gsum, _ = hedge_path(paths[i], t, K, T, r, v, kind, paths[i, -1])
            assert np.isclose(pnl, out["pnl"][i], rtol=1e-9, atol=1e-9)


def test_gamma_sum_tracks_the_hedged_pnl_when_hedging_at_the_sold_vol():
    T, steps, r, v = 30 / 365, 120, 0.065, 0.15
    paths = sim.simulate_paths(100.0, 0.15, T, steps, 400, seed=4)
    t = np.linspace(0, T, steps + 1)
    got = np.array([hedge_path(p, t, 100.0, T, r, v, "call", p[-1])[:2] for p in paths])
    assert np.corrcoef(got[:, 0], got[:, 1])[0, 1] > 0.95

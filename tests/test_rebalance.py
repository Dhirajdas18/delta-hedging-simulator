import numpy as np

from dhs import rebalance as rb
from dhs import sim
from dhs.real import hedge_path

T, STEPS, R = 30 / 365, 30, 0.065
P = sim.simulate_paths(25000.0, 0.13, T, STEPS, 3, seed=8)
TT = np.linspace(0, T, STEPS + 1)


def test_cost_rates_follow_the_table():
    # sell Rs 25,000: STT 0.05%, NSE and SEBI with GST, Rs 20 x 1.18 over 650 units
    c = rb.trade_cost(-1.0, 25000.0)
    want = 25000 * (0.0005 + (0.0000183 + 0.000001) * 1.18) + 20 * 1.18 / 650
    assert np.isclose(c, want)
    assert rb.trade_cost(1.0, 25000.0) < c  # buying pays stamp duty, not STT
    assert rb.trade_cost(0.0, 25000.0) == 0.0


def test_daily_rule_equals_the_plain_hedge_minus_its_costs():
    for p in P:
        pnl, cost, trades = rb.hedge_rule(p, TT, 25000.0, T, R, 0.15, "call", p[-1], "time", 1)
        plain = hedge_path(p, TT, 25000.0, T, R, 0.15, "call", p[-1])[0]
        assert np.isclose(pnl + cost, plain) and trades == STEPS


def test_bands_trade_less_and_zero_band_is_daily():
    p = P[0]
    zero = rb.hedge_rule(p, TT, 25000.0, T, R, 0.15, "call", p[-1], "band", 0.0)
    daily = rb.hedge_rule(p, TT, 25000.0, T, R, 0.15, "call", p[-1], "time", 1)
    wide = rb.hedge_rule(p, TT, 25000.0, T, R, 0.15, "call", p[-1], "band", 0.1)
    assert np.isclose(zero[0], daily[0]) and wide[2] < daily[2] and wide[1] < daily[1]
    ww = rb.hedge_rule(p, TT, 25000.0, T, R, 0.15, "call", p[-1], "ww", 1e-4)
    assert ww[2] <= daily[2]

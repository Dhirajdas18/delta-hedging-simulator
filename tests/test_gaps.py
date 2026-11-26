import numpy as np
import pandas as pd

from dhs import sim
from dhs.gaps import hedge_twice, month_gaps
from dhs.real import hedge_path

T, STEPS, R = 30 / 365, 30, 0.065


def test_no_gap_twice_a_day_is_close_to_once_a_day():
    p = sim.simulate_paths(25000.0, 0.15, T, STEPS, 1, seed=9)[0]
    t = np.linspace(0, T, STEPS + 1)
    once = hedge_path(p, t, 25000.0, T, R, 0.15, "call", p[-1])[0]
    twice = hedge_twice(p, t, np.ones(STEPS), 25000.0, T, R, 0.15, "call", p[-1])
    assert abs(twice - once) < 0.02 * 25000 * 0.15 * np.sqrt(T)  # tiny vs the option's scale


def test_a_gap_against_the_hedge_costs_the_same_whether_or_not_you_rebalance_at_the_open():
    # one step: a 5% overnight gap, then no move; rebalancing at the open comes too late
    t = np.array([0.0, 1 / 365])
    F = np.array([100.0, 95.0])
    once = hedge_path(F, t, 100.0, 1 / 365, 0.0, 0.2, "call", 95.0)[0]
    twice = hedge_twice(F, t, np.array([0.95]), 100.0, 1 / 365, 0.0, 0.2, "call", 95.0)
    assert np.isclose(once, twice)


def test_month_gaps_reads_open_over_previous_close_by_date():
    d0 = pd.Timestamp("2024-01-01")
    idx = pd.date_range(d0, periods=4)
    ohlc = pd.DataFrame({"open": [100, 102, 99, 101], "close": [101, 100, 100, 102]}, index=idx)
    case = {"date": d0, "t": np.array([0, 1, 2, 3]) / 365}
    assert np.allclose(month_gaps(case, ohlc), [102 / 101, 99 / 100, 101 / 100])

import numpy as np
import pytest

from dhs import sim

F0, K, T, R = 25000.0, 25000.0, 30 / 365, 0.065


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("k", [22000.0, 25000.0, 27500.0])
def test_greeks_match_finite_differences(kind, k):
    s, h, e = 0.15, 1.0, 1e-5
    p = lambda f=F0, v=s: sim.black76(f, k, T, R, v, kind)  # noqa: E731
    assert sim.delta(F0, k, T, R, s, kind) == pytest.approx((p(F0 + h) - p(F0 - h)) / (2 * h),
                                                            rel=1e-5, abs=1e-9)
    assert sim.gamma(F0, k, T, R, s) == pytest.approx(
        (p(F0 + h) - 2 * p() + p(F0 - h)) / h**2, rel=1e-3, abs=1e-9)
    assert sim.vega(F0, k, T, R, s) == pytest.approx((p(v=s + e) - p(v=s - e)) / (2 * e), rel=1e-6)


def test_put_call_parity_and_vectorised():
    ks = np.array([23000.0, 25000.0, 27000.0])
    c, p = sim.black76(F0, ks, T, R, 0.15, "call"), sim.black76(F0, ks, T, R, 0.15, "put")
    assert c - p == pytest.approx(np.exp(-R * T) * (F0 - ks), rel=1e-12)
    with pytest.raises(ValueError):
        sim.black76(F0, K, T, R, 0.15, "straddle")


@pytest.mark.parametrize("jumps", [(0.0, 0.0, 0.0), (8.0, -0.03, 0.04)])
def test_paths_are_martingales(jumps):
    paths = sim.simulate_paths(F0, 0.2, 0.5, 50, 40_000, 7, *jumps)
    assert paths.shape == (40_000, 51) and np.all(paths[:, 0] == F0)
    for col in (10, 50):
        f = paths[:, col]
        assert abs(f.mean() - F0) < 4 * f.std() / np.sqrt(len(f))


def _run(sold=0.15, real=0.15, hedge=None, steps=90, n=4000, seed=11, **kw):
    paths = sim.simulate_paths(F0, real, 0.25, steps, n, seed)
    return sim.hedge_pnl(paths, K, 0.25, R, hedge or sold, sold, "call", **kw)


def test_hedging_at_the_right_vol_averages_zero_and_error_halves_with_4x_rebalancing():
    daily, every4 = _run(every=1), _run(every=4)
    sd1, sd4 = daily["pnl"].std(), every4["pnl"].std()
    assert abs(daily["pnl"].mean()) < 3 * sd1 / np.sqrt(4000)
    assert sd1 < 0.15 * daily["unhedged"].std()
    assert sd4 / sd1 == pytest.approx(2, rel=0.12)
    assert daily["trades"].mean() == 90 and every4["trades"].mean() == 23


def test_pnl_follows_the_gamma_formula():
    out = _run(steps=120)
    pred = out["edge"] + out["gamma"]
    assert np.corrcoef(out["pnl"], pred)[0, 1] > 0.95
    assert np.std(out["pnl"] - pred) < 0.35 * out["pnl"].std()


def test_wrong_hedge_vol_still_matches_formula_with_premium_edge():
    out = _run(sold=0.18, real=0.15, hedge=0.15, steps=120)
    assert out["edge"][0] > 0
    assert np.corrcoef(out["pnl"], out["edge"] + out["gamma"])[0, 1] > 0.9
    assert np.mean(out["pnl"] - out["edge"] - out["gamma"]) == pytest.approx(0, abs=2)


@pytest.mark.parametrize("hedge", [None, 0.15])
def test_selling_above_realised_vol_pays_on_average(hedge):
    out = _run(sold=0.2, real=0.15, hedge=hedge)
    assert out["pnl"].mean() > 5 * out["pnl"].std() / np.sqrt(4000)


def test_delta_band_trades_less_than_daily():
    daily, banded = _run(), _run(rebalance="band", band=0.05)
    assert 1 < banded["trades"].mean() < 0.5 * daily["trades"].mean()
    assert np.all(banded["held"][:, 0] == daily["held"][:, 0])  # the first step always trades


@pytest.mark.parametrize("rule", [{"every": 2}, {"rebalance": "band", "band": 0.03}])
def test_costs_reduce_pnl_by_exactly_the_reported_cost(rule):
    free = _run(**rule)
    paid = _run(cost_pct=0.0002, cost_fixed=20.0, **rule)
    assert np.all(paid["costs"] > 0)
    assert paid["pnl"] == pytest.approx(free["pnl"] - paid["costs"], abs=1e-9)
    assert np.all(paid["trades"] == free["trades"])


def test_bad_rules_are_rejected():
    with pytest.raises(ValueError):
        _run(rebalance="band")
    with pytest.raises(ValueError):
        _run(every=0)

import numpy as np
import pandas as pd

from dhs import gamma
from dhs.real import hedge_path


def _month(vol=0.15, days=20):
    d0 = pd.Timestamp("2024-01-31")
    idx = pd.bdate_range(d0, periods=days)
    rng = np.random.default_rng(1)
    F = 25000 * np.exp(np.cumsum(np.r_[0, rng.normal(0, 0.008, days - 1)]))
    exp1, exp2 = idx[-1] + pd.Timedelta(days=3), idx[-1] + pd.Timedelta(days=31)
    def slc(exp):
        T = ((exp - idx).days / 365).to_numpy()
        return pd.DataFrame({"F": F, "T": T, "a": vol**2 * T, "b": 0.0, "rho": 0.0, "m": 0.0,
                             "sigma": 0.1}, index=idx)
    f, b = slc(exp1), slc(exp2)
    return {"date": d0, "front": f, "back": b, "expiry": exp1, "settle": float(F[-1] * 1.01),
            "back_last": b.iloc[-1]}


def test_option_cost_pays_stt_on_sales_and_half_the_spread():
    sell = gamma.option_cost(-1.0, 400.0)
    buy = gamma.option_cost(1.0, 400.0)
    assert sell > buy > 0.5  # at least half a point of spread
    assert np.isclose(sell - buy, 400.0 * (0.0015 - 0.00003))


def test_futures_only_is_the_plain_hedge_minus_its_costs():
    m = _month()
    out = gamma.run_month(m, "futures")
    f = m["front"]
    t = np.r_[((f.index - m["date"]).days / 365).to_numpy(), float(f["T"].iloc[0])]
    F = np.r_[f["F"].to_numpy(), m["settle"]]
    plain = hedge_path(F, t, F[0], t[-1], gamma.R, 0.15, "call", m["settle"])[0]
    assert np.isclose(out["pnl"] + out["cost"], plain, rtol=1e-9)


def test_gamma_hedges_hold_back_options_and_pay_more_costs():
    m = _month()
    fut, st, dyn = (gamma.run_month(m, k) for k in ("futures", "static", "dynamic"))
    assert st["n_back"] > 0 and dyn["n_back"] > 0
    assert st["cost"] > fut["cost"] and dyn["cost"] > st["cost"]

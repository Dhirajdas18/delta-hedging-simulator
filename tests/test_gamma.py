import numpy as np
import pandas as pd

from dhs import gamma, sim
from dhs.real import hedge_path


def _month(days=20):
    d0 = pd.Timestamp("2024-01-31")
    idx = pd.bdate_range(d0, periods=days)
    rng = np.random.default_rng(1)
    F1 = 25000 * np.exp(np.cumsum(np.r_[0, rng.normal(0, 0.008, days - 1)]))
    T1 = (idx[-1] + pd.Timedelta(days=3) - d0).days / 365
    back_exp = idx[-1] + pd.Timedelta(days=31)
    sd = idx[-1] + pd.Timedelta(days=3)

    def row(day, F):
        T = (back_exp - day).days / 365
        return {"date": day, "F": F, "T": T, "a": 0.15**2 * T, "b": 0.0, "rho": 0.0, "m": 0.0,
                "sigma": 0.1}

    back = pd.DataFrame([row(d, F) for d, F in zip(idx, F1, strict=True)])
    settle = float(F1[-1] * 1.01)
    return {"date": d0, "days": idx, "F1": F1, "T1": T1, "K": float(F1[0]), "vol1": 0.15,
            "settle": settle, "back": back, "back_exp": back_exp,
            "back_settle": pd.Series(row(sd, settle)), "settle_day": sd}


def test_option_cost_pays_stt_on_sales_and_half_the_spread():
    sell, buy = gamma.option_cost(-1.0, 400.0), gamma.option_cost(1.0, 400.0)
    assert sell > buy > 0.5
    assert np.isclose(sell - buy, 400.0 * (0.0015 - 0.00003))


def test_futures_only_is_the_plain_hedge_minus_its_costs():
    m = _month()
    out = gamma.run_month(m, "futures")
    t = np.r_[((m["days"] - m["date"]).days / 365).to_numpy(), m["T1"]]
    F = np.r_[m["F1"], m["settle"]]
    plain = hedge_path(F, t, m["K"], m["T1"], gamma.R, 0.15, "call", m["settle"])[0]
    assert np.isclose(out["pnl"] + out["cost"], plain, rtol=1e-9)


def test_dynamic_hedge_keeps_the_book_gamma_neutral_and_costs_more():
    m = _month()
    fut, st, dyn = (gamma.run_month(m, k) for k in ("futures", "static", "dynamic"))
    assert st["cost"] > fut["cost"] and dyn["cost"] > st["cost"]
    # last day: n back options cancel the short call's gamma
    i = len(m["days"]) - 1
    T1i = m["T1"] - (m["days"][i] - m["date"]).days / 365
    g1 = float(sim.gamma(m["F1"][i], m["K"], T1i, gamma.R, 0.15))
    b = m["back"].iloc[i]
    g2 = float(sim.gamma(b["F"], m["K"], (m["back_exp"] - m["days"][i]).days / 365, gamma.R, 0.15))
    assert np.isclose(dyn["n_back"] * g2, g1, rtol=1e-6)

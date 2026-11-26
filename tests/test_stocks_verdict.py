import math

import numpy as np
import pandas as pd

from dhs import sim, stocks, verdict


def _panel(tmp_path, days, price=1000.0):
    p = pd.DataFrame({"date": days, "symbol": "ABC", "close": price, "adj_close": price})
    p.to_parquet(tmp_path / "panel.parquet")


def test_stock_option_settles_on_the_stock_not_the_carried_forward(tmp_path, monkeypatch):
    days = pd.bdate_range("2024-01-31", "2024-02-29")
    _panel(tmp_path, days)
    monkeypatch.setattr(stocks, "lake_dir", lambda: tmp_path)
    monkeypatch.setattr(stocks, "current_symbols", lambda: {})
    T = (days[-1] - days[0]).days / 365
    case = pd.DataFrame([{"date": days[0], "symbol": "ABC", "expiry": days[-1], "T": T,
                          "F0": 1010.0, "K": 1000.0, "vol": 0.3}])
    out = stocks.hedge_months(case).iloc[0]
    prem = float(sim.black76(1010.0, 1000.0, T, stocks.R, 0.3, "call"))
    # flat stock at 1000: the forward's carry runs off, the call expires at the money, worth 0
    assert np.isclose(out["unhedged"], 100 * prem * math.exp(stocks.R * T) / 1010.0)


def test_verdict_book_sums_call_and_put_and_nets_costs(tmp_path, monkeypatch):
    t = np.arange(21) / 365
    F = 25000 * np.exp(np.r_[0, np.cumsum(np.random.default_rng(3).normal(0, 0.008, 20))])
    d0 = pd.Timestamp("2024-01-31")
    cases = [{"date": d0, "kind": k, "F": F, "t": t, "K": 25000 * m, "T": t[-1], "vol": 0.14,
              "settle": F[-1]} for k, m in (("call", 1.0), ("put", 0.95))]
    monkeypatch.setattr(verdict, "month_cases", lambda: cases)
    monkeypatch.setattr(verdict, "trailing_vols", lambda s, d: {"hist20": 0.10})
    pd.DataFrame({"date": [d0], "nifty": [25000.0]}).to_parquet(tmp_path / "market.parquet")
    monkeypatch.setattr("dhs.real.lake_dir", lambda: tmp_path)
    m = verdict.book().iloc[0]
    assert m["n"] == 2 and bool(m["premium_rich"])
    assert np.isclose(m["gross"] - m["net"], m["fut_cost"] + m["opt_cost"])
    assert m["fut_cost"] > 0 and m["opt_cost"] > 0

"""Is selling and hedging NIFTY options an edge after costs and gaps?

The book: at each month-end, sell one monthly at-the-money call and one 95% put (finding 1's
cases) at their SVI implied volatilities; hedge each with futures on finding 3's delta band (the
settings chosen there on 2015-2020: 0.03 for the call, 0.01 for the put), paying futures costs
(`dhs.rebalance`); pay the option sale's costs (`dhs.gamma.option_cost`: STT, NSE, SEBI, GST,
brokerage and half of an assumed 1-point spread). Results are per unit of NIFTY, as % of the
forward, summed over the two options. Also a rule fixed in advance: sell only in months when the
sold volatility is above NIFTY's last-20-day volatility.
"""

import math

import numpy as np
import pandas as pd

from dhs import sim
from dhs.gamma import option_cost
from dhs.real import R, month_cases, trailing_vols
from dhs.rebalance import hedge_rule

BANDS = {"call": 0.03, "put": 0.01}


def book(r=R) -> pd.DataFrame:
    from dhs.real import lake_dir

    nifty = pd.read_parquet(lake_dir() / "market.parquet").set_index("date")["nifty"].dropna()
    rows = []
    for c in month_cases():
        F0 = c["F"][0]
        pnl, fut_cost, trades = hedge_rule(c["F"], c["t"], c["K"], c["T"], r, c["vol"], c["kind"],
                                           c["settle"], "band", BANDS[c["kind"]])
        prem = float(sim.black76(F0, c["K"], c["T"], r, c["vol"], c["kind"]))
        opt_cost = option_cost(-1.0, prem) * math.exp(r * c["T"])
        rows.append({"date": c["date"], "kind": c["kind"], "F0": F0,
                     "gross": 100 * (pnl + fut_cost) / F0, "net": 100 * (pnl - opt_cost) / F0,
                     "fut_cost": 100 * fut_cost / F0, "opt_cost": 100 * opt_cost / F0,
                     "sold_vol": c["vol"], "hist20": trailing_vols(nifty, c["date"])["hist20"]})
    d = pd.DataFrame(rows)
    m = d.groupby("date").agg(gross=("gross", "sum"), net=("net", "sum"),
                              fut_cost=("fut_cost", "sum"), opt_cost=("opt_cost", "sum"),
                              n=("kind", "size"))
    call = d[d["kind"] == "call"].set_index("date")
    m["premium_rich"] = call["sold_vol"] > call["hist20"]
    return m[m["n"] == 2]


def summary(x: pd.Series) -> dict:
    """Mean, spread, t (months taken as independent), share positive, worst month, worst
    drawdown of the running sum, all in % of NIFTY."""
    run = x.cumsum()
    dd = float((run - run.cummax()).min())
    n = len(x)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std()),
            "t": float(x.mean() / x.std() * np.sqrt(n)) if n > 1 else float("nan"),
            "pos": float((x > 0).mean()), "worst": float(x.min()), "dd": dd,
            "annual": float(x.mean() * 12), "sharpe": float(x.mean() / x.std() * np.sqrt(12))}

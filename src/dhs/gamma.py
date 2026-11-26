"""Gamma hedging a short NIFTY option with a second option, on real months.

At each month-end, sell the monthly at-the-money call (front expiry 20-45 days out) as in
finding 1, and hedge with:
- futures only (daily delta hedge);
- static: buy n = Gamma_front / Gamma_back of the next monthly expiry's option at the same strike
  at the start, hold it, delta-hedge the rest daily;
- dynamic: re-set n to keep the book gamma-neutral at every close, then delta-hedge the rest.
The back option is priced every day at Black-76 with its own slice's SVI volatility at the strike
(implied-vol-surface's daily table) and sold back on the front expiry day. Its delta and gamma
are in its own forward's units, converted to front-futures units with F_back / F_front.

Costs at today's Indian rates: futures as `dhs.rebalance`; options per black-scholes-pricer's
table: STT 0.15% of premium on sales, NSE 0.03553% and SEBI 0.0001% of premium, stamp 0.003% on
purchases, GST 18% on charges and brokerage, Rs 20 an order over a 650-unit book, plus half of an
assumed 1-point bid-ask spread on every option trade. The front option's own sale cost is the
same for every method and left out.
"""

import math

import pandas as pd

from dhs import sim
from dhs.real import R, iv_dir, lake_dir, svi_vol
from dhs.rebalance import trade_cost

OPT_NSE, OPT_SEBI, OPT_STAMP, OPT_STT, GST, ORDER = 0.0003553, 0.000001, 0.00003, 0.0015, 0.18, 20.0
UNITS, HALF_SPREAD = 650, 0.5


def option_cost(units, premium):
    """Cost (Rs per unit of NIFTY in the book) of buying (units > 0) or selling options."""
    if units == 0:
        return 0.0
    v = abs(units) * premium
    side = OPT_STAMP if units > 0 else OPT_STT
    return v * (side + (OPT_NSE + OPT_SEBI) * (1 + GST)) + ORDER * (1 + GST) / UNITS + abs(
        units) * HALF_SPREAD


def month_pairs() -> list[dict]:
    """Front and back monthly slices for each month-end, aligned by date."""
    svi = pd.read_parquet(iv_dir() / "data" / "derived" / "svi_daily.parquet")
    svi = svi[svi["converged"]]
    nifty = pd.read_parquet(lake_dir() / "market.parquet").set_index("date")["nifty"].dropna()
    dates = pd.DatetimeIndex(sorted(svi["date"].unique()))
    out = []
    for d0 in pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max():
        g = svi[(svi["date"] == d0) & svi["monthly"]].sort_values("days")
        front = g[g["days"].between(20, 45)]
        if front.empty:
            continue
        f0 = front.iloc[0]
        back = g[g["expiry"] > f0["expiry"]]
        if back.empty or f0["expiry"] not in nifty.index:
            continue
        b0 = back.iloc[0]
        fp = svi[(svi["expiry"] == f0["expiry"]) & (svi["date"] >= d0)].set_index("date")
        bp = svi[(svi["expiry"] == b0["expiry"]) & (svi["date"] >= d0)].set_index("date")
        days = fp.index.intersection(bp.index)
        if len(days) < 10 or days[0] != d0 or days[-1] >= f0["expiry"]:
            continue
        out.append({"date": d0, "front": fp.loc[days], "back": bp.loc[days],
                    "expiry": f0["expiry"], "settle": float(nifty.loc[f0["expiry"]]),
                    "back_last": bp.loc[days[-1]]})
    return out


def run_month(m, method="futures", r=R):
    """Hedged P&L (Rs per unit of NIFTY, at the front expiry) and costs for one month."""
    f, b = m["front"], m["back"]
    d0 = m["date"]
    K = float(f["F"].iloc[0])
    T1 = float(f["T"].iloc[0])
    t = ((f.index - d0).days / 365).to_numpy()
    vol1 = svi_vol(f.iloc[0], 0.0)  # sold at-the-money vol, held for the hedge (finding 2)
    prem = float(sim.black76(K, K, T1, r, vol1, "call"))
    grow = lambda tt: math.exp(r * (T1 - tt))  # noqa: E731
    cash = prem * grow(0.0)
    held_fut, held_opt, cost = 0.0, 0.0, 0.0
    n_days = len(f)
    for i in range(n_days):
        F1, F2 = float(f["F"].iloc[i]), float(b["F"].iloc[i])
        T1i, T2i = T1 - t[i], float(b["T"].iloc[i])
        s2 = svi_vol(b.iloc[i], math.log(K / F2))
        p2 = float(sim.black76(F2, K, T2i, r, s2, "call"))
        g1 = float(sim.gamma(F1, K, T1i, r, vol1))
        g2 = float(sim.gamma(F2, K, T2i, r, s2)) * (F2 / F1) ** 2
        d1 = float(sim.delta(F1, K, T1i, r, vol1, "call"))
        d2 = float(sim.delta(F2, K, T2i, r, s2, "call")) * (F2 / F1)
        if method == "dynamic" or (method == "static" and i == 0):
            new = g1 / g2 if g2 > 0 else 0.0
            if new != held_opt:
                trade = new - held_opt
                cost += option_cost(trade, p2) * grow(t[i])
                cash -= trade * p2 * grow(t[i])
                held_opt = new
        target = d1 - held_opt * d2  # futures needed: short call delta minus the back option's
        if target != held_fut:
            cost += trade_cost(target - held_fut, F1) * grow(t[i])
            held_fut = target
        nxt = float(f["F"].iloc[i + 1]) if i + 1 < n_days else m["settle"]
        t_next = t[i + 1] if i + 1 < n_days else T1
        cash += held_fut * (nxt - F1) * grow(t_next)
    # front expiry: pay the call, sell the back option at that day's price (last slice rolled on)
    pay = max(m["settle"] - K, 0.0)
    bl = m["back_last"]
    F2e = float(bl["F"]) * m["settle"] / float(f["F"].iloc[-1])
    T2e = max(float(bl["T"]) - (T1 - t[-1]), 1e-6)
    p2e = float(sim.black76(F2e, K, T2e, r, svi_vol(bl, math.log(K / F2e)), "call"))
    if held_opt:
        cost += option_cost(-held_opt, p2e)
        cash += held_opt * p2e
    return {"pnl": cash - pay - cost, "cost": cost, "n_back": held_opt, "F0": K}

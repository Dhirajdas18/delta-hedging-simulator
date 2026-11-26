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

from dhs import sim
from dhs.real import R, svi_vol
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
    """Each month-end: the front path over every trading day to expiry (`real.forward_path`),
    the back expiry's latest slice on or before each day, and the back slice on the settlement
    day itself (it exists: the back expiry is still a month away)."""
    from dhs.real import forward_path, load, month_ends

    svi, nifty = load()
    out = []
    for d0 in month_ends(svi):
        g = svi[(svi["date"] == d0) & svi["monthly"]].sort_values("days")
        front = g[g["days"].between(20, 45)]
        if front.empty:
            continue
        f0 = front.iloc[0]
        back = g[g["expiry"] > f0["expiry"]]
        if back.empty:
            continue
        b0 = back.iloc[0]
        fp = forward_path(svi[(svi["expiry"] == f0["expiry"]) & (svi["date"] >= d0)], nifty, d0,
                          f0["expiry"])
        if fp is None:
            continue
        days, F1, sd, settle, _ = fp
        bs = svi[(svi["expiry"] == b0["expiry"]) & (svi["date"] >= d0)].set_index("date")
        bs = bs.sort_index()
        if d0 not in bs.index or sd not in bs.index:
            continue
        # latest back slice on or before each day (rolled to the day in run_month)
        idx = bs.index.searchsorted(days, side="right") - 1
        out.append({"date": d0, "days": days, "F1": F1, "T1": float(f0["T"]),
                    "K": float(f0["F"]), "vol1": svi_vol(f0, 0.0), "settle": settle,
                    "back": bs.iloc[idx].reset_index(),  # keeps each slice's own date
                    "back_exp": b0["expiry"], "back_settle": bs.loc[sd], "settle_day": sd})
    return out


def _back(row, day, back_exp, F1_now, F1_then):
    """The back option's forward, time and SVI row on `day`, from a slice taken on or before it:
    time recomputed to the day, forward moved with the front forward."""
    T2 = max((back_exp - day).days / 365, 1e-6)
    return float(row["F"]) * F1_now / F1_then, T2


def run_month(m, method="futures", r=R):
    """Hedged P&L (Rs per unit of NIFTY, at the front expiry) and costs for one month."""
    d0, K, T1, vol1 = m["date"], m["K"], m["T1"], m["vol1"]
    days, F1 = m["days"], m["F1"]
    t = ((days - d0).days / 365).to_numpy()
    prem = float(sim.black76(F1[0], K, T1, r, vol1, "call"))
    grow = lambda tt: math.exp(r * (T1 - tt))  # noqa: E731
    cash = prem * grow(0.0)
    held_fut, held_opt, cost = 0.0, 0.0, 0.0
    f1_at = dict(zip(days, F1, strict=True))
    for i, day in enumerate(days):
        brow = m["back"].iloc[i]
        then = brow["date"] if "date" in brow else day
        F2, T2i = _back(brow, day, m["back_exp"], F1[i], f1_at.get(then, F1[i]))
        s2 = svi_vol(brow, math.log(K / F2)) if T2i > 0 else vol1
        p2 = float(sim.black76(F2, K, T2i, r, s2, "call"))
        T1i = T1 - t[i]
        g1 = float(sim.gamma(F1[i], K, T1i, r, vol1))
        g2 = float(sim.gamma(F2, K, T2i, r, s2)) * (F2 / F1[i]) ** 2
        d1 = float(sim.delta(F1[i], K, T1i, r, vol1, "call"))
        d2 = float(sim.delta(F2, K, T2i, r, s2, "call")) * (F2 / F1[i])
        if method == "dynamic" or (method == "static" and i == 0):
            new = g1 / g2 if g2 > 0 else 0.0
            if new != held_opt:
                trade = new - held_opt
                cost += option_cost(trade, p2) * grow(t[i])
                cash -= trade * p2 * grow(t[i])
                held_opt = new
        target = d1 - held_opt * d2  # futures: short call's delta less the back option's
        if target != held_fut:
            cost += trade_cost(target - held_fut, F1[i]) * grow(t[i])
            held_fut = target
        nxt = F1[i + 1] if i + 1 < len(days) else m["settle"]
        t_next = t[i + 1] if i + 1 < len(days) else T1
        cash += held_fut * (nxt - F1[i]) * grow(t_next)
    pay = max(m["settle"] - K, 0.0)
    if held_opt:  # sell the back option on the settlement day at that day's own smile
        bs = m["back_settle"]
        p2e = float(sim.black76(float(bs["F"]), K, float(bs["T"]), r,
                                svi_vol(bs, math.log(K / float(bs["F"]))), "call"))
        cost += option_cost(-held_opt, p2e)
        cash += held_opt * p2e
    return {"pnl": cash - pay - cost, "cost": cost, "n_back": held_opt, "F0": F1[0]}

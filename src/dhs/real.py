"""Selling and hedging real NIFTY options, month by month.

At the last trading day of each month, the monthly expiry 20-45 days out is taken from
implied-vol-surface's daily SVI table (its forward F and the SVI volatility at the strike), an
option is sold at Black-76 with that volatility, and delta-hedged at each trading day's close with
that expiry's futures (the slice's forward each day), the delta always computed at the sold
volatility with the actual calendar time left. At expiry the option pays out on NIFTY's close.

The hedged P&L at expiry (all cash moved to expiry at rate r):
    premium e^{rT} - payoff + sum_t delta_t (F_{t+1} - F_t) e^{r (T - t_{t+1})}.
Two predictions of it:
- the volatility gap (path-free): Black-76 price at the sold vol minus at the month's realised vol,
  both at the start, moved to expiry;
- the gamma sum (path-aware): sum_t 1/2 Gamma_t F_t^2 (sigma^2 dt_t - (dF_t / F_t)^2) e^{r(T - t)},
  with Gamma at the sold vol, dt in calendar years; days near the strike, where gamma is high,
  count more.
Set DHS_IV_DIR to implied-vol-surface (default ~/quant-learn/implied-vol-surface) and DHS_LAKE_DIR
to the local lake (default /path/to/lake) for NIFTY's close.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dhs import sim

R = 0.065
PARAMS = ("a", "b", "rho", "m", "sigma")


def iv_dir() -> Path:
    return Path(os.environ.get("DHS_IV_DIR", Path.home() / "quant-learn" / "implied-vol-surface"))


def lake_dir() -> Path:
    return Path(os.environ.get("DHS_LAKE_DIR") or os.environ.get("IV_LAKE_DIR")
                or Path.home() / "data" / "lake")


def svi_vol(row, k):
    a, b, rho, m, s = (row[x] for x in PARAMS)
    w = a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + s * s))
    return math.sqrt(max(w, 1e-12) / row["T"])


def hedge_path(F, t, K, T, r, sigma, kind, payoff_price, sigma_hedge=None):
    """Hedged P&L of a short option sold at `sigma` on one real forward path F at times t (years
    from the start; t[0] = 0, last t <= T), settled on `payoff_price`, the delta at `sigma_hedge`
    (default: the sold vol); plus the gamma-sum prediction (at the hedge vol, plus the premium gap
    when the two vols differ)."""
    F, t = np.asarray(F, float), np.asarray(t, float)
    sh = sigma if sigma_hedge is None else sigma_hedge
    prem = float(sim.black76(F[0], K, T, r, sigma, kind))
    left = T - t[:-1]
    d = sim.delta(F[:-1], K, left, r, sh, kind)
    g = sim.gamma(F[:-1], K, left, r, sh)
    dF = np.diff(F)
    grow = np.exp(r * (T - t[1:]))
    hedge = float(np.sum(d * dF * grow))
    pay = max(payoff_price - K, 0.0) if kind == "call" else max(K - payoff_price, 0.0)
    pnl = prem * math.exp(r * T) - pay + hedge
    dt = np.diff(t)
    gsum = float(np.sum(0.5 * g * F[:-1] ** 2 * (sh**2 * dt - (dF / F[:-1]) ** 2)
                        * np.exp(r * (T - t[:-1]))))
    gsum += (prem - float(sim.black76(F[0], K, T, r, sh, kind))) * math.exp(r * T)
    return pnl, gsum, prem


def trailing_vols(nifty: pd.Series, d0) -> dict:
    """NIFTY's volatility known at the close of d0 (a year of 252 trading days): the last 20 daily
    log returns, and EWMA (lambda 0.94) of all returns to d0."""
    r = np.log(nifty.loc[:d0]).diff().dropna()
    return {"hist20": float(np.sqrt((r.iloc[-20:] ** 2).mean() * 252)),
            "ewma": float(np.sqrt((r**2).ewm(alpha=0.06, adjust=False).mean().iloc[-1] * 252))}


def months(strikes=(("call", 1.0), ("put", 0.95)), r=R) -> pd.DataFrame:
    """One row per month-end and option: sold vol, realised vol, hedged P&L and its predictions,
    and the hedged P&L with the delta at other volatilities (`pnl_<name>`): NIFTY's last-20-day
    and EWMA volatility at the start, and the month's realised volatility (hindsight)."""
    svi = pd.read_parquet(iv_dir() / "data" / "derived" / "svi_daily.parquet")
    svi = svi[svi["converged"]]
    nifty = pd.read_parquet(lake_dir() / "market.parquet").set_index("date")["nifty"].dropna()
    dates = pd.DatetimeIndex(sorted(svi["date"].unique()))
    ends = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max()
    rows = []
    for d0 in ends:
        g = svi[(svi["date"] == d0) & svi["monthly"] & svi["days"].between(20, 45)]
        if g.empty:
            continue
        s0 = g.sort_values("days").iloc[0]
        exp = s0["expiry"]
        path = svi[(svi["expiry"] == exp) & (svi["date"] >= d0)].sort_values("date")
        if exp not in nifty.index or len(path) < 10:
            continue
        F = path["F"].to_numpy()
        t = ((path["date"] - d0).dt.days / 365).to_numpy()
        T = float(s0["T"])
        if t[-1] > T:
            continue
        settle = float(nifty.loc[exp])
        # realised volatility of the forward over the month, calendar-time annualised
        lr = np.diff(np.log(np.r_[F, settle]))
        realised = math.sqrt(np.sum(lr**2) / T)
        for kind, m in strikes:
            K = float(s0["F"]) * m
            vol = svi_vol(s0, math.log(K / s0["F"]))
            Fp, tp = np.r_[F, settle], np.r_[t, T]
            pnl, gsum, prem = hedge_path(Fp, tp, K, T, r, vol, kind, settle)
            gap = (prem - float(sim.black76(F[0], K, T, r, realised, kind))) * math.exp(r * T)
            pay = max(settle - K, 0.0) if kind == "call" else max(K - settle, 0.0)
            alt = trailing_vols(nifty, d0) | {"realised": realised}
            extra = {f"pnl_{nm}": hedge_path(Fp, tp, K, T, r, vol, kind, settle, sh)[0]
                     for nm, sh in alt.items()}
            extra |= {f"vol_{nm}": sh for nm, sh in alt.items() if nm != "realised"}
            rows.append({**extra, "date": d0, "expiry": exp, "kind": kind, "moneyness": m,
                         "F0": F[0],
                         "K": K, "T": T, "sold_vol": vol, "realised": realised, "premium": prem,
                         "pnl": pnl, "gamma_sum": gsum, "vol_gap": gap,
                         "unhedged": prem * math.exp(r * T) - pay, "days": len(path)})
    return pd.DataFrame(rows)

"""Selling and hedging single-stock options at month-ends, against NIFTY.

At each month-end in implied-vol-surface's stock option file (NSE bhavcopy, local), for every
stock with a monthly expiry 15-45 days out: the forward F0 is the stock future's settlement
price; the at-the-money strike is the one nearest F0 with a traded call and put; the sold
volatility is the mean of their Black-76 implied volatilities. A call at that strike is sold at
that volatility and hedged at every close with the stock's price (split-adjusted closes from the
the NSE bhavcopy archive, scaled to F0; carry over the month ignored) until expiry, settled on the expiry
day's close. Option symbols are mapped to current NSE symbols (the archive files a company's
history under its latest symbol) with the archive's symbol-change list.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from dhs import sim
from dhs.real import R, hedge_path, iv_dir, lake_dir


def implied_vol(price, F, K, T, r, kind):
    lo, hi = 1e-4, 5.0
    f = lambda v: float(sim.black76(F, K, T, r, v, kind)) - price  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-8)


def current_symbols() -> dict[str, str]:
    """Old symbol -> latest symbol, following chains, from the archive's symbol-change list."""
    ch = pd.read_csv(lake_dir() / "_ref" / "symbolchange.csv")
    ch.columns = [c.strip().upper() for c in ch.columns]
    ch["d"] = pd.to_datetime(ch["EFFECTIVE DATE"].str.strip(), format="%d-%b-%Y", errors="coerce")
    step = {str(r[1]).strip(): str(r[2]).strip()
            for r in ch.sort_values("d").itertuples(index=False)}
    out = {}
    for old in step:
        cur, seen = old, set()
        while cur in step and cur not in seen:
            seen.add(cur)
            cur = step[cur]
        out[old] = cur
    return out


def atm_cases(r=R) -> pd.DataFrame:
    """One row per month-end and stock: forward, expiry, T, at-the-money strike and sold vol."""
    so = pd.read_parquet(iv_dir() / "data" / "options" / "stock_options.parquet")
    rows = []
    for (d, sym), g in so.groupby(["date", "symbol"]):
        fut = g[g["type"] == "FUT"]
        fut = fut[((fut["expiry"] - d).dt.days).between(15, 45)]
        if fut.empty:
            continue
        f = fut.sort_values("expiry").iloc[0]
        exp, F0 = f["expiry"], float(f["settle"])
        o = g[(g["expiry"] == exp) & g["type"].isin(["CE", "PE"]) & (g["contracts"] > 0)
              & (g["price"] > 0.05)]
        both = o.pivot_table(index="strike", columns="type", values="price", aggfunc="last")
        if not {"CE", "PE"} <= set(both.columns) or not F0 > 0:
            continue
        both = both.dropna()
        if both.empty:
            continue
        K = float(both.index[np.argmin(np.abs(both.index.to_numpy() - F0))])
        if abs(K / F0 - 1) > 0.05:
            continue
        T = (exp - d).days / 365
        vols = [implied_vol(float(both.at[K, "CE"]), F0, K, T, r, "call"),
                implied_vol(float(both.at[K, "PE"]), F0, K, T, r, "put")]
        if not all(np.isfinite(vols)):
            continue
        rows.append({"date": d, "symbol": sym, "expiry": exp, "T": T, "F0": F0, "K": K,
                     "vol": float(np.mean(vols))})
    return pd.DataFrame(rows)


def hedge_months(cases: pd.DataFrame, r=R) -> pd.DataFrame:
    """Hedged and unhedged P&L (% of F0) of each case, with the month's realised vol and its
    largest daily move."""
    names = current_symbols()
    cases = cases.assign(sym=cases["symbol"].map(lambda s: names.get(s, s)))
    p = pd.read_parquet(lake_dir() / "panel.parquet", columns=["date", "symbol", "adj_close"])
    p = p[p["symbol"].isin(cases["sym"].unique())]
    px = p.pivot_table(index="date", columns="symbol", values="adj_close")
    out = []
    for c in cases.itertuples():
        if c.sym not in px:
            continue
        s = px[c.sym].loc[c.date: c.expiry].dropna()
        if len(s) < 8 or s.index[0] != c.date or s.index[-1] != c.expiry:
            continue
        F = c.F0 * (s / s.iloc[0]).to_numpy()
        t = ((s.index - c.date).days / 365).to_numpy()
        pnl, gsum, prem = hedge_path(F, t, c.K, c.T, r, c.vol, "call", float(F[-1]))
        lr = np.diff(np.log(F))
        out.append({"date": c.date, "symbol": c.sym, "vol": c.vol, "F0": c.F0,
                    "pnl": 100 * pnl / c.F0, "gamma_sum": 100 * gsum / c.F0,
                    "unhedged": 100 * (prem * math.exp(r * c.T) - max(F[-1] - c.K, 0)) / c.F0,
                    "realised": float(np.sqrt(np.sum(lr**2) / c.T)),
                    "max_move": float(np.max(np.abs(lr)))})
    return pd.DataFrame(out)

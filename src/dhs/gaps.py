"""What a hedge set at the close can't catch: NIFTY's overnight gaps and jumps.

NIFTY's open and close come from the NSE bhavcopy archive's daily index files (`nifty_ohlc`). Each
real month's daily forward move r_t = ln(F_{t+1} / F_t) is split into the overnight gap
g_t = ln(open_{t+1} / close_t) of the index and the rest, the trading-day move. The futures' own
open isn't in the data, so the forward is assumed to gap by the same proportion as the index.

`hedge_twice`: the same hedge, rebalanced at the open as well as the close (the forward at the
open taken as F_t x open_{t+1} / close_t). Hedging at the open can only remove the trading-day
part; the overnight gap stays.
"""

import glob
import math

import numpy as np
import pandas as pd

from dhs import sim
from dhs.real import lake_dir

NAMES = {"s&p cnx nifty", "cnx nifty", "nifty 50"}


def nifty_ohlc() -> pd.DataFrame:
    files = sorted(glob.glob(str(lake_dir() / "index" / "*.parquet")))
    d = pd.concat(pd.read_parquet(f) for f in files)
    d = d[d["index"].str.strip().str.lower().isin(NAMES)]
    return d.set_index("date")[["open", "high", "low", "close"]].sort_index()


def hedge_twice(F, t, gap, K, T, r, sigma, kind, settle):
    """Hedged P&L of a short option rebalanced at each close and at the next open. `gap[i]` is
    the open/close ratio between close i and the next open (1.0 = no gap)."""
    F, t, gap = np.asarray(F, float), np.asarray(t, float), np.asarray(gap, float)
    prem = float(sim.black76(F[0], K, T, r, sigma, kind))
    hedge = 0.0
    for i in range(len(F) - 1):
        left = T - t[i]
        d_close = float(sim.delta(F[i], K, left, r, sigma, kind))
        f_open = F[i] * gap[i]
        t_open = t[i] + min(1 / 365, 0.5 * (t[i + 1] - t[i]))  # the next morning
        d_open = float(sim.delta(f_open, K, max(T - t_open, 1e-8), r, sigma, kind))
        hedge += d_close * (f_open - F[i]) * math.exp(r * (T - t_open))
        hedge += d_open * (F[i + 1] - f_open) * math.exp(r * (T - t[i + 1]))
    pay = max(settle - K, 0.0) if kind == "call" else max(K - settle, 0.0)
    return prem * math.exp(r * T) - pay + hedge


def month_gaps(case, ohlc: pd.DataFrame) -> np.ndarray | None:
    """Open/close ratios for each step of a `month_cases` path (index gap, by date)."""
    dates = pd.DatetimeIndex([case["date"] + pd.Timedelta(days=round(x * 365)) for x in case["t"]])
    c = ohlc["close"].reindex(dates[:-1])
    o = ohlc["open"].reindex(dates[1:])
    if c.isna().any() or o.isna().any():
        return None
    return o.to_numpy() / c.to_numpy()

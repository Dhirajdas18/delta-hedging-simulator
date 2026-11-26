"""When to rebalance a delta hedge, after Indian trading costs, on real NIFTY months.

Costs of trading NIFTY futures at the rates from 1 Apr 2026 (black-scholes-pricer's cost table,
zerodha.com/charges, checked 2026-11-23), applied to every year ("what it would cost now"): STT
0.05% of sale value; NSE transaction charge 0.00183% and SEBI fee 0.0001% of traded value; stamp
duty 0.002% of buy value; brokerage Rs 20 an order; GST 18% on brokerage + NSE + SEBI. The book
is 10 lots of 65 (650 units of NIFTY), so the Rs 20 order fee is spread over 650 units; futures
are treated as divisible (no lot rounding); the bid-ask spread is left out. The option's own
selling cost is the same for every rule and left out.

Rules, all deciding at each trading day's close:
- time: rebalance to the target delta every k trading days;
- band: rebalance to the target when |target - held| > h;
- Whalley-Wilmott: rebalance to the nearer edge of a band of half-width
  (1.5 e^{-r tau} lambda F Gamma^2 / A)^{1/3}, lambda the one-way cost rate, A a risk-aversion
  setting; the band widens with costs and with gamma (it grows as Gamma^{2/3});
- Leland: rebalance every day, at a delta computed with sigma^2 (1 + Le),
  Le = sqrt(2/pi) k / (sigma sqrt(dt)), k the round-trip cost rate (a short option raises it).
"""

import math

import numpy as np

from dhs import sim

UNITS = 650  # 10 lots of 65
NSE, SEBI, STAMP, STT, BROKER, GST = 0.0000183, 0.000001, 0.00002, 0.0005, 20.0, 0.18
RATE_BUY = NSE * (1 + GST) + SEBI * (1 + GST) + STAMP
RATE_SELL = NSE * (1 + GST) + SEBI * (1 + GST) + STT
ORDER = BROKER * (1 + GST) / UNITS  # Rs per unit of NIFTY per order


def trade_cost(units, F):
    """Cost (Rs per unit of NIFTY hedged) of buying (units > 0) or selling futures at F."""
    if units == 0:
        return 0.0
    rate = RATE_BUY if units > 0 else RATE_SELL
    return abs(units) * F * rate + ORDER


def hedge_rule(F, t, K, T, r, sigma, kind, settle, rule="time", param=1.0):
    """Hedged P&L after futures costs of a short option sold at `sigma` on a real path, and the
    total cost and number of trades. All cash moved to expiry at r."""
    F, t = np.asarray(F, float), np.asarray(t, float)
    prem = float(sim.black76(F[0], K, T, r, sigma, kind))
    lam = 0.5 * (RATE_BUY + RATE_SELL)
    held, hedge, cost, trades = 0.0, 0.0, 0.0, 0
    n = len(F) - 1
    for i in range(n):
        left = T - t[i]
        if rule == "leland":
            le = math.sqrt(2 / math.pi) * 2 * lam / (sigma * math.sqrt(max(t[i + 1] - t[i], 1e-6)))
            target = float(sim.delta(F[i], K, left, r, sigma * math.sqrt(1 + le), kind))
            new = target
        else:
            target = float(sim.delta(F[i], K, left, r, sigma, kind))
            if rule == "time":
                new = target if i % int(param) == 0 else held
            elif rule == "band":
                new = target if (i == 0 or abs(target - held) > param) else held
            else:  # Whalley-Wilmott: move to the nearer band edge
                g = float(sim.gamma(F[i], K, left, r, sigma))
                h = (1.5 * math.exp(-r * left) * lam * F[i] * g * g / param) ** (1 / 3)
                new = target if i == 0 else min(max(held, target - h), target + h)
        if new != held:
            c = trade_cost(new - held, F[i])
            cost += c * math.exp(r * (T - t[i]))
            trades += 1
            held = new
        hedge += held * (F[i + 1] - F[i]) * math.exp(r * (T - t[i + 1]))
    pay = max(settle - K, 0.0) if kind == "call" else max(K - settle, 0.0)
    return prem * math.exp(r * T) - pay + hedge - cost, cost, trades

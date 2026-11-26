"""Delta hedging on a forward: Black-76 prices and Greeks, simulated forward paths (geometric
Brownian motion with optional Merton jumps) and the P&L of selling one option and hedging it with
futures.

Conventions: F forward (futures) price, K strike, T years to expiry (calendar days / 365), r
continuously compounded rate a year, sigma volatility a year, kind "call" or "put". Money is in
rupees per unit of the underlying and, in hedge_pnl, valued at expiry.
"""

import numpy as np
from scipy.stats import norm


def _kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _d1(F, K, T, sigma):
    return (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))


def black76(F, K, T, r, sigma, kind="call"):
    """Black-76 price of a European option on a forward F."""
    _kind(kind)
    d1 = _d1(F, K, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if kind == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def delta(F, K, T, r, sigma, kind="call"):
    """dPrice/dF: the number of futures that hedge one option (negative for a put)."""
    _kind(kind)
    return np.exp(-r * T) * (norm.cdf(_d1(F, K, T, sigma)) - (kind == "put"))


def gamma(F, K, T, r, sigma):
    """d2Price/dF2, the same for calls and puts."""
    return np.exp(-r * T) * norm.pdf(_d1(F, K, T, sigma)) / (F * sigma * np.sqrt(T))


def vega(F, K, T, r, sigma):
    """dPrice/dsigma (per 1.00 of volatility, not per percentage point)."""
    return np.exp(-r * T) * F * norm.pdf(_d1(F, K, T, sigma)) * np.sqrt(T)


def paths_from_shocks(F0, sigma, T, z, jumps=None, z_jump=None, jump_intensity=0.0,
                      jump_mean=0.0, jump_sd=0.0):
    """Forward paths from given shocks: z standard normals (n_paths, steps); with jumps, the
    number of jumps per step and one standard normal per step for their total size.

    Log steps are (-sigma^2/2 - lambda k) dt + sigma sqrt(dt) z + sum of the step's jumps, each
    jump normal in logs with mean jump_mean and sd jump_sd, and k = E[e^jump] - 1, so the forward
    is a martingale with or without jumps.
    """
    z = np.asarray(z, dtype=float)
    dt = T / z.shape[1]
    k = np.exp(jump_mean + 0.5 * jump_sd**2) - 1
    x = (-0.5 * sigma**2 - jump_intensity * k) * dt + sigma * np.sqrt(dt) * z
    if jumps is not None:
        x = x + (jumps * jump_mean + np.sqrt(jumps) * jump_sd * z_jump)
    logs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(x, axis=1)], axis=1)
    return F0 * np.exp(logs)


def simulate_paths(F0, sigma_real, T, steps, n_paths, seed=None, jump_intensity=0.0,
                   jump_mean=0.0, jump_sd=0.0):
    """GBM forward paths with optional compensated Merton jumps, shape (n_paths, steps + 1).
    jump_intensity is jumps a year; jump_mean and jump_sd are in log terms."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, steps))
    jumps = z_jump = None
    if jump_intensity > 0:
        jumps = rng.poisson(jump_intensity * T / steps, (n_paths, steps))
        z_jump = rng.standard_normal((n_paths, steps))
    return paths_from_shocks(F0, sigma_real, T, z, jumps, z_jump, jump_intensity, jump_mean,
                             jump_sd)


def hedge_pnl(paths, K, T, r, sigma_hedge, sigma_sold, kind="call", rebalance="time", every=1,
              band=None, cost_fixed=0.0, cost_pct=0.0) -> dict:
    """Sell one option at Black-76 with sigma_sold and delta-hedge it with futures, path by path.

    Accounting, everything valued at expiry: the premium goes into cash earning r; futures need no
    money up front and their gain over each step is credited to cash at the step's end, then earns
    r; costs are paid from cash when trading. The hedge holds Black-76 delta at sigma_hedge,
    rebalanced on the paths' time grid either every `every` steps (rebalance="time") or whenever
    |delta - held| > band (rebalance="band"); the first step always trades. A trade costs
    cost_pct x |units traded| x F, plus cost_fixed per trade. Futures are cash-settled at expiry
    with no closing trade, and the option's payoff is paid then.

    Returns arrays over paths: pnl (hedged, after costs), trades (the opening trade included),
    costs, unhedged (P&L of the bare short), gamma (the sum over steps of
    1/2 Gamma (sigma_hedge^2 dt F^2 - dF^2), each grown to expiry), edge (premium at sigma_sold
    minus at sigma_hedge, grown to expiry: the rest of the theoretical P&L) and held (futures held
    over each step, (n_paths, steps)). pnl is close to edge + gamma before costs.
    """
    _kind(kind)
    if rebalance not in ("time", "band"):
        raise ValueError(f"rebalance must be 'time' or 'band', got {rebalance!r}")
    if rebalance == "band" and not (band is not None and band > 0):
        raise ValueError("the delta band must be above zero")
    if rebalance == "time" and not (int(every) == every and every >= 1):
        raise ValueError(f"every must be a whole number of steps of at least 1, got {every}")
    F = np.asarray(paths, dtype=float)
    n, steps = F.shape[0], F.shape[1] - 1
    dt = T / steps
    grow = np.exp(r * (T - np.arange(steps + 1) * dt))  # rupees at step i -> rupees at expiry
    premium = black76(F[:, 0], K, T, r, sigma_sold, kind)
    h = np.zeros(n)
    held = np.empty((n, steps))
    trades, costs, hedge, gam = (np.zeros(n) for _ in range(4))
    for i in range(steps):
        tau = T - i * dt
        target = delta(F[:, i], K, tau, r, sigma_hedge, kind)
        if rebalance == "time":
            trade = np.full(n, i % every == 0)
        else:
            trade = (np.abs(target - h) > band) | (i == 0)
        new = np.where(trade, target, h)
        costs += (trade * cost_fixed + cost_pct * np.abs(new - h) * F[:, i]) * grow[i]
        trades += trade
        h = held[:, i] = new
        dF = F[:, i + 1] - F[:, i]
        hedge += h * dF * grow[i + 1]
        g = gamma(F[:, i], K, tau, r, sigma_hedge)
        gam += 0.5 * g * (sigma_hedge**2 * dt * F[:, i] ** 2 - dF**2) * grow[i + 1]
    payoff = np.maximum(F[:, -1] - K, 0) if kind == "call" else np.maximum(K - F[:, -1], 0)
    unhedged = premium * grow[0] - payoff
    edge = (premium - black76(F[:, 0], K, T, r, sigma_hedge, kind)) * grow[0]
    return {
        "pnl": unhedged + hedge - costs,
        "trades": trades,
        "costs": costs,
        "unhedged": unhedged,
        "gamma": gam,
        "edge": edge,
        "held": held,
    }

"""README findings, one function per topic: `dhs report <topic>` writes charts to out/."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dhs import plots


def report_path(out_dir: Path) -> dict:
    """Why hedged P&L depends on the path. Simulation: a 30-day at-the-money call sold at the
    volatility that then happens (15%), hedged every 1, 2, 3, 5, 10 or 15 days: how the spread of
    hedged P&L shrinks with more rebalancing. Real NIFTY: at each month-end, sell the monthly
    at-the-money call and the 95% put at their implied volatility, hedge daily with futures to
    expiry, and compare the hedged P&L with two predictions: the volatility gap (sold vol against
    the month's realised vol) and the path-aware gamma sum. Needs DHS_IV_DIR and the lake."""
    from dhs import sim
    from dhs.real import R, months

    res = {}
    F0, T, v = 25000.0, 30 / 365, 0.15
    paths = sim.simulate_paths(F0, v, T, 30, 20000, seed=7)
    every = (1, 2, 3, 5, 10, 15)
    sd = {}
    for e in every:
        out = sim.hedge_pnl(paths, F0, T, R, v, v, "call", every=e)
        sd[e] = float(np.std(out["pnl"]))
    rebal = np.array([30 / e for e in every])
    slope = float(np.polyfit(np.log(rebal), np.log([sd[e] for e in every]), 1)[0])
    prem = float(sim.black76(F0, F0, T, R, v))
    res[f"simulation: 30-day ATM call worth Rs {prem:.0f}, sold and realised at 15%"] = (
        "hedged P&L spread (Rs): " + ", ".join(f"every {e} day{'s' if e > 1 else ''} {sd[e]:.1f}"
                                               for e in every)
        + f"; log-log slope against the number of rebalances {slope:.2f} (theory -0.5); "
        f"unhedged spread Rs {float(np.std(out['unhedged'])):.0f}"
    )

    d = months(r=R)
    res["real months"] = (
        f"{d['date'].nunique()} month-ends ({d['date'].min():%b %Y}-{d['date'].max():%b %Y}), "
        f"{d['days'].median():.0f} trading days to expiry (median); sold at the SVI volatility at "
        f"the strike, hedged at each close with the expiry's futures"
    )
    fits = {}
    for kind, g in d.groupby("kind"):
        pnl = 100 * g["pnl"] / g["F0"]
        gs = 100 * g["gamma_sum"] / g["F0"]
        vg = 100 * g["vol_gap"] / g["F0"]
        r2g, r2v = np.corrcoef(gs, pnl)[0, 1] ** 2, np.corrcoef(vg, pnl)[0, 1] ** 2
        fits[kind] = (pnl, gs, vg, r2g, r2v)
        name = "at-the-money call" if kind == "call" else "95% put"
        up = g["sold_vol"] > g["realised"]
        res[f"{name}"] = (
            f"hedged P&L mean {pnl.mean():+.3f}% of NIFTY, spread {pnl.std():.3f}% (unhedged "
            f"{(100 * g['unhedged'] / g['F0']).std():.2f}%); share of its month-to-month variation "
            f"explained: gamma sum {r2g:.0%}, volatility gap {r2v:.0%}; implied above realised in "
            f"{up.mean():.0%} of months, yet a loss in {((g['pnl'] < 0) & up).sum()} of those "
            f"({((g['pnl'] < 0) & up).sum() / up.sum():.0%}); implied below realised yet a profit "
            f"in {((g['pnl'] > 0) & ~up).sum()} of {(~up).sum()}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(rebal, [sd[e] for e in every], color=plots.COLORS["hedged"], lw=2, marker="o")
    ref = sd[1] * (rebal / rebal[0]) ** -0.5
    ax.plot(rebal, ref, color=plots.COLORS["formula"], lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(rebal, [f"{x:g}" for x in rebal])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([70, 100, 150, 200, 250], ["70", "100", "150", "200", "250"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("hedge trades over the 30 days")
    ax.set_ylabel("spread of hedged P&L, Rs\n(dashed: halves with 4x the trades)")
    ax.set_title("Simulated: more trades, less noise", fontsize=10.5)
    pnl, gs, vg, r2g, r2v = fits["call"]
    lim = float(np.nanquantile(np.abs(np.r_[pnl, gs, vg]), 0.995)) * 1.05
    for ax, x, r2, lab in ((axes[1], vg, r2v, "the volatility gap alone"),
                           (axes[2], gs, r2g, "the day-by-day gamma sum")):
        ax.scatter(x, pnl, s=14, color=plots.COLORS["hedged"], alpha=0.7, lw=0)
        ax.plot([-lim, lim], [-lim, lim], color=plots.COLORS["formula"], lw=1, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"predicted by {lab}, % of NIFTY")
        ax.set_title(f"Real ATM calls: explains {r2:.0%}", fontsize=10.5)
    axes[1].set_ylabel("actual hedged P&L, % of NIFTY\n(one dot per month)")
    fig.suptitle("Hedged profit follows the day-by-day path, not just the month's volatility",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "path")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "path": report_path,
}

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


def report_hedge_vol(out_dir: Path) -> dict:
    """Which volatility to hedge at. Real NIFTY months as in finding 1 (sold at the implied
    volatility at the strike), the delta computed at: that implied volatility; NIFTY's last-20-day
    volatility at the start; EWMA (0.94) at the start; and the month's realised volatility, known
    only in hindsight. Nothing is fitted, so all months are out of sample; halves shown. Plus a
    simulation of hedging at the true volatility when it stays constant and when it changes within
    the month."""
    from dhs import sim
    from dhs.real import R, months

    res = {}
    d = months(r=R)
    meth = {"pnl": "implied (sold) vol", "pnl_hist20": "last 20 days' vol",
            "pnl_ewma": "EWMA vol", "pnl_realised": "the month's realised vol (hindsight)"}
    res["volatility medians"] = "; ".join(
        f"{k}: sold {g['sold_vol'].median():.1%}, last 20 days {g['vol_hist20'].median():.1%}, "
        f"EWMA {g['vol_ewma'].median():.1%}, realised {g['realised'].median():.1%}"
        for k, g in d.groupby("kind"))
    table = {}
    for kind, g in d.groupby("kind"):
        name = "at-the-money call" if kind == "call" else "95% put"
        for per, m in (("all", g["date"].dt.year > 0), ("2015-2020", g["date"].dt.year <= 2020),
                       ("2021-2026", g["date"].dt.year >= 2021)):
            x = g[m]
            pct = {c: 100 * x[c] / x["F0"] for c in meth}
            row = {c: (float(pct[c].mean()), float(pct[c].std()), float((x[c] < 0).mean()))
                   for c in meth}
            table[(kind, per)] = row
            key = f"{name}, {per} ({len(x)} months): mean / spread, % of NIFTY; losing months"
            res[key] = "; ".join(f"{meth[c]} {v[0]:+.3f} / {v[1]:.3f}, {v[2]:.0%}"
                                 for c, v in row.items())

    F0, T, steps = 25000.0, 30 / 365, 30
    paths = sim.simulate_paths(F0, 0.13, T, steps, 20000, seed=1)
    const = {lab: sim.hedge_pnl(paths, F0, T, R, sh, 0.15, "call")["pnl"]
             for lab, sh in (("implied", 0.15), ("true", 0.13))}
    z = np.random.default_rng(2).standard_normal((20000, steps))
    sig = np.r_[np.full(15, 0.08), np.full(15, 0.18)]
    dt = T / steps
    lr = -0.5 * sig**2 * dt + sig * np.sqrt(dt) * z
    paths2 = F0 * np.exp(np.c_[np.zeros(20000), np.cumsum(lr, 1)])
    avg = float(np.sqrt(np.mean(sig**2)))
    change = {lab: sim.hedge_pnl(paths2, F0, T, R, sh, 0.15, "call")["pnl"]
              for lab, sh in (("implied", 0.15), ("true", avg))}
    res["simulation, 30-day ATM call sold at 15%, hedged daily: spread of hedged P&L (Rs)"] = (
        f"constant 13% vol: hedged at 15% {np.std(const['implied']):.1f}, at the true 13% "
        f"{np.std(const['true']):.1f}; vol 8% then 18% (average {avg:.1%}): at 15% "
        f"{np.std(change['implied']):.1f}, at the average {np.std(change['true']):.1f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    cols = [plots.COLORS["hedged"], plots.BLUE_RAMP[2], plots.BLUE_RAMP[5],
            plots.COLORS["unhedged"]]
    labels = ["implied\n(sold)", "last\n20 days", "EWMA", "realised\n(hindsight)"]
    for ax, kind, title in ((axes[0], "call", "At-the-money calls"), (axes[1], "put", "95% puts")):
        vals = [table[(kind, "all")][c][1] for c in meth]
        ax.bar(range(4), vals, color=cols)
        for x, y in enumerate(vals):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=plots.TEXT)
        ax.set_xticks(range(4), labels)
        ax.set_title(f"{title}: hedged with the delta at…", fontsize=10.5)
    axes[0].set_ylabel("spread of monthly hedged P&L,\n% of NIFTY (141 months)")
    fig.suptitle("Hedging at the implied volatility left the least risk; hindsight's realised "
                 "volatility left the most", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "hedge_vol")
    plt.close(fig)
    return res


def report_rebalance(out_dir: Path) -> dict:
    """When to rebalance, after Indian futures costs at today's rates (dhs.rebalance): fixed
    times, delta bands, Whalley-Wilmott bands and Leland's cost-adjusted volatility, on finding 1's
    real NIFTY months. Each rule's setting is chosen on 2015-2020 as the cheapest one whose spread
    of monthly hedged P&L is no wider than daily hedging's (or, if none is, the one closest), then
    scored on 2021-2026."""
    import pandas as pd

    from dhs.real import R, month_cases
    from dhs.rebalance import hedge_rule

    res = {}
    grid = ([("time", k) for k in (1, 2, 3, 5, 10)]
            + [("band", h) for h in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)]
            + [("ww", a) for a in (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0)]
            + [("leland", 0.0)])
    rows = []
    for c in month_cases():
        for rule, par in grid:
            pnl, cost, trades = hedge_rule(c["F"], c["t"], c["K"], c["T"], R, c["vol"], c["kind"],
                                           c["settle"], rule, par)
            rows.append({"date": c["date"], "kind": c["kind"], "rule": rule, "param": par,
                         "pnl": 100 * pnl / c["F"][0], "cost": 100 * cost / c["F"][0],
                         "trades": trades})
    d = pd.DataFrame(rows)
    d["half"] = np.where(d["date"].dt.year <= 2020, "train", "test")
    agg = d.groupby(["kind", "half", "rule", "param"]).agg(
        mean=("pnl", "mean"), sd=("pnl", "std"), cost=("cost", "mean"), trades=("trades", "mean"))
    names = {"time": "every k days", "band": "delta band", "ww": "Whalley-Wilmott",
             "leland": "Leland (daily)"}
    chosen = {}
    for kind in ("call", "put"):
        tr = agg.loc[(kind, "train")]
        te = agg.loc[(kind, "test")]
        ref_tr, ref_te = tr.loc[("time", 1.0)], te.loc[("time", 1.0)]
        label = "at-the-money call" if kind == "call" else "95% put"
        res[f"{label}, daily hedging"] = (
            f"2015-2020: spread {ref_tr['sd']:.3f}%, cost {ref_tr['cost']:.4f}% of NIFTY a month, "
            f"{ref_tr['trades']:.0f} trades; 2021-2026: spread {ref_te['sd']:.3f}%, cost "
            f"{ref_te['cost']:.4f}%"
        )
        for rule in ("time", "band", "ww", "leland"):
            opts = tr.loc[rule]
            ok = opts[opts["sd"] <= ref_tr["sd"]]
            par = ok["cost"].idxmin() if len(ok) else (opts["sd"] - ref_tr["sd"]).abs().idxmin()
            chosen[(kind, rule)] = par
            a, b = tr.loc[(rule, par)], te.loc[(rule, par)]
            res[f"  {label}, {names[rule]} (setting {par:g}, chosen on 2015-2020)"] = (
                f"2015-2020 spread {a['sd']:.3f}%, cost {a['cost']:.4f}%; 2021-2026 spread "
                f"{b['sd']:.3f}% ({100 * (b['sd'] / ref_te['sd'] - 1):+.1f}% vs daily), cost "
                f"{b['cost']:.4f}% ({100 * (b['cost'] / ref_te['cost'] - 1):+.1f}%), "
                f"{b['trades']:.1f} trades, mean after costs {b['mean']:+.3f}%"
            )
        k5 = tr.loc[("band", 0.05)]
        k2 = tr.loc[("time", 2.0)]
        res[f"  {label}: a band of 0.05 against trading every 2 days (2015-2020)"] = (
            f"spread {k5['sd']:.3f}% vs {k2['sd']:.3f}%, "
            f"cost {k5['cost']:.4f}% vs {k2['cost']:.4f}%")

    full = d.groupby(["kind", "rule", "param"]).agg(sd=("pnl", "std"), cost=("cost", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    cols = {"time": plots.MUTED, "band": plots.COLORS["hedged"], "ww": plots.SERIES[3],
            "leland": plots.COLORS["unhedged"]}
    for ax, kind, title in ((axes[0], "call", "At-the-money calls"), (axes[1], "put", "95% puts")):
        for rule in ("time", "band", "ww", "leland"):
            g = full.loc[(kind, rule)].sort_values("cost")
            ax.plot(g["cost"], g["sd"], color=cols[rule], lw=2, marker="o", ms=5,
                    label=names[rule])
            p = chosen[(kind, rule)]
            ax.scatter([full.loc[(kind, rule, p), "cost"]], [full.loc[(kind, rule, p), "sd"]],
                       s=110, facecolors="none", edgecolors=plots.TEXT, lw=1.4, zorder=5)
        ax.set_xlabel("average hedging cost, % of NIFTY a month")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("spread of monthly hedged P&L,\n% of NIFTY (2015-2026)")
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Trading when the hedge drifts beat trading on a timetable (circled: settings "
                 "chosen on 2015-2020)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "rebalance")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "path": report_path,
    "hedge-vol": report_hedge_vol,
    "rebalance": report_rebalance,
}

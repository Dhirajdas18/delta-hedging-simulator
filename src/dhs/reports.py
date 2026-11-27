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
    axes[0].set_ylabel(f"spread of monthly hedged P&L,\n% of NIFTY ({d['date'].nunique()} months)")
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
            + [("ww", a) for a in (3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)]
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


def report_gaps(out_dir: Path) -> dict:
    """What a close-to-close hedge can't catch: NIFTY's overnight gaps (open against the previous
    close, the archive's index files, 2015-2026), their share of daily variance, the biggest
    gap days; finding 1's months hedged at each close and also at each open; and a simulation of
    hedging more often with and without jumps."""
    import pandas as pd

    from dhs import sim
    from dhs.gaps import hedge_twice, month_gaps, nifty_ohlc
    from dhs.real import R, hedge_path, month_cases

    res = {}
    o = nifty_ohlc()
    o = o[o.index >= "2015-01-01"]
    g = np.log(o["open"] / o["close"].shift()).dropna()
    i = np.log(o["close"] / o["open"]).reindex(g.index)
    tot = ((g + i) ** 2).sum()
    big = g.abs() > g.abs().quantile(0.99)
    res["NIFTY days"] = (
        f"{len(g):,} ({g.index.min():%b %Y}-{g.index.max():%b %Y}); overnight gap's share of "
        f"close-to-close variance {(g**2).sum() / tot:.1%}, trading day {(i**2).sum() / tot:.1%}, "
        f"cross term {(2 * g * i).sum() / tot:+.1%} (correlation {np.corrcoef(g, i)[0, 1]:+.2f})"
    )
    top = g.reindex(g.abs().sort_values().index[-5:][::-1])
    res["  biggest 1% of gaps"] = (
        f"{int(big.sum())} days carried {((g + i) ** 2)[big].sum() / tot:.1%} of all "
        "close-to-close variance; largest: "
        + ", ".join(f"{d:%d %b %Y} {100 * v:+.1f}%" for d, v in top.items())
    )
    rows = []
    for c in month_cases():
        gap = month_gaps(c, o)
        if gap is None:
            continue
        F = c["F"]
        once = hedge_path(F, c["t"], c["K"], c["T"], R, c["vol"], c["kind"], c["settle"])[0]
        twice = hedge_twice(F, c["t"], gap, c["K"], c["T"], R, c["vol"], c["kind"], c["settle"])
        r = np.diff(np.log(F))
        w = 0.5 * sim.gamma(F[:-1], c["K"], c["T"] - c["t"][:-1], R, c["vol"]) * F[:-1] ** 2
        rows.append({"kind": c["kind"], "once": 100 * once / F[0], "twice": 100 * twice / F[0],
                     "gamma_gap": float((w * np.log(gap) ** 2).sum()),
                     "gamma_all": float((w * r**2).sum())})
    d = pd.DataFrame(rows)
    for kind, x in d.groupby("kind"):
        name = "at-the-money call" if kind == "call" else "95% put"
        res[f"{name} ({len(x)} months with every open and close)"] = (
            f"gaps' share of the gamma-weighted squared moves "
            f"{x['gamma_gap'].sum() / x['gamma_all'].sum():.0%}; "
            f"hedged at each close: mean {x['once'].mean():+.3f}%, spread {x['once'].std():.3f}%; "
            f"also at each open: mean {x['twice'].mean():+.3f}%, spread {x['twice'].std():.3f}%"
        )
    T, F0 = 30 / 365, 25000.0
    per_day = (1, 2, 4, 8)
    sims = {}
    for lab, ji in (("no jumps", 0.0), ("jumps", 4.0)):
        sims[lab] = []
        for k in per_day:
            p = sim.simulate_paths(F0, 0.15, T, 30 * k, 8000, seed=3, jump_intensity=ji,
                                   jump_mean=-0.02, jump_sd=0.03)
            sims[lab].append(float(np.std(sim.hedge_pnl(p, F0, T, R, 0.15, 0.15, "call")["pnl"])))
    res["simulation, 30-day ATM call, hedges a day 1/2/4/8: spread of hedged P&L (Rs)"] = (
        "; ".join(f"{lab}: " + " / ".join(f"{v:.1f}" for v in vals) for lab, vals in sims.items())
        + " (jumps: 4 a year, average -2%, spread 3%)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    both = (g**2).sum() + (i**2).sum()  # the two parts' own sizes, so the bars add to 100%
    share = [(g**2).sum() / both * 100, (i**2).sum() / both * 100]
    res["  shares of the two parts' combined size (chart)"] = (
        f"overnight {share[0]:.1f}%, trading day {share[1]:.1f}%")
    ax.bar([0, 1], share, color=[plots.COLORS["unhedged"], plots.COLORS["hedged"]], width=0.6)
    for x, y in enumerate(share):
        ax.text(x, y + 1, f"{y:.0f}%", ha="center", va="bottom", fontsize=10, color=plots.TEXT)
    ax.set_xticks([0, 1], ["overnight gap\n(close to open)", "trading day\n(open to close)"])
    ax.set_ylabel("share of NIFTY's daily price\nvariance, 2015-2026, %")
    ax.set_ylim(0, 75)
    ax.set_title("Two-fifths of NIFTY's daily risk arrives overnight", fontsize=10.5)
    ax = axes[1]
    for lab, col in (("no jumps", plots.COLORS["hedged"]), ("jumps", plots.COLORS["unhedged"])):
        ax.plot(per_day, sims[lab], color=col, lw=2, marker="o")
        plots.label_line_end(ax, per_day[-1], sims[lab][-1],
                             "with jumps" if lab == "jumps" else "no jumps", col)
    ax.set_xscale("log")
    ax.set_xticks(per_day, [str(k) for k in per_day])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.8, 14)
    ax.set_ylim(0, None)
    ax.set_xlabel("hedge trades a day (simulated 30-day ATM call)")
    ax.set_ylabel("spread of hedged P&L, Rs")
    ax.set_title("More trades cut the noise, not the jumps", fontsize=10.5)
    fig.suptitle("A hedge set at the close can't catch gaps, and trading more often doesn't fix "
                 "jumps", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "gaps")
    plt.close(fig)
    return res


def report_gamma(out_dir: Path) -> dict:
    """Gamma hedging with a second option (dhs.gamma): finding 1's short at-the-money calls hedged
    with futures only, with a static gamma hedge (next monthly expiry's option at the same strike,
    bought once to cancel the gamma, held), or a dynamic one (re-set every close), all with today's
    Indian futures and option costs and a 1-point option spread. Nothing is fitted (the hedge ratio
    is Gamma_front / Gamma_back); halves shown."""
    import pandas as pd

    from dhs.gamma import month_pairs, run_month

    res = {}
    rows = []
    for m in month_pairs():
        for meth in ("futures", "static", "dynamic"):
            o = run_month(m, meth)
            rows.append({"date": m["date"], "meth": meth, "pnl": 100 * o["pnl"] / o["F0"],
                         "cost": 100 * o["cost"] / o["F0"], "n": o["n_back"]})
    d = pd.DataFrame(rows)
    names = {"futures": "futures only", "static": "static gamma hedge",
             "dynamic": "dynamic gamma hedge"}
    stats = {}
    for meth in ("futures", "static", "dynamic"):
        x = d[d["meth"] == meth]
        srt = x["pnl"].sort_values()
        k = max(1, round(0.1 * len(srt)))
        stats[meth] = {"mean": x["pnl"].mean(), "sd": x["pnl"].std(), "worst": srt.iloc[:k].mean(),
                       "best": srt.iloc[-k:].mean(), "cost": x["cost"].mean()}
        halves = x.groupby(x["date"].dt.year <= 2020)["pnl"].mean()
        res[f"{names[meth]} ({len(x)} months)"] = (
            f"mean after costs {stats[meth]['mean']:+.3f}% of NIFTY a month (2015-2020 "
            f"{halves[True]:+.3f}%, 2021-2026 {halves[False]:+.3f}%), "
            f"spread {stats[meth]['sd']:.3f}%, "
            f"average of the worst tenth {stats[meth]['worst']:+.3f}%, of the best tenth "
            f"{stats[meth]['best']:+.3f}%, costs {stats[meth]['cost']:.3f}%"
            + (f", {x['n'].mean():.2f} back options bought per front option" if meth == "static"
               else "")
        )
    w = d.pivot(index="date", columns="meth", values="pnl")
    worst5 = w["futures"].nsmallest(5).index
    res["  the five worst futures-only months, futures / static / dynamic"] = "; ".join(
        f"{t:%b %Y} {w.at[t, 'futures']:+.2f} / {w.at[t, 'static']:+.2f} / "
        f"{w.at[t, 'dynamic']:+.2f}" for t in worst5)
    gapm = (w["futures"] - w["dynamic"]).idxmax()
    res["  correlation of monthly P&L, futures only against dynamic"] = (
        f"{w[['futures', 'dynamic']].corr().iloc[0, 1]:+.2f}; widest split {gapm:%b %Y}: futures "
        f"{w.at[gapm, 'futures']:+.2f}%, dynamic {w.at[gapm, 'dynamic']:+.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    cols = {"futures": plots.COLORS["hedged"], "static": plots.SERIES[3],
            "dynamic": plots.COLORS["unhedged"]}
    groups = [("mean", "average month"), ("worst", "worst tenth\nof months")]
    xs = np.arange(len(groups))
    for j, meth in enumerate(("futures", "static", "dynamic")):
        vals = [stats[meth][k] for k, _ in groups]
        ax.bar(xs + (j - 1) * 0.26, vals, 0.26, color=cols[meth], label=names[meth])
        for x, y in zip(xs + (j - 1) * 0.26, vals, strict=True):
            ax.text(x, y + (0.03 if y >= 0 else -0.03), f"{y:+.2f}", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=8.5, color=plots.TEXT)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(xs, [g[1] for g in groups])
    ax.set_ylabel("hedged P&L after costs,\n% of NIFTY a month")
    ax.set_ylim(-1.15, 0.4)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("It cost the average and the worst months", fontsize=10.5)
    ax = axes[1]
    ax.scatter(w["futures"], w["dynamic"], s=16, color=plots.COLORS["hedged"], alpha=0.7, lw=0)
    lim = float(np.nanmax(np.abs(w[["futures", "dynamic"]].to_numpy()))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=plots.COLORS["formula"], lw=1, ls="--")
    ax.axhline(0, color=plots.GRID, lw=0.8)
    ax.axvline(0, color=plots.GRID, lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("futures only, % of NIFTY (one dot a month)")
    ax.set_ylabel("dynamic gamma hedge, % of NIFTY")
    ax.set_title("Different months went wrong", fontsize=10.5)
    fig.suptitle("A second option swapped one risk for another and cost the hedged profit",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "gamma")
    plt.close(fig)
    return res


def report_stocks(out_dir: Path) -> dict:
    """Single-stock options (dhs.stocks): at each month-end, every F&O stock's 1-month at-the-money
    call sold at its own implied volatility and hedged at each close until expiry, against
    finding 1's NIFTY calls; results by the size of the month's largest daily move. Nothing is
    fitted; halves shown. Needs DHS_IV_DIR (stock options) and the lake (prices, symbol changes)."""
    import pandas as pd

    from dhs.real import R, months
    from dhs.stocks import atm_cases, hedge_months

    res = {}
    h = hedge_months(atm_cases(R), R)
    n = months(r=R)
    n = n[n["kind"] == "call"].assign(pnl=lambda x: 100 * x["pnl"] / x["F0"],
                                      unhedged=lambda x: 100 * x["unhedged"] / x["F0"],
                                      gamma_sum=lambda x: 100 * x["gamma_sum"] / x["F0"])
    res["stock-months"] = (
        f"{len(h):,} on {h['date'].nunique()} month-ends, {h['symbol'].nunique()} stocks; sold "
        f"volatility median {h['vol'].median():.1%} (NIFTY {n['sold_vol'].median():.1%}); sold "
        f"against the month's realised volatility, median ratio "
        f"{(h['vol'] / h['realised']).median():.2f}"
    )
    for lab, x in (("stocks", h), ("NIFTY", n)):
        halves = x.groupby(x["date"].dt.year <= 2020)["pnl"].mean()
        r2 = np.corrcoef(x["pnl"], x["gamma_sum"])[0, 1] ** 2
        res[f"{lab}: hedged at-the-money call, % of the price a month"] = (
            f"mean {x['pnl'].mean():+.3f} (2015-2020 {halves[True]:+.3f}, 2021-2026 "
            f"{halves[False]:+.3f}), median {x['pnl'].median():+.3f}, spread {x['pnl'].std():.3f}, "
            f"mean / spread {x['pnl'].mean() / x['pnl'].std():.2f}; unhedged spread "
            f"{x['unhedged'].std():.2f}; gamma sum explains {r2:.0%}; losing months "
            f"{(x['pnl'] < 0).mean():.0%}"
        )
    bins = [0, 0.03, 0.05, 0.08, 1]
    labels = ["under 3%", "3-5%", "5-8%", "over 8%"]
    g = h.groupby(pd.cut(h["max_move"], bins, labels=labels), observed=True)["pnl"]
    tab = g.agg(["size", "mean", "std"])
    res["stocks by the month's largest daily move"] = "; ".join(
        f"{k}: {int(r_['size']):,} months ({r_['size'] / len(h):.0%}), mean {r_['mean']:+.2f}%, "
        f"spread {r_['std']:.2f}%" for k, r_ in tab.iterrows())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    vals = [[n["pnl"].mean(), h["pnl"].mean()], [n["pnl"].std(), h["pnl"].std()]]
    xs = np.arange(2)
    for j, (lab, col) in enumerate((("NIFTY", plots.COLORS["hedged"]),
                                    ("single stocks", plots.COLORS["unhedged"]))):
        v = [vals[0][j], vals[1][j]]
        ax.bar(xs + (j - 0.5) * 0.38, v, 0.38, color=col, label=lab)
        for x, y in zip(xs + (j - 0.5) * 0.38, v, strict=True):
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=plots.TEXT)
    ax.set_xticks(xs, ["average month", "spread of months"])
    ax.set_ylabel("hedged short ATM call,\n% of the price a month")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Stocks paid more, with three times the risk", fontsize=10.5)
    ax = axes[1]
    ax.bar(range(len(tab)), tab["mean"], color=[plots.COLORS["hedged"] if v >= 0 else
                                               plots.COLORS["unhedged"] for v in tab["mean"]])
    for x, (v, size) in enumerate(zip(tab["mean"], tab["size"], strict=True)):
        ax.text(x, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}\n({size / len(h):.0%} of months)",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=plots.TEXT)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(range(len(tab)), list(tab.index))
    ax.set_xlabel("the stock's largest one-day move that month")
    ax.set_ylabel("average hedged P&L, % of the price")
    ax.set_ylim(min(tab["mean"]) - 0.6, max(tab["mean"]) + 0.5)
    ax.set_title("One big day turned the month into a loss", fontsize=10.5)
    fig.suptitle("Hedged single-stock options earned more than NIFTY's but were ruled by jump days",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "stocks")
    plt.close(fig)
    return res


def report_verdict(out_dir: Path) -> dict:
    """Is selling and hedging NIFTY options an edge after costs and gaps? The book of dhs.verdict
    (an at-the-money call and a 95% put sold each month, delta-band hedged with finding 3's
    2015-2020 settings, all Indian costs and half of a 1-point option spread), before and after
    costs, by half (2021-2026 out of sample for the band settings), and with a rule fixed in
    advance: sell only when the sold volatility is above NIFTY's last-20-day volatility."""
    from dhs.verdict import book, summary

    res = {}
    m = book()
    res["book"] = (
        f"{len(m)} months ({m.index.min():%b %Y}-{m.index.max():%b %Y}); costs a month "
        f"{m['fut_cost'].mean():.3f}% of NIFTY for the futures hedge and "
        f"{m['opt_cost'].mean():.3f}% "
        f"for selling the two options; the rule sells in {m['premium_rich'].mean():.0%} of months"
    )
    keep = {}
    for per, mask in (("all", m.index.year > 0), ("2015-2020", m.index.year <= 2020),
                      ("2021-2026", m.index.year >= 2021)):
        x = m[mask]
        for lab, ser in (("before costs", x["gross"]), ("after costs", x["net"]),
                         ("after costs, sell only when implied > last 20 days",
                          x.loc[x["premium_rich"], "net"])):
            s_ = summary(ser)
            keep[(per, lab)] = s_
            res[f"{per}, {lab}"] = (
                f"{s_['n']} months: mean {s_['mean']:+.3f}% a month ({s_['annual']:+.2f}% a year), "
                f"spread {s_['sd']:.3f}%, t {s_['t']:.2f}, positive {s_['pos']:.0%}, worst month "
                f"{s_['worst']:+.2f}%, worst fall in the running total {s_['dd']:+.2f}%"
            )
    res["  worst months after costs"] = "; ".join(
        f"{d:%b %Y} {v:+.2f}%" for d, v in m["net"].nsmallest(5).items())
    a = keep[("all", "after costs")]
    res["  worst fall against the average year"] = (
        f"{-a['dd'] / a['annual']:.1f} years of the average profit")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    run_g, run_n = m["gross"].cumsum(), m["net"].cumsum()
    rich = m["net"].where(m["premium_rich"], 0.0).cumsum()
    ax.axvspan(pd_ts("2021-01-01"), m.index.max(), color=plots.GRID, alpha=0.6, lw=0)
    ax.plot(run_g.index, run_g, color=plots.MUTED, lw=1.6)
    ax.plot(run_n.index, run_n, color=plots.COLORS["hedged"], lw=2.2)
    ax.plot(rich.index, rich, color=plots.SERIES[3], lw=1.6)
    plots.label_line_end(ax, run_g.index[-1], float(run_g.iloc[-1]), "before costs", plots.MUTED)
    plots.label_line_end(ax, run_n.index[-1], float(run_n.iloc[-1]) - 2.0, "after costs",
                         plots.COLORS["hedged"])
    plots.label_line_end(ax, rich.index[-1], float(rich.iloc[-1]) + 1.6,
                         "only when implied > recent", plots.SERIES[3])
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.text(pd_ts("2021-02-01"), ax.get_ylim()[1] * 0.92, "2021-2026: out of sample",
            fontsize=9, color=plots.TEXT_2)
    ax.set_ylabel("running total of monthly P&L,\n% of NIFTY (call + put each month)")
    ax.set_title("Selling and hedging NIFTY options earned a thin profit with a few deep falls")
    fig.tight_layout()
    plots.save(fig, out_dir, "verdict")
    plt.close(fig)
    return res


def pd_ts(s):
    import pandas as pd

    return pd.Timestamp(s)


REPORTS: dict[str, Callable[[Path], dict]] = {
    "path": report_path,
    "hedge-vol": report_hedge_vol,
    "rebalance": report_rebalance,
    "gaps": report_gaps,
    "gamma": report_gamma,
    "stocks": report_stocks,
    "verdict": report_verdict,
}

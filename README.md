# Delta Hedging Simulator

[![CI](https://github.com/Dhirajdas18/delta-hedging-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/delta-hedging-simulator/actions/workflows/ci.yml)
[![Hedging simulator](https://github.com/Dhirajdas18/delta-hedging-simulator/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/delta-hedging-simulator/)

Sell an option, hedge it with NIFTY futures, and study what is left: why the hedged P&L depends on
the path the market takes, which volatility to hedge at, how often to rebalance after Indian trading
costs, and what overnight gaps do to a hedge.

**Live hedging simulator: [dhirajdas18.github.io/delta-hedging-simulator](https://dhirajdas18.github.io/delta-hedging-simulator/)**,
set the NIFTY level, strike, days, the volatility you sold at, the volatility that actually
happens and the one you hedge at, the rebalancing rule, costs and jumps, and see the spread of
hedged P&L against the bare short, how well the gamma formula explains each path, and sample
hedges; all computed in the browser.

## Features

- [ ] Why hedged P&L depends on the path: gamma times the gap between implied and realised variance
- [ ] Which volatility to hedge at: implied, realised or a forecast
- [ ] When to rebalance: fixed times against delta bands, after Indian trading costs
- [ ] What the close can't hedge: overnight gaps and jumps
- [ ] Gamma hedging with a second option
- [ ] Hedging single-stock options against NIFTY
- [ ] Is selling and hedging options an edge after costs and gaps

## Findings

Findings are added as each study is completed.

## Quick start

```bash
uv sync
uv run dhs report <topic>               # a study's numbers and charts -> out/
                                        # (topics are added as each study is completed)
uv run python scripts/build_site.py     # hedging simulator -> site/index.html (code only, no data)
uv run pytest
```

The simulator and the tests need no market data; the tests of the browser maths need Node.

## Design

- `src/dhs/sim.py`: the core maths in numpy (scipy only for the normal CDF): Black-76 price,
  delta, gamma and vega on a forward; forward paths as geometric Brownian motion with drift
  −σ²/2 in logs, with optional Merton jumps compensated so the forward stays a martingale; and
  `hedge_pnl`, which sells one option at one volatility, hedges it with futures at Black-76 delta
  at another, rebalancing on a fixed schedule or when delta drifts outside a band, with costs as
  a percentage of the traded value plus a fixed amount per trade. It returns each path's hedged
  P&L, trades, costs, the bare short's P&L and the gamma formula's prediction,
  Σ ½ΓF²(σ²Δt − (ΔF/F)²) plus the premium gap.
- `src/dhs/sim.js`: the simulator's browser copy of Black-76, the seeded paths (mulberry32 with
  Box-Muller) and `hedgePnl`, tested against Python through Node: same seed, same paths, P&L equal
  to 1e-9.
- `src/dhs/reports.py`: README findings, reproducible with `dhs report <topic>`.
- `src/dhs/site.py`: the hedging simulator, one static Plotly page built from code alone.
- `src/dhs/plots.py`: the shared chart style and the animated README charts.
- Conventions: time to expiry is calendar days / 365; rates are continuously compounded, a year;
  money is in rupees per unit of NIFTY and valued at expiry; the premium sits in cash earning the
  rate and futures need no money up front.
- Charts: one validated palette; README charts are animated GIFs with a static fallback for
  reduced-motion users.

Research only, not investment advice. MIT licence.

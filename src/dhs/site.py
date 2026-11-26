"""The hedging simulator: one static Plotly page built from code alone (sim.js does the maths in
the browser), so the Pages build needs no data and no schedule."""

import re
from pathlib import Path
from string import Template

from dhs import plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 10c2-6 4-6 6 0s4 6 6 0M1 8h14%22 fill=%22none%22"
    " stroke=%22%232a78d6%22 stroke-width=%221.6%22 stroke-linecap=%22round%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "sim.js").read_text(encoding="utf-8")
LAB = (HERE / "lab.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/delta-hedging-simulator"


def lab_section() -> str:
    return LAB.replace("$js", JS)


def method_section() -> str:
    return """<ul class="plain">
<li>NIFTY futures move as geometric Brownian motion with one step per calendar day (weekends
count as days, a simplification), at the constant volatility that actually happens. The futures
price is a martingale: on average it ends where it started, so there is no market drift to profit
from. The paths come from a fixed seed, so the same inputs always give the same paths.</li>
<li>With jumps on, each day can bring a Poisson number of jumps (the average number a year is the
intensity); each jump’s size is normal in log terms with the average and spread given. The drift is
lowered to compensate, so the futures price stays a martingale.</li>
<li>You sell one European option on one unit of NIFTY at its Black-76 price at the volatility you
sold at, and hold Black-76 delta in futures at the volatility you hedge at, set at each daily close
and held to the next. Futures need no money up front (margin is ignored); the premium sits in cash
earning the interest rate, and every futures gain and cost earns or loses that rate until expiry.
Futures settle in cash at expiry, so there is no closing trade, and the option’s payoff is paid
then. All profit and loss is in rupees per unit of NIFTY, valued at expiry.</li>
<li>Rebalancing is either every N days or whenever the delta you should hold has moved more than
the band away from what you hold. The first day always trades.</li>
<li>Costs are simplified to one rate on the value traded each time, standing in for brokerage,
taxes, exchange charges and the bid-ask spread together.</li>
<li>The formula’s prediction for each path adds up, day by day, half of gamma times the futures
price squared times the gap between the variance you hedge at and the day’s squared return, plus
the difference between the premium at the volatility you sold at and at the volatility you hedge
at. It is exact only for continuous hedging without jumps; the scatter shows how far a daily hedge
strays from it.</li>
<li>The expected profit from the volatility gap is today’s Black-76 price at the volatility you
sold at minus at the volatility that happens; the simulated average is valued at expiry, so it
includes a few days’ interest on top.</li>
<li>Real markets have changing volatility, a smile, intraday moves and liquidity limits; none of
that is here. Research only, not investment advice.</li>
</ul>"""


SECTIONS = [
    (
        "Hedging simulator",
        "Change any input and the simulation reruns. Research only, not investment advice.",
        lab_section,
    ),
    (
        "Method and caveats",
        "What the simulator assumes, and what it leaves out.",
        method_section,
    ),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html() -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        body = make()
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    dk = plots.DARK
    s = plots.SERIES
    return PAGE.substitute(
        icon=ICON,
        font=FONT,
        repo=REPO,
        surface=plots.SURFACE,
        text=plots.TEXT,
        text2=plots.TEXT_2,
        grid=plots.GRID,
        accent=s[0],
        accent2=s[1],
        green=s[2],
        violet=s[6],
        muted=plots.MUTED,
        d_surface=dk[plots.SURFACE],
        d_text=dk[plots.TEXT],
        d_text2=dk[plots.TEXT_2],
        d_grid=dk[plots.GRID],
        d_accent=dk[s[0]],
        d_accent2=dk[s[1]],
        d_green=dk[s[2]],
        d_violet=dk[s[6]],
        d_muted="#6b6a64",
        nav="\n".join(nav),
        sections="\n".join(parts),
    )

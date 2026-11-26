import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dhs import sim

JS = Path(__file__).resolve().parents[1] / "src" / "dhs" / "sim.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
M32 = 0xFFFFFFFF


def mulberry32(seed: int):
    """Python copy of sim.js's PRNG, for these tests only (the research code uses numpy's)."""
    a = seed & M32

    def imul(x, y):
        return (x * y) & M32

    def u() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return (t ^ (t >> 14)) / 4294967296

    return u


def js_shocks(seed, n_paths, steps, lam_dt=0.0):
    """The normals and jump counts sim.js draws, in its order (Box-Muller, Knuth's Poisson)."""
    u = mulberry32(seed)

    def gauss():
        return math.sqrt(-2 * math.log(1 - u())) * math.cos(2 * math.pi * u())

    z, jumps, zj = (np.zeros((n_paths, steps)) for _ in range(3))
    for p in range(n_paths):
        for i in range(steps):
            z[p, i] = gauss()
            if lam_dt > 0:
                lim, k, q = math.exp(-lam_dt), 0, u()
                while q > lim:
                    k += 1
                    q *= u()
                jumps[p, i], zj[p, i] = k, gauss()
    return z, jumps, zj


def _node(expr: str):
    # The script goes in on stdin: Linux caps a single command-line argument at 128 KB.
    script = (f"const g = require({json.dumps(str(JS))});"
              "const arr = (x) => ArrayBuffer.isView(x) ? Array.from(x) : Array.isArray(x) ? "
              "x.map(arr) : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x)"
              ".map(([k, v]) => [k, arr(v)])) : x;"
              f"console.log(JSON.stringify(arr({expr})));")
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_prng_matches_python_copy():
    u = mulberry32(42)
    assert _node("(() => { const u = g.mulberry32(42); return Array.from({length: 1000}, u); })()"
                 ) == [u() for _ in range(1000)]


def test_browser_black76_matches_python():
    xs = [(25000.0, k, t, 0.065, v) for k in (22000.0, 25000.0, 28000.0) for t in (2 / 365, 0.25)
          for v in (0.1, 0.3)]
    js = _node(f"{json.dumps(xs)}.map((a) => [g.black76(...a, 'call'), g.black76(...a, 'put'),"
               "g.delta76(...a, 'call'), g.delta76(...a, 'put'), g.gamma76(...a), g.vega76(...a)])")
    for a, row in zip(xs, js, strict=True):
        py = [sim.black76(*a, "call"), sim.black76(*a, "put"), sim.delta(*a, "call"),
              sim.delta(*a, "put"), sim.gamma(*a), sim.vega(*a)]
        assert row == pytest.approx(py, rel=1e-9, abs=1e-12)


CASES = [  # (paths args, hedge args) with the lab's defaults first
    ((25000.0, 0.13, 30 / 365, 30, 200, 1, 0, 0, 0),
     (25000.0, 30 / 365, 0.065, 0.15, 0.15, "call", "time", 1, None, 0.0, 0.0002)),
    ((25000.0, 0.15, 30 / 365, 30, 200, 9, 6.0, -0.02, 0.03),
     (25500.0, 30 / 365, 0.065, 0.14, 0.16, "put", "time", 5, None, 20.0, 0.0002)),
    ((25000.0, 0.2, 60 / 365, 60, 150, 123, 0, 0, 0),
     (24000.0, 60 / 365, 0.065, 0.2, 0.18, "call", "band", 1, 0.05, 20.0, 0.0005)),
]


@pytest.mark.parametrize(("pa", "ha"), CASES)
def test_browser_paths_and_hedge_pnl_match_python(pa, ha):
    F0, sig, T, steps, n, seed, lam, mu, sd = pa
    js = _node(f"(() => {{ const p = g.simulatePaths(...{json.dumps(pa)});"
               f"const a = {json.dumps(ha)};"
               "return { paths: p, res: g.hedgePnl(p, ...a) }; })()")
    z, jumps, zj = js_shocks(seed, n, steps, lam * T / steps)
    paths = sim.paths_from_shocks(F0, sig, T, z, jumps if lam else None, zj, lam, mu, sd)
    assert np.array(js["paths"]) == pytest.approx(paths, rel=1e-12)  # libm last-bit differences
    py = sim.hedge_pnl(paths, *ha)
    for key, val in py.items():
        assert np.array(js["res"][key]) == pytest.approx(val, rel=1e-9, abs=1e-9), key

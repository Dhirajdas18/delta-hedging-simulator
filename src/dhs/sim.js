// Hedging maths in the browser. Mirrors src/dhs/sim.py; tests compare the two through Node.
// Arguments as in Python: F forward, K strike, T years, r rate, sigma vol a year, kind "call" or
// "put". Random numbers come from mulberry32 with Box-Muller (tests/test_sim_js.py has a Python
// copy), so a seed gives the same paths in both.

// Standard normal CDF to double precision (Hart 1968, as given by West 2005).
function normCdf(x) {
  const a = Math.abs(x);
  let c = 0;
  if (a <= 37) {
    const e = Math.exp(-a * a / 2);
    if (a < 7.07106781186547) {
      let num = 3.52624965998911e-2 * a + 0.700383064443688;
      num = num * a + 6.37396220353165;
      num = num * a + 33.912866078383;
      num = num * a + 112.079291497871;
      num = num * a + 221.213596169931;
      num = num * a + 220.206867912376;
      let den = 8.83883476483184e-2 * a + 1.75566716318264;
      den = den * a + 16.064177579207;
      den = den * a + 86.7807322029461;
      den = den * a + 296.564248779674;
      den = den * a + 637.333633378831;
      den = den * a + 793.826512519948;
      den = den * a + 440.413735824752;
      c = e * num / den;
    } else {
      c = e / (a + 1 / (a + 2 / (a + 3 / (a + 4 / (a + 0.65))))) / 2.506628274631;
    }
  }
  return x > 0 ? 1 - c : c;
}

const normPdf = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
const d1Of = (F, K, T, sigma) => (Math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * Math.sqrt(T));

function black76(F, K, T, r, sigma, kind) {
  const d1 = d1Of(F, K, T, sigma), d2 = d1 - sigma * Math.sqrt(T), df = Math.exp(-r * T);
  return kind === "call" ? df * (F * normCdf(d1) - K * normCdf(d2)) : df * (K * normCdf(-d2) - F * normCdf(-d1));
}
const delta76 = (F, K, T, r, sigma, kind) => Math.exp(-r * T) * (normCdf(d1Of(F, K, T, sigma)) - (kind === "put" ? 1 : 0));
const gamma76 = (F, K, T, r, sigma) => Math.exp(-r * T) * normPdf(d1Of(F, K, T, sigma)) / (F * sigma * Math.sqrt(T));
const vega76 = (F, K, T, r, sigma) => Math.exp(-r * T) * F * normPdf(d1Of(F, K, T, sigma)) * Math.sqrt(T);

// Seeded uniforms in [0, 1).
function mulberry32(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (u) => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
function poisson(u, mean) {  // Knuth: fine for the small means of one step
  const lim = Math.exp(-mean);
  let k = 0, p = u();
  while (p > lim) { k++; p *= u(); }
  return k;
}

// Forward paths, one Float64Array of steps + 1 prices per path. Draw order per path and step:
// the diffusion normal, then (with jumps) the jump count and one normal for their total size.
function simulatePaths(F0, sigma, T, steps, nPaths, seed, lam = 0, mu = 0, sd = 0) {
  const u = mulberry32(seed), dt = T / steps, k = Math.exp(mu + 0.5 * sd * sd) - 1;
  const drift = (-0.5 * sigma * sigma - lam * k) * dt, vol = sigma * Math.sqrt(dt);
  const out = [];
  for (let p = 0; p < nPaths; p++) {
    const f = new Float64Array(steps + 1);
    f[0] = F0;
    let lf = 0;
    for (let i = 0; i < steps; i++) {
      let x = drift + vol * gauss(u);
      if (lam > 0) {
        const n = poisson(u, lam * dt);
        x = x + (n * mu + Math.sqrt(n) * sd * gauss(u));
      }
      lf += x;
      f[i + 1] = F0 * Math.exp(lf);
    }
    out.push(f);
  }
  return out;
}

// Sell one option at sigmaSold, hedge with futures at sigmaHedge; see hedge_pnl in sim.py.
function hedgePnl(paths, K, T, r, sigmaHedge, sigmaSold, kind, rebalance = "time", every = 1, band = null, costFixed = 0, costPct = 0) {
  const n = paths.length, steps = paths[0].length - 1, dt = T / steps;
  const grow = Array.from({ length: steps + 1 }, (_, i) => Math.exp(r * (T - i * dt)));
  const res = { pnl: new Float64Array(n), trades: new Float64Array(n), costs: new Float64Array(n),
    unhedged: new Float64Array(n), gamma: new Float64Array(n), edge: new Float64Array(n), held: [] };
  for (let p = 0; p < n; p++) {
    const F = paths[p], held = new Float64Array(steps);
    const premium = black76(F[0], K, T, r, sigmaSold, kind);
    let h = 0, trades = 0, costs = 0, hedge = 0, gam = 0;
    for (let i = 0; i < steps; i++) {
      const tau = T - i * dt;
      const target = delta76(F[i], K, tau, r, sigmaHedge, kind);
      const trade = rebalance === "time" ? i % every === 0 : (Math.abs(target - h) > band || i === 0);
      const nw = trade ? target : h;
      costs += ((trade ? costFixed : 0) + costPct * Math.abs(nw - h) * F[i]) * grow[i];
      trades += trade ? 1 : 0;
      h = held[i] = nw;
      const dF = F[i + 1] - F[i];
      hedge += h * dF * grow[i + 1];
      const g = gamma76(F[i], K, tau, r, sigmaHedge);
      gam += 0.5 * g * (sigmaHedge * sigmaHedge * dt * F[i] * F[i] - dF * dF) * grow[i + 1];
    }
    const FT = F[steps], payoff = kind === "call" ? Math.max(FT - K, 0) : Math.max(K - FT, 0);
    const unhedged = premium * grow[0] - payoff;
    res.pnl[p] = unhedged + hedge - costs;
    res.trades[p] = trades; res.costs[p] = costs; res.unhedged[p] = unhedged; res.gamma[p] = gam;
    res.edge[p] = (premium - black76(F[0], K, T, r, sigmaHedge, kind)) * grow[0];
    res.held.push(held);
  }
  return res;
}

if (typeof module !== "undefined") {
  module.exports = { normCdf, black76, delta76, gamma76, vega76, mulberry32, simulatePaths, hedgePnl };
}

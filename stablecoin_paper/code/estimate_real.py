"""Estimate the threshold-arbitrage model on real data.

Input CSV (one row per minute), columns:
  timestamp, local_price, benchmark_price, buy_volume, sell_volume
local_price     : USDT price in local currency (mid-quote or last trade)
benchmark_price : USD price in local currency (official/interbank FX, or crypto-implied)
buy/sell_volume : taker-buy and taker-sell USDT volume in that minute

Usage: python estimate_real.py data.csv [label]
Writes tables/real_<label>.tex with the structural estimates.
"""
import sys, os
import numpy as np
import pandas as pd
from model import fit_threshold, fit_linear

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(path))[0]
df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").dropna()
b = 1e4 * (np.log(df.local_price.values) - np.log(df.benchmark_price.values))   # bp
ofi = (df.buy_volume - df.sell_volume).values
ofi = (ofi - ofi.mean()) / ofi.std()
q = ofi[:-1]                     # OFI_t predicts db_{t+1}
grid = np.arange(0, np.percentile(np.abs(b), 95), 2.5)
f = fit_threshold(b, q, grid, grid)
lin = fit_linear(b, q)

# In-band estimate of primitive impact beta (arbitrage flow is zero in the band)
x, y = b[:-1], np.diff(b)
m = (x >= -f["cm"]) & (x <= f["cp"])
X = np.column_stack([np.ones(m.sum()), q[m]])
beta_in = np.linalg.lstsq(X, y[m], rcond=None)[0][1]

# Block bootstrap (1-day blocks) for slope SEs given thresholds
rng = np.random.default_rng(0); B = 199; blk = 1440; n = len(y)
Xf = np.column_stack([np.ones(n), q, -np.maximum(x - f["cp"], 0), np.maximum(-x - f["cm"], 0)])
draws = []
for _ in range(B):
    idx = np.concatenate([np.arange(s, min(s + blk, n)) for s in rng.integers(0, n - blk, n // blk + 1)])[:n]
    draws.append(np.linalg.lstsq(Xf[idx], y[idx], rcond=None)[0])
se = np.std(draws, axis=0)

rows = [("$c^+$ (bp)", f["cp"], None), ("$c^-$ (bp)", f["cm"], None),
        ("$\\kappa^+$ (per min)", f["kp"], se[2]), ("$\\kappa^-$ (per min)", f["km"], se[3]),
        ("Half-life$^+$ (min)", np.log(2) / f["kp"], None), ("Half-life$^-$ (min)", np.log(2) / f["km"], None),
        ("$\\beta$ (full sample)", f["beta"], se[1]), ("$\\beta$ (in-band)", beta_in, None),
        ("Long-run impact $\\beta/\\kappa^+$", beta_in / f["kp"], None),
        ("Linear ECM $\\kappa$", lin["k"], None), ("Share in band", m.mean(), None), ("$N$", n, None)]
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "tables"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "..", "tables", f"real_{label}.tex")
with open(out, "w") as fh:
    fh.write("\\begin{tabular}{lc}\n\\toprule\n & " + label + " \\\\\n\\midrule\n")
    for name, v, s in rows:
        fh.write(f"{name} & {v:.4g}" + (f" ({s:.3g})" if s is not None else "") + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(open(out).read())

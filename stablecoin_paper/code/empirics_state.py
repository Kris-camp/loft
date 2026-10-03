"""Bank-state-specific bands and in-band identification of order-flow impact.

Separates the friction (band width) from effective capacity (reversion speed):
bands [l, u] are profiled separately for each (month, banks open/closed) cell, and the
reversion speeds are then estimated with state-specific bands. A second specification
estimates beta from in-band observations only (where a* = 0), fixes it, and estimates the
outside-band speeds on the order-flow-adjusted premium change.
Requires data_raw/a5.parquet from empirics_main.py.
"""
import os, json
import numpy as np, pandas as pd
from model import _ssr_grid

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
a["day"] = a.index.date
OUT = {}


def profile(d, trim=0.05, step=1.0):
    y, x, q = d.y.values, d.x.values, d.q.values
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def ols_fe(d, ycol, cols, fe="month"):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    gm = d.groupby(fe)
    Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(zip(cols, b)), dict(zip(cols, np.sqrt(np.diag(V)))), n


def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


# ------------------------------------------------------------------ state-specific bands
a["state"] = np.where(a.bh == 1, "open", "closed")
bands = {}
for (m, st), d in a.groupby(["month", "state"]):
    if len(d) < 300: continue
    bands[(m, st)] = profile(d)
a["l_s"] = [bands.get((m, s), (np.nan, np.nan))[0] for m, s in zip(a.month, a.state)]
a["u_s"] = [bands.get((m, s), (np.nan, np.nan))[1] for m, s in zip(a.month, a.state)]
a = a[a.l_s.notna()].copy()
a["up_s"] = -np.maximum(a.x - a.u_s, 0); a["dn_s"] = np.maximum(a.l_s - a.x, 0)
for c in ["q", "up_s", "dn_s"]:
    a[c + "_bh"] = a[c] * a.bh
a["ms"] = a.month + "_" + a.state       # month x state fixed effects
r_s = ols_fe(a, "y", ["q", "up_s", "dn_s", "q_bh", "up_s_bh", "dn_s_bh"], fe="ms")
B = pd.DataFrame([(m, s, l, u) for (m, s), (l, u) in bands.items()], columns=["month", "state", "l", "u"])
B["w"] = B.u - B.l
wide = B.pivot(index="month", columns="state", values="w").dropna()
diff = wide.closed - wide.open
OUT["state_bands"] = dict(width_open_mean=float(wide.open.mean()), width_closed_mean=float(wide.closed.mean()),
                          width_open_median=float(wide.open.median()), width_closed_median=float(wide.closed.median()),
                          diff_mean=float(diff.mean()), diff_se=float(diff.std() / np.sqrt(len(diff))),
                          share_months_wider_closed=float((diff > 0).mean()), n_months=int(len(diff)),
                          coef={k: [float(r_s[0][k]), float(r_s[1][k])] for k in r_s[0]}, n=r_s[2])

# ------------------------------------------------------------------ in-band beta, fixed
inb = a[(a.x >= a.l_s) & (a.x <= a.u_s)]
r_in = ols_fe(inb, "y", ["q", "q_bh"], fe="ms")
b_closed, b_open = r_in[0]["q"], r_in[0]["q"] + r_in[0]["q_bh"]
a["y_adj"] = a.y - np.where(a.bh == 1, b_open, b_closed) * a.q
r_fix = ols_fe(a, "y_adj", ["up_s", "dn_s", "up_s_bh", "dn_s_bh"], fe="ms")
OUT["inband_beta"] = dict(beta_closed=float(b_closed), beta_open=float(b_open),
                          se_q=float(r_in[1]["q"]), se_q_bh=float(r_in[1]["q_bh"]), n_in=int(r_in[2]),
                          coef={k: [float(r_fix[0][k]), float(r_fix[1][k])] for k in r_fix[0]})

with open(os.path.join(TAB, "emp_state.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcc}\n\\toprule\n & State-specific bands & In-band $\\beta$, fixed \\\\\n & (1) & (2) \\\\\n\\midrule\n")
    rows = [("$\\kappa^+$, banks closed", "up_s"), ("$\\quad\\times$ banks open", "up_s_bh"),
            ("$\\kappa^-$, banks closed", "dn_s"), ("$\\quad\\times$ banks open", "dn_s_bh")]
    for lab, k in rows:
        c1, s1 = r_s[0][k], r_s[1][k]; c2, s2 = r_fix[0][k], r_fix[1][k]
        fh.write(f"{lab} & ${c1:.4f}{star(c1, s1)}$ & ${c2:.4f}{star(c2, s2)}$ \\\\\n & $({s1:.4f})$ & $({s2:.4f})$ \\\\\n")
    c1, s1 = r_s[0]["q"], r_s[1]["q"]; c2, s2 = r_in[0]["q"], r_in[1]["q"]
    fh.write(f"$\\beta$, banks closed & ${c1:.3f}{star(c1, s1)}$ & ${c2:.3f}{star(c2, s2)}$ \\\\\n & $({s1:.3f})$ & $({s2:.3f})$ \\\\\n")
    c1, s1 = r_s[0]["q_bh"], r_s[1]["q_bh"]; c2, s2 = r_in[0]["q_bh"], r_in[1]["q_bh"]
    fh.write(f"$\\quad\\times$ banks open & ${c1:.3f}{star(c1, s1)}$ & ${c2:.3f}{star(c2, s2)}$ \\\\\n & $({s1:.3f})$ & $({s2:.3f})$ \\\\\n")
    sb = OUT["state_bands"]
    fh.write("\\midrule\n")
    fh.write(f"Mean band width, banks open (bp) & \\multicolumn{{2}}{{c}}{{{sb['width_open_mean']:.1f}}} \\\\\n")
    fh.write(f"Mean band width, banks closed (bp) & \\multicolumn{{2}}{{c}}{{{sb['width_closed_mean']:.1f}}} \\\\\n")
    fh.write(f"Difference, closed $-$ open (bp) & \\multicolumn{{2}}{{c}}{{{sb['diff_mean']:.1f} ({sb['diff_se']:.1f})}} \\\\\n")
    fh.write(f"$N$ & {r_s[2]:,} & {r_fix[2]:,} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
json.dump(OUT, open(os.path.join(TAB, "emp_state.json"), "w"), indent=1)
a[["l_s", "u_s"]].to_parquet(os.path.join(RAW, "state_bands.parquet"))
print(json.dumps(OUT, indent=1))

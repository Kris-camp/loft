"""Two additional tests (run after empirics_main.py, which writes data_raw/a5.parquet).

1. Benchmark-quality control: the banking-hours result re-estimated on intervals with an
   actively quoted interbank market, and the global leg's reversion by banking hours.
2. Capacity shock: MASAK General Circular No. 29 (Official Gazette 32940, 28 June 2025),
   which capped stablecoin transfers (USD 3,000/day, 50,000/month) and imposed 48-72h
   withdrawal holds.
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
a["day"] = a.index.date
OUT = {}


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


def cell(r, k, nd=4):
    return f"${r[0][k]:.{nd}f}{star(r[0][k], r[1][k])}$", f"$({r[1][k]:.{nd}f})$"


for c in ["q", "up", "dn"]:
    a[c + "_bh"] = a[c] * a.bh

# ------------------------------------------------------------------ 1. benchmark quality
cols = ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"]
thr = a.loc[a.bh == 1, "fxv"].quantile(0.25)       # 25th pct of interbank activity in business hours
act = a[a.fxv >= thr]
r_all = ols_fe(a, "y", cols)
r_act = ols_fe(act, "y", cols)
# global leg by banking hours: if night-time benchmark noise drove the result, the interbank
# leg would show reversion at night
dd = a.dropna(subset=["dpg", "dpl"])
r_g = ols_fe(dd, "dpg", cols)
r_l = ols_fe(dd, "dpl", cols)
OUT["quality"] = dict(fxv_threshold=float(thr), share_off_kept=float((act.bh == 0).sum() / (a.bh == 0).sum()),
                      share_bh_kept=float((act.bh == 1).sum() / (a.bh == 1).sum()),
                      all={k: [float(r_all[0][k]), float(r_all[1][k])] for k in cols},
                      active={k: [float(r_act[0][k]), float(r_act[1][k])] for k in cols},
                      glob={k: [float(r_g[0][k]), float(r_g[1][k])] for k in cols},
                      loc={k: [float(r_l[0][k]), float(r_l[1][k])] for k in cols}, n_active=r_act[2])
rows = [("$-(b_{t-1}-u_m)_+$ [$\\kappa^+$, banks closed]", "up"), ("$\\quad\\times$ banks open", "up_bh"),
        ("$(\\ell_m-b_{t-1})_+$ [$\\kappa^-$, banks closed]", "dn"), ("$\\quad\\times$ banks open", "dn_bh"),
        ("OFI$_t$ [banks closed]", "q"), ("$\\quad\\times$ banks open", "q_bh")]
with open(os.path.join(TAB, "emp_quality.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{$\\Delta b_t$} & $\\Delta\\log P^L_t$ & $\\Delta\\log P^G_t$ \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-5}\n & All & Active interbank & All & All \\\\\n"
             " & (1) & (2) & (3) & (4) \\\\\n\\midrule\n")
    for lab, k in rows:
        nd = 3 if k.startswith("q") else 4
        rr = [r_all, r_act, r_l, r_g]
        fh.write(lab + " & " + " & ".join(cell(r, k, nd)[0] for r in rr) + " \\\\\n")
        fh.write(" & " + " & ".join(cell(r, k, nd)[1] for r in rr) + " \\\\\n")
    fh.write(f"\\midrule\n$N$ & {r_all[2]:,} & {r_act[2]:,} & {r_l[2]:,} & {r_g[2]:,} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ------------------------------------------------------------------ 2. MASAK Circular No. 29
ev = pd.Timestamp("2025-06-28", tz="UTC")
win = a[(a.index >= pd.Timestamp("2024-12-28", tz="UTC")) & (a.index < pd.Timestamp("2025-12-28", tz="UTC"))].copy()
# drop the 19 March - 30 April 2025 political shock from the pre-period
win = win[~((win.index >= pd.Timestamp("2025-03-19", tz="UTC")) & (win.index < pd.Timestamp("2025-05-01", tz="UTC")))]
win["post"] = (win.index >= ev).astype(float)
for c in ["q", "up", "dn"]:
    win[c + "_post"] = win[c] * win.post
r_ev = ols_fe(win, "y", ["q", "up", "dn", "q_post", "up_post", "dn_post"])
pre, post = win[win.post == 0], win[win.post == 1]


def bands(d):
    m = d.groupby("month")[["l", "u"]].first()
    return float(m.l.mean()), float(m.u.mean()), float((m.u - m.l).mean())


bp, bq = bands(pre), bands(post)
desc = {}
for lab, d in [("pre", pre), ("post", post)]:
    desc[lab] = dict(mean=float(d.x.mean()), p05=float(d.x.quantile(.05)), p95=float(d.x.quantile(.95)),
                     share_below_l=float((d.x < d.l).mean()), share_above_u=float((d.x > d.u).mean()), n=len(d))
OUT["masak"] = dict(coef={k: [float(r_ev[0][k]), float(r_ev[1][k])] for k in r_ev[0]}, n=r_ev[2],
                    band_pre=bp, band_post=bq, desc=desc)
with open(os.path.join(TAB, "emp_masak.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcc}\n\\toprule\n & Before & After \\\\\n & 28 Dec 2024 -- 27 Jun 2025 & 28 Jun -- 27 Dec 2025 \\\\\n\\midrule\n")
    c0, s0, _ = r_ev
    fh.write(f"$\\kappa^+$ & ${c0['up']:.4f}{star(c0['up'], s0['up'])}$ & ${c0['up'] + c0['up_post']:.4f}$ \\\\\n")
    fh.write(f" & $({s0['up']:.4f})$ & $\\Delta = {c0['up_post']:.4f}{star(c0['up_post'], s0['up_post'])}\\;({s0['up_post']:.4f})$ \\\\\n")
    fh.write(f"$\\kappa^-$ & ${c0['dn']:.4f}{star(c0['dn'], s0['dn'])}$ & ${c0['dn'] + c0['dn_post']:.4f}$ \\\\\n")
    fh.write(f" & $({s0['dn']:.4f})$ & $\\Delta = {c0['dn_post']:.4f}{star(c0['dn_post'], s0['dn_post'])}\\;({s0['dn_post']:.4f})$ \\\\\n")
    fh.write(f"$\\beta$ & ${c0['q']:.3f}{star(c0['q'], s0['q'])}$ & ${c0['q'] + c0['q_post']:.3f}$ \\\\\n")
    fh.write(f" & $({s0['q']:.3f})$ & $\\Delta = {c0['q_post']:.3f}{star(c0['q_post'], s0['q_post'])}\\;({s0['q_post']:.3f})$ \\\\\n")
    fh.write(f"Mean lower edge $\\ell_m$ (bp) & {bp[0]:.1f} & {bq[0]:.1f} \\\\\n")
    fh.write(f"Mean upper edge $u_m$ (bp) & {bp[1]:.1f} & {bq[1]:.1f} \\\\\n")
    fh.write(f"Mean premium (bp) & {desc['pre']['mean']:.1f} & {desc['post']['mean']:.1f} \\\\\n")
    fh.write(f"5th / 95th percentile of premium (bp) & {desc['pre']['p05']:.0f} / {desc['pre']['p95']:.0f} & {desc['post']['p05']:.0f} / {desc['post']['p95']:.0f} \\\\\n")
    fh.write(f"$N$ & {desc['pre']['n']:,} & {desc['post']['n']:,} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

json.dump(OUT, open(os.path.join(TAB, "emp_extra.json"), "w"), indent=1)
print(json.dumps(OUT, indent=1))

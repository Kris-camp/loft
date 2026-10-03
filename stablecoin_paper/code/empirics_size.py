"""Who trades when the premium leaves the band? Order flow split by trade size.

Uses data_raw/ofi_size/*.parquet (from aggtrades_ofi.py) and data_raw/a5.parquet.
Regression of standardized signed flow in each size class on the lagged excess premium:
the model predicts that arbitrage flow a* sells when b > u and buys when b < l, more so when
banks are open; end-user flow q has no such response.
"""
import os, glob, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="inner")
a["day"] = a.index.date
# lagged excess (the regressors in eq. emp use b_{t-1})
a["exc_up"] = np.maximum(a.x - a.u, 0)
a["exc_dn"] = np.maximum(a.l - a.x, 0)
for c in ["exc_up", "exc_dn"]:
    a[c + "_bh"] = a[c] * a.bh
CL = ["small", "mid", "large"]
for c in CL:
    s = a[f"ofi_{c}"].groupby(a.month).transform("std")
    a[f"z_{c}"] = a[f"ofi_{c}"] / s


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


cols = ["exc_up", "exc_dn", "exc_up_bh", "exc_dn_bh"]
res = {c: ols_fe(a, f"z_{c}", cols) for c in CL}
# price impact by size class
imp = ols_fe(a, "y", ["z_small", "z_mid", "z_large", "up", "dn"])
share = {c: float(a[f"vol_{c}"].sum() / sum(a[f"vol_{k}"].sum() for k in CL)) for c in CL}
out = {c: {k: [float(res[c][0][k]), float(res[c][1][k])] for k in cols} for c in CL}
out["impact"] = {k: [float(imp[0][k]), float(imp[1][k])] for k in imp[0]}
out["vol_share"] = share; out["n"] = int(len(a)); out["start"] = str(a.index.min().date()); out["end"] = str(a.index.max().date())
json.dump(out, open(os.path.join(TAB, "emp_size.json"), "w"), indent=1)

with open(os.path.join(TAB, "emp_size.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccc}\n\\toprule\n & \\multicolumn{3}{c}{Standardized signed order flow, by trade size} \\\\\n"
             "\\cmidrule(lr){2-4}\n & Small & Medium & Large \\\\\n & $<1$k USDT & 1k--10k & $\\ge 10$k \\\\\n\\midrule\n")
    labs = [("$(b_{t-1}-u_m)_+$", "exc_up"), ("$\\quad\\times$ banks open", "exc_up_bh"),
            ("$(\\ell_m-b_{t-1})_+$", "exc_dn"), ("$\\quad\\times$ banks open", "exc_dn_bh")]
    for lab, k in labs:
        fh.write(lab + " & " + " & ".join(f"${res[c][0][k]*100:.3f}{star(res[c][0][k], res[c][1][k])}$" for c in CL) + " \\\\\n")
        fh.write(" & " + " & ".join(f"$({res[c][1][k]*100:.3f})$" for c in CL) + " \\\\\n")
    fh.write("\\midrule\nShare of volume & " + " & ".join(f"{share[c]:.2f}" for c in CL) + " \\\\\n")
    fh.write("Price impact on $\\Delta b_t$ (bp per s.d.) & " + " & ".join(
        f"${imp[0]['z_'+c]:.3f}{star(imp[0]['z_'+c], imp[1]['z_'+c])}$" for c in CL) + " \\\\\n")
    fh.write(" & " + " & ".join(f"$({imp[1]['z_'+c]:.3f})$" for c in CL) + " \\\\\n")
    fh.write(f"$N$ & \\multicolumn{{3}}{{c}}{{{len(a):,}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
print(json.dumps(out, indent=1))

"""MASAK General Circular No. 29: difference-in-differences against USDT/MXN.

Treated market: USDT/TRY (Binance) vs interbank USD/TRY. Control market: USDT/MXN (Binance)
vs interbank USD/MXN (Dukascopy), a liquid local-currency stablecoin market not subject to
the Turkish rule. Window: 28 Dec 2024 -- 27 Dec 2025 (event 28 Jun 2025); the 19 Mar --
30 Apr 2025 lira shock is dropped from both markets. Both premia are built identically:
5-minute averages, month-specific bands, OFI standardized within market-month.
"""
import os, json
import numpy as np, pandas as pd
from model import _ssr_grid
from build_panel import load_binance, load_fx

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB = os.path.join(HERE, "..", "data_raw"), os.path.join(HERE, "..", "tables")
W0, W1, EV = "2024-12-28", "2025-12-27", pd.Timestamp("2025-06-28", tz="UTC")
X0, X1 = pd.Timestamp("2025-03-19", tz="UTC"), pd.Timestamp("2025-05-01", tz="UTC")


def build(local):
    u = load_binance(f"USDT{local}").loc[W0:W1]
    fx = load_fx(f"USD{local}")
    p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v}).join(fx.reindex(u.index, method="ffill", limit=2))
    p["b"] = 1e4 * np.log(p.pl / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    p.loc[(p.index >= X0) & (p.index < X1), "b"] = np.nan
    g = p.resample("5min")
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum()})
    a = a[a.nb >= 2]
    a["month"] = a.index.tz_localize(None).to_period("M").astype(str)
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    consec = a.index.to_series().diff() == pd.Timedelta("5min")
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    for m, d in a.groupby("month"):
        y, x, q = d.y.values, d.x.values, d.q.values
        lo_, hi_ = np.percentile(x, [5, 95])
        grid = np.arange(np.floor(lo_), np.ceil(hi_) + 1.0, 1.0)
        S = _ssr_grid(y, x, q, grid, -grid)
        i, j = np.unravel_index(np.argmin(S), S.shape)
        a.loc[d.index, "l"], a.loc[d.index, "u"] = grid[j], grid[i]
    a["up"] = -np.maximum(a.x - a.u, 0); a["dn"] = np.maximum(a.l - a.x, 0)
    a["mkt"] = local
    a["day"] = a.index.date
    a["post"] = (a.index >= EV).astype(float)
    return a


def ols_fe(d, ycol, cols, fe):
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


tr, mx = build("TRY"), build("MXN")
d = pd.concat([tr, mx])
d["tr"] = (d.mkt == "TRY").astype(float)
d["mm"] = d.mkt + "_" + d.month
cols = []
for c in ["q", "up", "dn"]:
    for k, v in {"": 1.0, "_tr": d.tr}.items():
        for kk, vv in {"": 1.0, "_post": d.post}.items():
            name = c + k + kk
            d[name] = d[c] * v * vv; cols.append(name)
r = ols_fe(d, "y", cols, fe="mm")
# premium level DiD: daily mean premium on market and post dummies, market FE
day = d.groupby(["mkt", "day"]).agg(x=("x", "mean"), tr=("tr", "first"), post=("post", "first")).reset_index()
day["trpost"] = day.tr * day.post
Xd = np.column_stack([np.ones(len(day)), day.tr, day.post, day.trpost])
bd = np.linalg.lstsq(Xd, day.x.values, rcond=None)[0]; ed = day.x.values - Xd @ bd
wk = pd.to_datetime(day.day).dt.to_period("W").astype(str)
codes = pd.factorize(wk)[0]
G = np.zeros((codes.max() + 1, 4)); np.add.at(G, codes, Xd * ed[:, None])
XtXi = np.linalg.inv(Xd.T @ Xd); Vd = XtXi @ G.T @ G @ XtXi
sed = np.sqrt(np.diag(Vd))

out = {"coef": {k: [float(r[0][k]), float(r[1][k])] for k in cols}, "n": r[2],
       "level": {"trpost": [float(bd[3]), float(sed[3])], "post": [float(bd[2]), float(sed[2])]},
       "n_try": int(len(tr)), "n_mxn": int(len(mx)),
       "mxn_premium_sd": float(mx.x.std()), "mxn_premium_mean": float(mx.x.mean())}
json.dump(out, open(os.path.join(TAB, "emp_did.json"), "w"), indent=1)

with open(os.path.join(TAB, "emp_did.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccc}\n\\toprule\n & Control (MXN): & Treated (TRY): & Difference-in- \\\\\n"
             " & change after & change after & differences \\\\\n & 28 Jun 2025 & 28 Jun 2025 & \\\\\n\\midrule\n")
    for lab, c in [("$\\kappa^+$", "up"), ("$\\kappa^-$", "dn"), ("$\\beta$", "q")]:
        nd = 3 if c == "q" else 4
        ctrl = r[0][c + "_post"]; sc = r[1][c + "_post"]
        did = r[0][c + "_tr_post"]; sdid = r[1][c + "_tr_post"]
        # treated change = control change + DiD; SE via the covariance is omitted, report the two estimated terms
        fh.write(f"{lab} & ${ctrl:.{nd}f}{star(ctrl, sc)}$ & ${ctrl + did:.{nd}f}$ & ${did:.{nd}f}{star(did, sdid)}$ \\\\\n")
        fh.write(f" & $({sc:.{nd}f})$ & & $({sdid:.{nd}f})$ \\\\\n")
    lv, sl = out["level"]["trpost"]; lp, slp = out["level"]["post"]
    fh.write(f"Mean premium (bp) & ${lp:.1f}{star(lp, slp)}$ & ${lp + lv:.1f}$ & ${lv:.1f}{star(lv, sl)}$ \\\\\n")
    fh.write(f" & $({slp:.1f})$ & & $({sl:.1f})$ \\\\\n")
    fh.write(f"\\midrule\n$N$ (five-minute intervals) & {len(mx):,} & {len(tr):,} & {r[2]:,} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
print(json.dumps(out, indent=1))

"""Cross-country contrast: Turkey (large-value EFT closes after business hours) vs Mexico
(SPEI, the central-bank RTGS system, runs 24/7/365).

Identical construction for both markets over the common window November 2024 -- August 2026:
Binance USDT/local minute VWAP vs Dukascopy USD/local quotes, 5-minute averages, bands by
month and local banking state, month-by-state effects, day-clustered SEs.
Local business hours 09:00-17:00 (Istanbul UTC+3; Mexico City UTC-6, no DST since 2022).
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import _ssr_grid
from build_panel import load_binance, load_fx

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
W0, W1 = "2024-11-01", "2026-08-31"
UTC_OFF = {"TRY": 3, "MXN": -6}


def profile(d, trim=0.05):
    y, x, q = d.y.values, d.x.values, d.q.values
    lo_, hi_ = np.percentile(x, [100 * trim, 100 * (1 - trim)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + 1.0, 1.0)
    S = _ssr_grid(y, x, q, grid, -grid)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[j], grid[i]


def build(local):
    u = load_binance(f"USDT{local}").loc[W0:W1]
    fx = load_fx(f"USD{local}")
    p = pd.DataFrame({"pl": u.vwap, "ofi": 2 * u.tbv - u.v, "vol": u.v}).join(fx.reindex(u.index, method="ffill", limit=2))
    p["b"] = 1e4 * np.log(p.pl / p.fx_mid)
    hr = p.index.hour + p.index.minute / 60
    p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan
    p.loc[p.index.dayofweek >= 5, "b"] = np.nan
    p.loc[p.b.abs() > 1000, "b"] = np.nan
    g = p.resample("5min")
    a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(),
                      "pl": np.log(g.pl.mean()), "pg": np.log(g.fx_mid.mean())})
    a = a[a.nb >= 2]
    a["month"] = a.index.tz_localize(None).to_period("M").astype(str)
    a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
    a["dpl"] = 1e4 * a.pl.diff(); a["dpg"] = 1e4 * a.pg.diff()
    consec = a.index.to_series().diff() == pd.Timedelta("5min")
    a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
    a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
    lh = (a.index.hour + UTC_OFF[local]) % 24
    a["lh"] = lh
    a["bh"] = ((lh >= 9) & (lh < 17)).astype(float)
    a["state"] = np.where(a.bh == 1, "open", "closed")
    a["ms"] = a.month + "_" + a.state
    for key, d in a.groupby("ms"):
        if len(d) < 300: continue
        l, u_ = profile(d)
        a.loc[d.index, "l"], a.loc[d.index, "u"] = l, u_
    a = a[a.l.notna()].copy()
    m = (a.l + a.u) / 2
    a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - m)))
    a["d_out"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
    a["d_in"] = np.where(a.d_out == 0, np.abs(a.x - m), 0.0)
    a["up"] = -np.maximum(a.x - a.u, 0); a["dn"] = np.maximum(a.l - a.x, 0)
    a["day"] = a.index.date
    a["mkt"] = local
    return a


def ols(d, ycol, cols, fe="ms"):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    if fe:
        gm = d.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V = XtXi @ (G.T @ G) @ XtXi * ng / (ng - 1) * (n - 1) / (n - k)
    return dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), n=n, V=V, cols=cols)


def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


OUT = {}
A = {k: build(k) for k in ["TRY", "MXN"]}
res = {}
for k, a in A.items():
    beta = ols(a, "y", ["q", "up", "dn"])["b"]["q"]
    a["R"] = -a.s * (a.y - beta * a.q)
    for c in ["d_out", "d_in", "up", "dn", "q"]:
        a[c + "_bh"] = a[c] * a.bh
    r_sym = ols(a, "R", ["d_out", "d_out_bh", "d_in", "d_in_bh"])
    r_asym = ols(a, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
    res[k] = dict(sym=r_sym, asym=r_asym)
    w = a.groupby(["month", "state"]).agg(l=("l", "first"), u=("u", "first")).reset_index()
    w["w"] = w.u - w.l
    OUT[k] = dict(n=int(len(a)), days=int(a.day.nunique()), sd_prem=float(a.x.std()), mean_prem=float(a.x.mean()),
                  share_out=float((a.d_out > 0).mean()),
                  width_open=float(w[w.state == "open"].w.mean()), width_closed=float(w[w.state == "closed"].w.mean()),
                  sym={c: [float(r_sym["b"][c]), float(r_sym["se"][c])] for c in r_sym["cols"]},
                  asym={c: [float(r_asym["b"][c]), float(r_asym["se"][c])] for c in r_asym["cols"]},
                  vol_per_min=float(a.vol.mean() / 5))

# difference in the business-hours increment across markets (independent samples)
dd = res["TRY"]["sym"]["b"]["d_out_bh"] - res["MXN"]["sym"]["b"]["d_out_bh"]
sd = np.sqrt(res["TRY"]["sym"]["se"]["d_out_bh"] ** 2 + res["MXN"]["sym"]["se"]["d_out_bh"] ** 2)
OUT["diff_bh"] = [float(dd), float(sd)]
dd2 = res["TRY"]["asym"]["b"]["up_bh"] - res["MXN"]["asym"]["b"]["up_bh"]
sd2 = np.sqrt(res["TRY"]["asym"]["se"]["up_bh"] ** 2 + res["MXN"]["asym"]["se"]["up_bh"] ** 2)
OUT["diff_bh_up"] = [float(dd2), float(sd2)]

# daily cycle by local two-hour block
cyc = {}
for k, a in A.items():
    rows = []
    for h0 in range(0, 24, 2):
        d = a[(a.lh >= h0) & (a.lh < h0 + 2)]
        if (d.d_out > 0).sum() < 200: continue
        r = ols(d, "R", ["d_out", "d_in"], fe="month")
        rows.append(dict(h=h0 + 1, k=r["b"]["d_out"], se=r["se"]["d_out"], n=r["n"]))
    cyc[k] = rows
OUT["cycle"] = cyc
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for k, col, mk, lab in [("TRY", C1, "o", "Turkey: large-value EFT closes after hours"),
                        ("MXN", C2, "s", "Mexico: SPEI runs 24/7")]:
    c = pd.DataFrame(cyc[k])
    ax.errorbar(c.h + (0 if k == "TRY" else 0.3), c.k, yerr=1.96 * c.se, fmt=mk + "-", ms=3.5, color=col, ecolor=col, lw=0.9, alpha=0.9, label=lab)
ax.axvspan(9, 17, color=C3, alpha=0.12); ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("local hour"); ax.set_ylabel(r"reversion speed outside the band, $\hat\kappa$")
ax.set_xticks(range(0, 25, 4)); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_mxn_cycle.pdf")); plt.close(fig)

with open(os.path.join(TAB, "emp_mxn.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccc}\n\\toprule\n & Turkey & Mexico & Difference \\\\\n"
             " & (EFT closes after hours) & (SPEI 24/7) & \\\\\n\\midrule\n")
    labs = [("Speed outside band, banks closed", "d_out"), ("$\\quad\\times$ business hours", "d_out_bh"),
            ("In-band placebo, banks closed", "d_in"), ("$\\quad\\times$ business hours", "d_in_bh")]
    for lab, c in labs:
        t, m_ = res["TRY"]["sym"], res["MXN"]["sym"]
        diff = f"${t['b'][c]-m_['b'][c]:.4f}{star(t['b'][c]-m_['b'][c], np.hypot(t['se'][c], m_['se'][c]))}$" if c == "d_out_bh" else ""
        dse = f"$({np.hypot(t['se'][c], m_['se'][c]):.4f})$" if c == "d_out_bh" else ""
        fh.write(f"{lab} & ${t['b'][c]:.4f}{star(t['b'][c], t['se'][c])}$ & ${m_['b'][c]:.4f}{star(m_['b'][c], m_['se'][c])}$ & {diff} \\\\\n")
        fh.write(f" & $({t['se'][c]:.4f})$ & $({m_['se'][c]:.4f})$ & {dse} \\\\\n")
    fh.write("\\midrule\n")
    fh.write(f"Above band only: $\\kappa^+\\times$ business hours & ${res['TRY']['asym']['b']['up_bh']:.4f}{star(res['TRY']['asym']['b']['up_bh'], res['TRY']['asym']['se']['up_bh'])}$ & "
             f"${res['MXN']['asym']['b']['up_bh']:.4f}{star(res['MXN']['asym']['b']['up_bh'], res['MXN']['asym']['se']['up_bh'])}$ & "
             f"${dd2:.4f}{star(dd2, sd2)}$ \\\\\n")
    fh.write(f" & $({res['TRY']['asym']['se']['up_bh']:.4f})$ & $({res['MXN']['asym']['se']['up_bh']:.4f})$ & $({sd2:.4f})$ \\\\\n")
    fh.write(f"SD of premium (bp) & {OUT['TRY']['sd_prem']:.1f} & {OUT['MXN']['sd_prem']:.1f} & \\\\\n")
    fh.write(f"Mean band width, open / closed hours (bp) & {OUT['TRY']['width_open']:.0f} / {OUT['TRY']['width_closed']:.0f} & {OUT['MXN']['width_open']:.0f} / {OUT['MXN']['width_closed']:.0f} & \\\\\n")
    fh.write(f"USDT volume per minute (thousand) & {OUT['TRY']['vol_per_min']/1e3:.1f} & {OUT['MXN']['vol_per_min']/1e3:.1f} & \\\\\n")
    fh.write(f"$N$ & {OUT['TRY']['n']:,} & {OUT['MXN']['n']:,} & \\\\\n\\bottomrule\n\\end{{tabular}}\n")
json.dump(OUT, open(os.path.join(TAB, "emp_mxn.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "cycle"}, indent=1, default=float))
for k in cyc: print(k); print(pd.DataFrame(cyc[k]).round(4).to_string())

# log open/closed ratio by market (delta method) and cross-market difference
for k in A:
    r = res[k]["sym"]; i0, i1 = r["cols"].index("d_out"), r["cols"].index("d_out_bh")
    c0, c1 = r["b"]["d_out"], r["b"]["d_out_bh"]
    g = np.zeros(len(r["cols"])); g[i0] = 1 / (c0 + c1) - 1 / c0; g[i1] = 1 / (c0 + c1)
    OUT[k]["ratio"] = float((c0 + c1) / c0); OUT[k]["log_ratio"] = float(np.log((c0 + c1) / c0))
    OUT[k]["se_log_ratio"] = float(np.sqrt(g @ r["V"] @ g))
dl = OUT["TRY"]["log_ratio"] - OUT["MXN"]["log_ratio"]
sdl = float(np.hypot(OUT["TRY"]["se_log_ratio"], OUT["MXN"]["se_log_ratio"]))
OUT["diff_log_ratio"] = [float(dl), sdl]
# night-time share of daytime speed
for k in A:
    c = pd.DataFrame(cyc[k]); night = c[c.h.isin([1, 3, 5, 7])].k.mean(); day = c[c.h.isin([11, 13, 15])].k.mean()
    OUT[k]["night_over_day"] = float(night / day)
json.dump(OUT, open(os.path.join(TAB, "emp_mxn.json"), "w"), indent=1, default=float)
with open(os.path.join(TAB, "emp_mxn.tex")) as fh:
    t = fh.read()
add = (f"Open/closed ratio of the speed outside the band & {OUT['TRY']['ratio']:.1f} & {OUT['MXN']['ratio']:.1f} & \\\\\n"
       f"$\\log$ ratio & ${OUT['TRY']['log_ratio']:.2f}$ & ${OUT['MXN']['log_ratio']:.2f}$ & ${dl:.2f}{star(dl, sdl)}$ \\\\\n"
       f" & $({OUT['TRY']['se_log_ratio']:.2f})$ & $({OUT['MXN']['se_log_ratio']:.2f})$ & $({sdl:.2f})$ \\\\\n")
t = t.replace("SD of premium (bp)", add + "SD of premium (bp)", 1)
open(os.path.join(TAB, "emp_mxn.tex"), "w").write(t)
print("RATIOS", {k: (OUT[k]["ratio"], OUT[k]["log_ratio"], OUT[k]["se_log_ratio"], OUT[k]["night_over_day"]) for k in A}, OUT["diff_log_ratio"])

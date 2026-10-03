"""Mechanism tests: flow-side arbitrage response, price-flow cross-equation test,
bank-opening event time, bank-holiday placebo, and the 24/7-banking counterfactual.

Requires data_raw/a5.parquet (empirics_main.py) and data_raw/ofi_size/*.parquet (aggtrades_ofi.py).
Notation: m = band midpoint; s = side of the band (+1 above u, -1 below l, sign(x-m) inside);
d_out = distance outside the band; d_in = distance from the midpoint for in-band observations.
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
OUT = {}

a = pd.read_parquet(os.path.join(RAW, "a5.parquet"))
z = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "ofi_size", "*.parquet")))])
z = z[~z.index.duplicated()]
a = a.join(z, how="left")
a["day"] = a.index.date
BANDS = os.environ.get("BANDS", "state")
if BANDS == "state":
    sb = pd.read_parquet(os.path.join(RAW, "state_bands.parquet"))
    a = a.join(sb, how="inner")
    a["l"], a["u"] = a.l_s, a.u_s
OUT["bands"] = BANDS
for c in ["small", "mid", "large"]:
    a[f"z_{c}"] = a[f"ofi_{c}"] / a[f"ofi_{c}"].groupby(a.month).transform("std")
a["z_dem"] = (a.ofi_small + a.ofi_mid) / (a.ofi_small + a.ofi_mid).groupby(a.month).transform("std")
m = (a.l + a.u) / 2
a["s"] = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - m)))
a["out"] = ((a.x > a.u) | (a.x < a.l)).astype(float)
a["d_out"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
a["d_in"] = np.where(a.out == 0, np.abs(a.x - m), 0.0)

# Turkish public holidays falling on weekdays (banks closed; EFT closed), 2024-2026
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
hol = pd.to_datetime(HOL).date
a["hol"] = np.isin(a.day, hol).astype(float)
a["bh_eff"] = a.bh * (1 - a.hol)          # banks actually open
a["bh_hol"] = a.bh * a.hol                # clock says open, banks closed
OUT["n_holidays_in_sample"] = int(a.loc[a.hol == 1, "day"].nunique())

# price-side reversion net of order flow (beta from the main estimate)
beta = json.load(open(os.path.join(TAB, "emp_numbers.json")))["reg"]["thr"]["coef"]["q"]
a["R"] = -a.s * (a.y - beta * a.q)


def ols(d, ycol, cols, fe="month", ret_scores=False):
    Y = d[ycol].values.astype(float); X = d[cols].values.astype(float)
    if fe:
        gm = d.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values; X = X - gm[cols].transform("mean").values
    else:
        X = np.column_stack([np.ones(len(Y)), X]); cols = ["const"] + list(cols)
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    adj = ng / (ng - 1) * (n - 1) / (n - k)
    V = XtXi @ (G.T @ G) @ XtXi * adj
    res = dict(b=dict(zip(cols, b)), se=dict(zip(cols, np.sqrt(np.diag(V)))), n=n)
    if ret_scores:
        res.update(G=G @ XtXi.T, codes_index=pd.factorize(d.day)[1], adj=adj, cols=cols)
    return res


def star(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


for c in ["d_out", "d_in"]:
    a[c + "_bh"] = a[c] * a.bh_eff
    a[c + "_hol"] = a[c] * a.bh_hol
cols = ["d_out", "d_out_bh", "d_in", "d_in_bh"]

# ------------------------------------------------------------------ 1. flow-side response
fl = a.dropna(subset=["z_large", "z_dem"]).copy()
# forward 30-minute flow (six consecutive intervals)
idx = fl.index
consec = pd.Series(idx, index=idx).diff().eq(pd.Timedelta("5min"))
seg = (~consec).cumsum()
for c in ["z_large", "z_dem"]:
    fwd = fl.groupby(seg)[c].transform(lambda s: s[::-1].rolling(6, min_periods=6).sum()[::-1])
    fl[c + "_h30"] = fwd
fl["A_L"] = -fl.s * fl.z_large
fl["A_D"] = -fl.s * fl.z_dem
fl["A_L30"] = -fl.s * fl.z_large_h30
fl["A_D30"] = -fl.s * fl.z_dem_h30
flow = {k: ols(fl.dropna(subset=[k]), k, cols) for k in ["A_L", "A_D", "A_L30", "A_D30"]}
price = ols(fl, "R", cols)
OUT["flow"] = {k: {c: [float(v["b"][c]), float(v["se"][c])] for c in cols} for k, v in flow.items()}
OUT["price"] = {c: [float(price["b"][c]), float(price["se"][c])] for c in cols}

# ------------------------------------------------------------------ 2. cross-equation test
# joint covariance of (kappa_C, kappa_BH) from R and (theta_C, theta_BH) from A_L on the same sample
jj = fl.dropna(subset=["A_L"]).copy()
pr = ols(jj, "R", cols, ret_scores=True)
fw = ols(jj, "A_L", cols, ret_scores=True)
G = np.hstack([pr["G"], fw["G"]])                    # per-day influence contributions
V = G.T @ G * pr["adj"]
k = len(cols)
bvec = np.r_[[pr["b"][c] for c in cols], [fw["b"][c] for c in cols]]
iC, iB = cols.index("d_out"), cols.index("d_out_bh")
kC, kB, tC, tB = bvec[iC], bvec[iB], bvec[k + iC], bvec[k + iB]
# ratios open/closed and H0: (kC+kB)/kC = (tC+tB)/tC  <=>  g = kB*tC - tB*kC = 0
g = kB * tC - tB * kC
grad = np.zeros(2 * k); grad[iB] = tC; grad[k + iC] = kB; grad[k + iB] = -kC; grad[iC] = -tB
se_g = float(np.sqrt(grad @ V @ grad))
rk, rt = (kC + kB) / kC, (tC + tB) / tC
# delta-method SEs of log ratios
def logratio_se(i0, i1):
    gr = np.zeros(2 * k); c0, c1 = bvec[i0], bvec[i1]
    gr[i0] = 1 / (c0 + c1) - 1 / c0; gr[i1] = 1 / (c0 + c1)
    return float(np.sqrt(gr @ V @ gr))
OUT["cross"] = dict(kappa_closed=float(kC), kappa_open=float(kC + kB), theta_closed=float(tC), theta_open=float(tC + tB),
                    ratio_price=float(rk), ratio_flow=float(rt), se_log_ratio_price=logratio_se(iC, iB),
                    se_log_ratio_flow=logratio_se(k + iC, k + iB), g=float(g), se_g=se_g, z=float(g / se_g),
                    n=int(len(jj)))
# also the log-ratio difference test
gr = np.zeros(2 * k)
gr[iC] = 1 / (kC + kB) - 1 / kC; gr[iB] = 1 / (kC + kB)
gr[k + iC] = -(1 / (tC + tB) - 1 / tC); gr[k + iB] = -1 / (tC + tB)
dl = np.log(rk) - np.log(rt); se_dl = float(np.sqrt(gr @ V @ gr))
OUT["cross"].update(logdiff=float(dl), se_logdiff=se_dl, z_logdiff=float(dl / se_dl))

with open(os.path.join(TAB, "emp_flow.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccc}\n\\toprule\n & Price side & \\multicolumn{2}{c}{Large trades} & \\multicolumn{2}{c}{Small and medium trades} \\\\\n"
             "\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
             " & $-s_t(\\Delta b-\\hat\\beta\\,\\mathit{OFI})$ & next 5 min & next 30 min & next 5 min & next 30 min \\\\\n"
             " & (1) & (2) & (3) & (4) & (5) \\\\\n\\midrule\n")
    labs = [("Distance outside band, $d^{out}$", "d_out"), ("$\\quad\\times$ banks open", "d_out_bh"),
            ("Distance inside band, $d^{in}$ (placebo)", "d_in"), ("$\\quad\\times$ banks open", "d_in_bh")]
    regs = [price, flow["A_L"], flow["A_L30"], flow["A_D"], flow["A_D30"]]
    scale = [1, 100, 100, 100, 100]
    for lab, c in labs:
        fh.write(lab + " & " + " & ".join(f"${r['b'][c]*sc:.4f}{star(r['b'][c], r['se'][c])}$" if sc == 1 else
                                         f"${r['b'][c]*sc:.3f}{star(r['b'][c], r['se'][c])}$" for r, sc in zip(regs, scale)) + " \\\\\n")
        fh.write(" & " + " & ".join(f"$({r['se'][c]*sc:.4f})$" if sc == 1 else f"$({r['se'][c]*sc:.3f})$" for r, sc in zip(regs, scale)) + " \\\\\n")
    cr = OUT["cross"]
    fh.write("\\midrule\nOpen/closed ratio of the $d^{out}$ response & " + f"{cr['ratio_price']:.1f} & {cr['ratio_flow']:.1f} & & & \\\\\n")
    fh.write(f"Test of equal ratios, $\\log$ difference (s.e.) & \\multicolumn{{2}}{{c}}{{{cr['logdiff']:.2f} ({cr['se_logdiff']:.2f})}} & & & \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{r['n']:,}" for r in regs) + " \\\\\n\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ 3. event time around bank opening/closing
def bin_kappa(d, ycol):
    r = ols(d, ycol, ["d_out", "d_in"], fe=None)
    return r["b"]["d_out"], r["se"]["d_out"], r["b"]["d_in"], r["se"]["d_in"]


utc_min = a.index.hour * 60 + a.index.minute
ev = {}
for name, center in [("open", 6 * 60), ("close", 14 * 60)]:       # 09:00 and 17:00 Istanbul = 06:00, 14:00 UTC
    rows = []
    for lo in range(-180, 180, 30):
        msk = (utc_min >= center + lo) & (utc_min < center + lo + 30)
        for grp, gm in [("normal", a.hol == 0), ("holiday", a.hol == 1)]:
            d = a[msk & gm]
            if len(d) < 150 or d.out.sum() < 50: continue
            kp, sp, ki, si = bin_kappa(d, "R")
            dd = fl[(fl.index.isin(d.index))].dropna(subset=["A_L"])
            th, st = (ols(dd, "A_L", ["d_out", "d_in"], fe=None)["b"]["d_out"], ols(dd, "A_L", ["d_out", "d_in"], fe=None)["se"]["d_out"]) if len(dd) > 150 else (np.nan, np.nan)
            rows.append(dict(t=lo + 15, grp=grp, kappa=kp, se=sp, theta=th, se_theta=st, n=len(d)))
    ev[name] = rows
OUT["event_time"] = ev
# holiday placebo on the full specification
hc = ["d_out", "d_out_bh", "d_out_hol", "d_in", "d_in_bh", "d_in_hol"]
hp = ols(a, "R", hc)
hf = ols(fl.assign(d_out_hol=fl.d_out * fl.bh_hol, d_in_hol=fl.d_in * fl.bh_hol).dropna(subset=["A_L"]), "A_L", hc)
OUT["holiday"] = dict(price={c: [float(hp["b"][c]), float(hp["se"][c])] for c in hc},
                      flow={c: [float(hf["b"][c]), float(hf["se"][c])] for c in hc})

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
for j, name in enumerate(["open", "close"]):
    R = pd.DataFrame(ev[name])
    for grp, col, mk in [("normal", C1, "o"), ("holiday", C2, "s")]:
        r = R[R.grp == grp]
        if len(r):
            off = 0 if grp == "normal" else 4
            ax[j].errorbar(r.t + off, r.kappa, yerr=1.96 * r.se, fmt=mk + "-", ms=3.5, color=col, ecolor=col, lw=0.9, alpha=0.9,
                           label="weekday" if grp == "normal" else "weekday public holiday")
    ax[j].axvline(0, color="k", lw=0.6, ls="--"); ax[j].axhline(0, color="k", lw=0.4)
    ax[j].set_xlabel("minutes relative to " + ("09:00 (banks open)" if name == "open" else "17:00 (banks close)"))
    ax[j].set_title(("(a) Opening" if name == "open" else "(b) Closing"), fontsize=10)
ax[0].set_ylabel(r"reversion speed outside the band, $\hat\kappa$")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_eventtime.pdf")); plt.close(fig)

# ------------------------------------------------------------------ 4. counterfactual: 24/7 banking
# Simulate the premium on the observed sample with observed order flow and bootstrapped residuals.
# Baseline: state-specific speeds; counterfactual: open-hours speeds at all times. Bands, beta,
# month effects and residuals are held fixed.
st = json.load(open(os.path.join(TAB, "emp_state.json")))["state_bands"]["coef"]
kp_c, kp_o = st["up_s"][0], st["up_s"][0] + st["up_s_bh"][0]
km_c, km_o = st["dn_s"][0], st["dn_s"][0] + st["dn_s_bh"][0]
b_q = st["q"][0]
sim = a[["x", "y", "q", "l", "u", "bh", "month", "day"]].copy()
up = -np.maximum(sim.x - sim.u, 0); dn = np.maximum(sim.l - sim.x, 0)
kp = np.where(sim.bh == 1, kp_o, kp_c); km = np.where(sim.bh == 1, km_o, km_c)
resid = sim.y - b_q * sim.q - kp * up - km * dn
alpha = resid.groupby(sim.month).transform("mean")
eps = (resid - alpha).values
newseg = np.r_[True, np.diff(sim.index.values).astype("timedelta64[s]").astype(int) != 300]
q = sim.q.values; L = sim.l.values; U = sim.u.values; BH = sim.bh.values; al = alpha.values; x0 = sim.x.values
dem = (a.vol_small.fillna(0) + a.vol_mid.fillna(0)).values


def simulate(kp_open, kp_closed, km_open, km_closed, seed):
    rng = np.random.default_rng(seed)
    e = eps[rng.integers(0, len(eps), len(eps))]
    out = np.empty(len(q)); b = x0[0]
    for t in range(len(q)):
        if newseg[t]: b = x0[t]
        out[t] = b
        kpp = kp_open if BH[t] == 1 else kp_closed
        kmm = km_open if BH[t] == 1 else km_closed
        b = b + al[t] + b_q * q[t] - kpp * max(b - U[t], 0) + kmm * max(L[t] - b, 0) + e[t]
    return out


try:
    from numba import njit
    @njit(cache=True)
    def _sim(q, L, U, BH, al, x0, newseg, e, b_q, kpo, kpc, kmo, kmc):
        out = np.empty(len(q)); b = x0[0]
        for t in range(len(q)):
            if newseg[t]: b = x0[t]
            out[t] = b
            kpp = kpo if BH[t] == 1 else kpc
            kmm = kmo if BH[t] == 1 else kmc
            b = b + al[t] + b_q * q[t] - kpp * max(b - U[t], 0.0) + kmm * max(L[t] - b, 0.0) + e[t]
        return out

    def simulate(kpo, kpc, kmo, kmc, seed):
        rng = np.random.default_rng(seed)
        e = eps[rng.integers(0, len(eps), len(eps))]
        return _sim(q, L, U, BH.astype(np.float64), al, x0, newseg, e, b_q, kpo, kpc, kmo, kmc)
except Exception:
    pass


def stats(path):
    exc = np.maximum(path - U, 0) + np.maximum(L - path, 0)
    mid = (L + U) / 2
    out = exc > 0
    # durations of excursions outside the band (in minutes)
    runs, cur = [], 0
    for o, ns in zip(out, newseg):
        if ns and cur: runs.append(cur); cur = 0
        if o: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    return dict(sd=float(np.std(path - mid)), p50=float(np.mean(exc > 50)), mean_exc=float(np.mean(exc)),
                dur=float(np.mean(runs) * 5 if runs else 0.0), wedge=float(np.sum(dem * exc) / np.sum(dem)))


R_SIM = 40
base = [stats(simulate(kp_o, kp_c, km_o, km_c, s)) for s in range(R_SIM)]
cf = [stats(simulate(kp_o, kp_o, km_o, km_o, s)) for s in range(R_SIM)]
data = stats(sim.x.values)
agg = lambda L_: {k: float(np.mean([d[k] for d in L_])) for k in L_[0]}
B_, C_ = agg(base), agg(cf)
OUT["counterfactual"] = dict(data=data, baseline=B_, cf=C_,
                             change={k: float(C_[k] / B_[k] - 1) for k in B_}, R=R_SIM)
with open(os.path.join(TAB, "emp_cf.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & Data & Model: & Counterfactual: & Change \\\\\n"
             " & & observed banking hours & banks always open & \\\\\n\\midrule\n")
    rows = [("SD of premium around band midpoint (bp)", "sd", "{:.1f}"), ("Share of time $>50$ bp outside band", "p50", "{:.3f}"),
            ("Mean distance outside band (bp)", "mean_exc", "{:.1f}"), ("Mean duration of an excursion (min)", "dur", "{:.0f}"),
            ("Volume-weighted wedge on small/medium trades (bp)", "wedge", "{:.1f}")]
    for lab, k_, f in rows:
        fh.write(f"{lab} & {f.format(data[k_])} & {f.format(B_[k_])} & {f.format(C_[k_])} & {100*(C_[k_]/B_[k_]-1):.0f}\\% \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

json.dump(OUT, open(os.path.join(TAB, "emp_mech.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "event_time"}, indent=1, default=float))
for name in ev:
    print(name); print(pd.DataFrame(ev[name]).round(4).to_string())

# ------------------------------------------------------------------ 5. daily cycle with holiday placebo
ih = (a.index.hour + 3) % 24
cyc = []
for h0 in range(0, 24, 2):
    msk = (ih >= h0) & (ih < h0 + 2) & (a.hol == 0)
    d = a[msk]
    if d.out.sum() < 200: continue
    r = ols(d, "R", ["d_out", "d_in"], fe="month")
    dd = fl[fl.index.isin(d.index)].dropna(subset=["A_L"])
    rf = ols(dd, "A_L", ["d_out", "d_in"], fe="month")
    cyc.append(dict(h=h0 + 1, k=r["b"]["d_out"], ks=r["se"]["d_out"], t=rf["b"]["d_out"] * 100, ts=rf["se"]["d_out"] * 100))
d = a[(a.hol == 1) & (a.bh == 1)]
r = ols(d, "R", ["d_out", "d_in"], fe="month")
rf = ols(fl[fl.index.isin(d.index)].dropna(subset=["A_L"]), "A_L", ["d_out", "d_in"], fe="month")
hol_pt = dict(k=r["b"]["d_out"], ks=r["se"]["d_out"], t=rf["b"]["d_out"] * 100, ts=rf["se"]["d_out"] * 100, n=r["n"])
OUT["cycle"] = cyc; OUT["holiday_daytime"] = hol_pt
CY = pd.DataFrame(cyc)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for j, (col, scol, lab) in enumerate([("k", "ks", r"price side: $\hat\kappa$ per 5 min"), ("t", "ts", r"large-trade flow toward the band, $\hat\theta$")]):
    ax[j].axvspan(9, 17, color=C2, alpha=0.08)
    ax[j].errorbar(CY.h, CY[col], yerr=1.96 * CY[scol], fmt="o-", ms=3.5, color=C1, ecolor=C3, lw=0.9, label="weekdays")
    ax[j].errorbar([13], [hol_pt[col]], yerr=[1.96 * hol_pt[scol]], fmt="s", ms=6, color=C2, ecolor=C2, label="public holidays, 09-17")
    ax[j].axhline(0, color="k", lw=0.4)
    ax[j].set_xlabel("Istanbul hour"); ax[j].set_title(("(a) " if j == 0 else "(b) ") + lab, fontsize=9.5)
    ax[j].set_xticks(range(0, 25, 4))
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_cycle.pdf")); plt.close(fig)

# ------------------------------------------------------------------ 6. four-panel summary figure
pos = np.where(a.x > a.u, a.x - a.u, np.where(a.x < a.l, a.x - a.l, 0.0))
th = json.load(open(os.path.join(TAB, "emp_state.json")))["state_bands"]["coef"]
yy = a.y - beta * a.q
yy = (yy - yy.groupby(a.month).transform("mean")).values
EDG = np.r_[-60, -40, -25, -15, -8, -3, -1e-9, 1e-9, 3, 8, 15, 25, 40, 60]


def binned(v, mask):
    xs, ms, ss = [], [], []
    for lo_, hi_ in zip(EDG[:-1], EDG[1:]):
        mm = mask & ((pos > lo_) & (pos <= hi_) if lo_ != -1e-9 else (pos == 0)) & np.isfinite(v)
        if mm.sum() < 150: continue
        xs.append(pos[mm].mean()); ms.append(np.nanmean(v[mm]))
        gs = pd.Series(v[mm] - np.nanmean(v[mm])).groupby(a.day.values[mm]).sum()
        ss.append(np.sqrt(np.sum(gs ** 2)) / mm.sum())
    return np.array(xs), np.array(ms), np.array(ss)


allm = np.ones(len(a), bool)
op = (a.bh_eff == 1).values; cl = (a.bh == 0).values
gx = np.linspace(-50, 50, 200)
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.0))
# A
x_, m_, s_ = binned(yy, allm)
A = ax[0, 0]; A.axvspan(-1.5, 1.5, color=C3, alpha=0.25)
A.errorbar(x_, m_, yerr=1.96 * s_, fmt="o", ms=3.5, color=C1, ecolor=C3, lw=0.8)
Xa = np.column_stack([np.maximum(pos, 0), np.maximum(-pos, 0)])
Xa = Xa - pd.DataFrame(Xa).groupby(a.month.values).transform("mean").values
ca = np.linalg.lstsq(Xa, yy, rcond=None)[0]
A.plot(gx, ca[0] * np.maximum(gx, 0) + ca[1] * np.maximum(-gx, 0), color=C2, lw=1.1)
A.axhline(0, color="k", lw=0.4); A.set_title("(a) Arbitrage switches on outside the band", fontsize=9.5)
A.set_ylabel("expected 5-min premium change (bp)")
# B
B = ax[0, 1]; B.axvspan(-1.5, 1.5, color=C3, alpha=0.25)
for mask, col, mk, lab, kp_, km_ in [(op, C1, "o", "banks open", th["up_s"][0] + th["up_s_bh"][0], th["dn_s"][0] + th["dn_s_bh"][0]),
                                     (cl, C2, "s", "banks closed", th["up_s"][0], th["dn_s"][0])]:
    x_, m_, s_ = binned(yy, mask)
    B.errorbar(x_, m_, yerr=1.96 * s_, fmt=mk, ms=3.2, color=col, ecolor=col, alpha=0.85, lw=0.6, label=lab)
    B.plot(gx, -kp_ * np.maximum(gx, 0) + km_ * np.maximum(-gx, 0), color=col, lw=1.0)
B.axhline(0, color="k", lw=0.4); B.legend(frameon=False, fontsize=8); B.set_ylim(-1.0, 0.8)
B.set_title("(b) Faster arbitrage when banks are open", fontsize=9.5)
# C
C = ax[1, 0]; C.axvspan(-1.5, 1.5, color=C3, alpha=0.25)
zl = (a.z_large - a.z_large.groupby(a.month).transform("mean")).values
zd = (a.z_dem - a.z_dem.groupby(a.month).transform("mean")).values
for v, mask, col, mk, lab in [(zl, op, C1, "o", "large trades, banks open"), (zl, cl, C2, "s", "large trades, banks closed"),
                              (zd, allm, C3, "^", "small and medium trades")]:
    x_, m_, s_ = binned(v, mask)
    C.errorbar(x_, m_, yerr=1.96 * s_, fmt=mk + "-", ms=3.2, color=col, ecolor=col, alpha=0.85, lw=0.8, label=lab)
C.axhline(0, color="k", lw=0.4); C.legend(frameon=False, fontsize=7.5, loc="lower left")
C.set_title("(c) Large trades lean against the premium", fontsize=9.5)
C.set_ylabel("signed order flow next 5 min (s.d.)"); C.set_xlabel("signed distance outside the band (bp)")
# D
D = ax[1, 1]; D.axvspan(-1.5, 1.5, color=C3, alpha=0.25)
for v, col, mk, lab in [(a.dpl.values, C1, "o", "USDT/TRY price"), (a.dpg.values, C2, "s", "interbank USD/TRY")]:
    vv = v - pd.Series(v).groupby(a.month.values).transform("mean").values
    x_, m_, s_ = binned(vv, allm)
    D.errorbar(x_, m_, yerr=1.96 * s_, fmt=mk + "-", ms=3.2, color=col, ecolor=col, lw=0.8, label=lab)
D.axhline(0, color="k", lw=0.4); D.legend(frameon=False, fontsize=8)
D.set_title("(d) The stablecoin price does the adjusting", fontsize=9.5)
D.set_ylabel("expected 5-min log change (bp)"); D.set_xlabel("signed distance outside the band (bp)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "hero4.pdf")); plt.close(fig)
json.dump(OUT, open(os.path.join(TAB, "emp_mech.json"), "w"), indent=1, default=float)
print("cycle", pd.DataFrame(cyc).round(4).to_string(), hol_pt)

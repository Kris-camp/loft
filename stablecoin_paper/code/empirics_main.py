"""Main empirical analysis: the USDT/TRY dollar premium (Section 6 of the paper).

Premium b_t = log(USDT/TRY minute VWAP, Binance) - log(USD/TRY interbank mid, Dukascopy), bp.
Step 1: month-by-month profiled threshold estimation of the band [l_m, u_m].
Step 2: pooled regressions with month-specific bands, month fixed effects and
        day-clustered standard errors (thresholds are super-consistent).
Env: FREQ (default 5min), BOOT (bootstrap draws per month, default 99), BAND (M or Q).
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from model import _ssr_grid
from build_panel import load_fx
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, TAB, FIG = (os.path.join(HERE, "..", d) for d in ("data_raw", "tables", "figures"))
FREQ = os.environ.get("FREQ", "5min")
BOOT = int(os.environ.get("BOOT", 99))
BAND = os.environ.get("BAND", "M")
TAG = os.environ.get("TAG", "")
PRICE = os.environ.get("PRICE", "vwap")
EXCL = os.environ.get("EXCL", "")
START, END = os.environ.get("START", "2024-01-01"), os.environ.get("END", "2026-08-31")
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"
NUM = dict(freq=FREQ, band=BAND)

# ------------------------------------------------------------------ data
p = pd.read_parquet(os.path.join(RAW, "panel_TRY.parquet")).drop(columns=["fx_mid", "fx_spread", "fxv"], errors="ignore")
p = p.loc[START:END]
fx = load_fx("USDTRY")
p = p.join(fx.reindex(p.index, method="ffill", limit=2))
if PRICE == "close":
    import glob
    rc = pd.concat([pd.read_csv(f, header=None, usecols=[0, 4]) for f in
                    sorted(glob.glob(os.path.join(RAW, "binance", "USDTTRY-1m-202[456]-*.csv")))])
    t = rc[0].astype("int64"); t = np.where(t > 1e14, t // 1000, t)
    cl = pd.Series(rc[4].values, index=pd.to_datetime(t, unit="ms", utc=True))
    p["p_usdt"] = cl[~cl.index.duplicated()].reindex(p.index)
p["b"] = 1e4 * np.log(p.p_usdt / p.fx_mid)
p["b_tri"] = 1e4 * np.log(p.p_usdt / p.p_tri)
hr = p.index.hour + p.index.minute / 60
p.loc[(hr >= 20.5) & (hr < 22.5), "b"] = np.nan      # FX rollover: stale/wide interbank quotes
p.loc[p.index.dayofweek >= 5, "b"] = np.nan           # FX market closed
p.loc[p.b.abs() > 1000, "b"] = np.nan                 # obviously bad prints
if EXCL:
    a0, a1 = EXCL.split(",")
    p.loc[a0:a1, "b"] = np.nan

g = p.resample(FREQ)
a = pd.DataFrame({"b": g.b.mean(), "nb": g.b.count(), "ofi": g.ofi.sum(), "vol": g.vol.sum(), "fxv": g.fxv.sum(),
                  "pl": np.log(g.p_usdt.mean()), "pg": np.log(g.fx_mid.mean())})
a = a[a.nb >= (1 if FREQ == "1min" else 2)]
naive = a.index.tz_localize(None)
a["month"] = naive.to_period("M").astype(str)
a["bandper"] = naive.to_period(BAND).astype(str)
a["q"] = a.ofi / a.ofi.groupby(a.month).transform("std")
a["dpl"] = 1e4 * a.pl.diff(); a["dpg"] = 1e4 * a.pg.diff()
consec = a.index.to_series().diff() == pd.Timedelta(FREQ)
a["y"] = a.b.diff(); a["x"] = a.b.shift(1)
a = a[consec & a.y.notna() & a.x.notna() & a.q.notna()].copy()
a["day"] = a.index.date
ih = (a.index.hour + 3) % 24
a["bh"] = ((ih >= 9) & (ih < 17)).astype(float)       # Istanbul business hours 09-17
NUM.update(n_obs=int(len(a)), n_days=int(a.day.nunique()), start=str(a.index.min().date()),
           end=str(a.index.max().date()))


# ------------------------------------------------------------------ step 1: band per period
def profile(y, x, q, step=1.0):
    # trimming (Hansen 2000): at least TRIM of the observations above u and below l
    TRIM = float(os.environ.get("TRIM", 0.05))
    lo_, hi_ = np.percentile(x, [100 * TRIM, 100 * (1 - TRIM)])
    grid = np.arange(np.floor(lo_), np.ceil(hi_) + step, step)
    S = _ssr_grid(y, x, q, grid, -grid)     # S[i, j]: u = grid[i], l = grid[j]; inf where l > u
    i, j = np.unravel_index(np.argmin(S), S.shape)
    return grid[i], grid[j], S.min(), grid


def lin_ssr(y, x, q):
    X = np.column_stack([np.ones(len(y)), q, -x])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return np.sum((y - X @ c) ** 2), X, c


rng = np.random.default_rng(0)
bands, lr_obs, lr_boot = {}, [], []
for per, d in a.groupby("bandper"):
    if len(d) < 500: continue
    y, x, q = d.y.values, d.x.values, d.q.values
    u, l, smin, grid = profile(y, x, q)
    s0, Xl, cl = lin_ssr(y, x, q)
    lr = len(y) * np.log(s0 / smin)
    # wild bootstrap under the linear null, Rademacher weights clustered by day
    codes = pd.factorize(d.day)[0]; e0 = y - Xl @ cl; gb = grid[::2]
    stats = []
    for _ in range(BOOT):
        w = rng.choice([-1.0, 1.0], codes.max() + 1)[codes]
        yb = Xl @ cl + e0 * w
        Sb = _ssr_grid(yb, x, q, gb, -gb)
        sb0 = np.sum((yb - Xl @ np.linalg.lstsq(Xl, yb, rcond=None)[0]) ** 2)
        stats.append(len(y) * np.log(sb0 / Sb.min()))
    stats = np.array(stats)
    bands[per] = dict(l=float(l), u=float(u), n=len(y), supLR=float(lr),
                      p=float((1 + np.sum(stats >= lr)) / (1 + len(stats))) if BOOT else np.nan,
                      mean_b=float(d.x.mean()), sd_b=float(d.x.std()))
    lr_obs.append(lr); lr_boot.append(stats)
NUM["bands"] = bands
pv = np.array([v["p"] for v in bands.values()])
NUM["share_reject_1pct"] = float(np.mean(pv <= 0.01)) if BOOT else None
NUM["share_reject_5pct"] = float(np.mean(pv <= 0.05)) if BOOT else None
if BOOT:
    tot = np.sum(lr_obs); totb = np.sum(np.vstack(lr_boot), axis=0)
    NUM["sum_supLR"] = float(tot); NUM["sum_supLR_p"] = float((1 + np.sum(totb >= tot)) / (1 + len(totb)))

a["l"] = a.bandper.map(lambda k: bands.get(k, {}).get("l", np.nan))
a["u"] = a.bandper.map(lambda k: bands.get(k, {}).get("u", np.nan))
a = a[a.l.notna()].copy()
a["up"] = -np.maximum(a.x - a.u, 0)       # coefficient = kappa^+
a["dn"] = np.maximum(a.l - a.x, 0)        # coefficient = kappa^-
a["exc"] = np.maximum(a.x - a.u, 0) + np.maximum(a.l - a.x, 0)
a["inband"] = (a.exc == 0).astype(float)
NUM["share_inband"] = float(a.inband.mean())
if not TAG:
    a.to_parquet(os.path.join(RAW, "a5.parquet"))
NUM["band_width_median"] = float(np.median([v["u"] - v["l"] for v in bands.values()]))


# ------------------------------------------------------------------ step 2: pooled regressions
def ols_fe(d, ycol, cols, fe="month"):
    """OLS with fixed effects (within transformation) and day-clustered SEs."""
    Y = d[ycol].values.astype(float)
    X = d[cols].values.astype(float)
    if fe:
        gm = d.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values
        X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    V = XtXi @ (G.T @ G) @ XtXi
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V *= ng / (ng - 1) * (n - 1) / (n - k)
    r2 = 1 - (e @ e) / (Y @ Y)
    return dict(zip(cols, b)), dict(zip(cols, np.sqrt(np.diag(V)))), n, r2, V


res = {}
# (1) linear ECM with month FE
a["negx"] = -a.x
res["linear"] = ols_fe(a, "y", ["q", "negx"])
# (2) threshold, common kappa
res["thr"] = ols_fe(a, "y", ["q", "up", "dn"])
# (3) threshold + linear term inside the model (nesting test: is there reversion inside the band?)
a["negx_in"] = -(a.x - (a.l + a.u) / 2) * a.inband
res["thr_in"] = ols_fe(a, "y", ["q", "up", "dn", "negx_in"])
# (4) business hours interactions
for c in ["q", "up", "dn"]:
    a[c + "_bh"] = a[c] * a.bh
res["bh"] = ols_fe(a, "y", ["q", "up", "dn", "q_bh", "up_bh", "dn_bh"])
# (5) by year
for yr in ["2024", "2025", "2026"]:
    m = (a.index.year == int(yr)).astype(float)
    for c in ["q", "up", "dn"]:
        a[f"{c}_{yr}"] = a[c] * m
res["year"] = ols_fe(a, "y", [f"{c}_{yr}" for yr in ["2024", "2025", "2026"] for c in ["q", "up", "dn"]])
# in-band beta (no arbitrage flow inside the band)
res["beta_in"] = ols_fe(a[a.inband == 1], "y", ["q"])
res["beta_out"] = ols_fe(a[a.inband == 0], "y", ["q", "up", "dn"])


def cs(r, k, nd=4):
    return f"{r[0][k]:.{nd}f}", f"({r[1][k]:.{nd}f})"


def star(r, k):
    t = abs(r[0][k] / r[1][k])
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


def cell(r, k, nd=4):
    if k not in r[0]: return "", ""
    return f"${r[0][k]:.{nd}f}{star(r, k)}$", f"$({r[1][k]:.{nd}f})$"


rows_spec = [("OFI$_t$", ["q"]), ("$-(b_{t-1}-u_m)_+$ \\;[$\\kappa^+$]", ["up"]), ("$(\\ell_m-b_{t-1})_+$ \\;[$\\kappa^-$]", ["dn"]),
             ("$-b_{t-1}$ \\;[linear $\\kappa$]", ["negx"]), ("$-(b_{t-1}-m_m)\\cdot\\mathbb 1\\{\\text{in band}\\}$", ["negx_in"]),
             ("OFI$_t\\times$BH", ["q_bh"]), ("$-(b_{t-1}-u_m)_+\\times$BH", ["up_bh"]), ("$(\\ell_m-b_{t-1})_+\\times$BH", ["dn_bh"])]
cols_order = ["linear", "thr", "thr_in", "bh"]
with open(os.path.join(TAB, f"emp_main{TAG}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{l" + "c" * len(cols_order) + "}\n\\toprule\n & " +
             " & ".join(f"({i+1})" for i in range(len(cols_order))) + " \\\\\n\\midrule\n")
    for lab, ks in rows_spec:
        k = ks[0]
        if not any(k in res[c][0] for c in cols_order): continue
        nd = 3 if k.startswith("q") else 4
        top = [cell(res[c], k, nd)[0] for c in cols_order]
        bot = [cell(res[c], k, nd)[1] for c in cols_order]
        fh.write(lab + " & " + " & ".join(top) + " \\\\\n & " + " & ".join(bot) + " \\\\\n")
    fh.write("\\midrule\nMonth fixed effects & " + " & ".join(["Yes"] * len(cols_order)) + " \\\\\n")
    fh.write("Month-specific band $[\\ell_m,u_m]$ & No & " + " & ".join(["Yes"] * (len(cols_order) - 1)) + " \\\\\n")
    fh.write("$N$ & " + " & ".join(f"{res[c][2]:,}" for c in cols_order) + " \\\\\n")
    fh.write("Within $R^2$ & " + " & ".join(f"{res[c][3]:.3f}" for c in cols_order) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")


def pack(r):
    return {"coef": {k: float(v) for k, v in r[0].items()}, "se": {k: float(v) for k, v in r[1].items()},
            "n": int(r[2]), "r2": float(r[3])}


NUM["reg"] = {k: pack(v) for k, v in res.items()}
c = res["thr"][0]
NUM["half_life_p_min"] = float(np.log(2) / c["up"] * pd.Timedelta(FREQ).total_seconds() / 60)
NUM["half_life_m_min"] = float(np.log(2) / c["dn"] * pd.Timedelta(FREQ).total_seconds() / 60)
cb = res["bh"][0]
NUM["bh_kp"], NUM["off_kp"] = float(cb["up"] + cb["up_bh"]), float(cb["up"])
NUM["bh_km"], NUM["off_km"] = float(cb["dn"] + cb["dn_bh"]), float(cb["dn"])
NUM["bh_beta"], NUM["off_beta"] = float(cb["q"] + cb["q_bh"]), float(cb["q"])
NUM["lr_impact_bh_p"] = NUM["bh_beta"] / NUM["bh_kp"]; NUM["lr_impact_off_p"] = NUM["off_beta"] / NUM["off_kp"]
NUM["lr_impact_bh_m"] = NUM["bh_beta"] / NUM["bh_km"]; NUM["lr_impact_off_m"] = NUM["off_beta"] / NUM["off_km"]

# year table
with open(os.path.join(TAB, f"emp_year{TAG}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccc}\n\\toprule\n Year & $\\beta$ & $\\kappa^+$ & $\\kappa^-$ & Median band width (bp) & SD of premium (bp) \\\\\n\\midrule\n")
    ry = res["year"]
    for yr in ["2024", "2025", "2026"]:
        bw = np.median([v["u"] - v["l"] for k, v in bands.items() if k.startswith(yr)])
        sd = a.loc[yr].x.std()
        fh.write(f"{yr} & {cell(ry, 'q_'+yr, 3)[0]} & {cell(ry, 'up_'+yr)[0]} & {cell(ry, 'dn_'+yr)[0]} & {bw:.0f} & {sd:.1f} \\\\\n")
        fh.write(f" & {cell(ry, 'q_'+yr, 3)[1]} & {cell(ry, 'up_'+yr)[1]} & {cell(ry, 'dn_'+yr)[1]} & & \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ size bins (T2)
center = (a.l + a.u) / 2
half = (a.u - a.l) / 2
sgn = np.where(a.x > a.u, 1.0, np.where(a.x < a.l, -1.0, np.sign(a.x - center)))
a["sgn"] = sgn
a["ry"] = -(a.y - res["thr"][0]["q"] * a.q) * a.sgn   # reversion toward the band, net of order flow
rel = np.where(a.inband == 1, np.abs(a.x - center) / half.replace(0, np.nan), np.nan)
bins = [("Inside band, inner half", (a.inband == 1) & (rel < 0.5)),
        ("Inside band, outer half", (a.inband == 1) & (rel >= 0.5))]
edges = [0, 5, 10, 20, 40, np.inf]
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    lab = f"Outside, excess $\\in({lo_},{hi_}]$ bp" if np.isfinite(hi_) else f"Outside, excess $>{lo_}$ bp"
    bins.append((lab, (a.exc > lo_) & (a.exc <= hi_)))
brow = []
for lab, m in bins:
    d = a[m]
    if len(d) < 100: continue
    dev = np.where(d.inband == 1, np.abs(d.x - (d.l + d.u) / 2), np.abs(d.x - (d.l + d.u) / 2))
    # proportional reversion: mean reversion toward the band divided by mean excess (outside) or
    # by mean distance to the band centre (inside); SE by day clusters
    base = np.where(d.inband == 1, dev, d.exc)
    num = d.ry.values
    days = pd.factorize(d.day)[0]
    rho = num.sum() / base.sum()
    # delta-method cluster SE for a ratio of sums
    gnum = np.bincount(days, num); gden = np.bincount(days, base)
    z = (gnum - rho * gden) / base.sum()
    se = np.sqrt(np.sum(z ** 2) * len(gnum) / (len(gnum) - 1))
    brow.append((lab, rho, se, float(np.mean(base)), len(d)))
with open(os.path.join(TAB, f"emp_bins{TAG}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n & Mean distance (bp) & Reversion per 5 min, $\\hat\\rho$ & Half-life (min) & $N$ \\\\\n\\midrule\n")
    for lab, rho, se, mb, n in brow:
        hl = "$\\infty$" if rho <= 0 else f"{np.log(2) / rho * pd.Timedelta(FREQ).total_seconds() / 60:.0f}"
        fh.write(f"{lab} & {mb:.1f} & ${rho:.4f}$ $({se:.4f})$ & {hl} & {n:,} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
NUM["bins"] = [dict(label=b[0], rho=float(b[1]), se=float(b[2]), mean=float(b[3]), n=int(b[4])) for b in brow]

# ------------------------------------------------------------------ who adjusts? (T6)
dd = a.dropna(subset=["dpl", "dpg"]).copy()
dl = ols_fe(dd, "dpl", ["q", "up", "dn"])
dg = ols_fe(dd, "dpg", ["q", "up", "dn"])
NUM["legs"] = {"local": pack(dl), "global": pack(dg)}
kl_p, kg_p = dl[0]["up"], -dg[0]["up"]
kl_m, kg_m = dl[0]["dn"], -dg[0]["dn"]
NUM["omega_p"] = float(kg_p / (kl_p + kg_p)); NUM["omega_m"] = float(kg_m / (kl_m + kg_m))
with open(os.path.join(TAB, f"emp_legs{TAG}.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcc}\n\\toprule\n & $\\Delta\\log P^L_t$ (USDT/TRY) & $\\Delta\\log P^G_t$ (USD/TRY) \\\\\n\\midrule\n")
    for lab, k in [("OFI$_t$", "q"), ("$-(b_{t-1}-u_m)_+$", "up"), ("$(\\ell_m-b_{t-1})_+$", "dn")]:
        nd = 3 if k == "q" else 4
        fh.write(f"{lab} & {cell(dl, k, nd)[0]} & {cell(dg, k, nd)[0]} \\\\\n & {cell(dl, k, nd)[1]} & {cell(dg, k, nd)[1]} \\\\\n")
    fh.write("\\midrule\nMonth fixed effects & Yes & Yes \\\\\n")
    fh.write(f"Implied global share $\\omega$ (premium side) & \\multicolumn{{2}}{{c}}{{{NUM['omega_p']:.2f}}} \\\\\n")
    fh.write(f"Implied global share $\\omega$ (discount side) & \\multicolumn{{2}}{{c}}{{{NUM['omega_m']:.2f}}} \\\\\n")
    fh.write(f"$N$ & {dl[2]:,} & {dg[2]:,} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

if TAG:   # robustness runs stop here
    json.dump(NUM, open(os.path.join(TAB, f"emp_numbers{TAG}.json"), "w"), indent=1, default=float)
    raise SystemExit

# ------------------------------------------------------------------ summary statistics
ww = p.dropna(subset=["b"])
summ = [("Premium $b_t$ (bp), 1-min", ww.b),
        ("Premium (bp), 5-min average", a.x),
        ("USDT/TRY volume (thousand USDT per min)", ww.vol / 1e3),
        ("Signed order flow (thousand USDT per min)", ww.ofi / 1e3),
        ("Trades per minute", ww.ntrades),
        ("Interbank USD/TRY bid-ask spread (bp)", 1e4 * ww.fx_spread),
        ("Triangular basis USDT/TRY vs.\\ BTC cross (bp)", p.b_tri)]
with open(os.path.join(TAB, "emp_summary.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & Mean & SD & P1 & P50 & P99 & $N$ \\\\\n\\midrule\n")
    for k, v in summ:
        v = v.dropna()
        q1, q50, q99 = np.percentile(v, [1, 50, 99])
        fh.write(f"{k} & {v.mean():.2f} & {v.std():.2f} & {q1:.2f} & {q50:.2f} & {q99:.2f} & {len(v):,} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
NUM["tri_sd"] = float(p.b_tri.std()); NUM["prem_sd_1m"] = float(ww.b.std())

# ------------------------------------------------------------------ figures
B = pd.DataFrame(bands).T
B.index = pd.PeriodIndex(B.index, freq=BAND).to_timestamp()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
daily = p.b.resample("D").mean().dropna()
ax[0].plot(daily.index.tz_localize(None), daily.values, color=C3, lw=0.6, label="daily mean premium")
ax[0].step(B.index, B.l, where="post", color=C2, lw=1.0, label=r"band edges $\ell_m$, $u_m$")
ax[0].step(B.index, B.u, where="post", color=C2, lw=1.0)
ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_ylabel("bp"); ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")
ax[0].set_ylim(-120, 330)
ax[0].set_title("(a) Premium and estimated band", fontsize=10)
ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax[0].tick_params(axis="x", rotation=30)
ax[1].plot(B.index, B.sd_b, color=C1, marker="o", ms=3, label="SD of premium")
ax[1].plot(B.index, B.u - B.l, color=C2, marker="s", ms=3, label="band width $u_m-\\ell_m$")
ax[1].set_ylabel("bp"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) Dispersion and band width", fontsize=10)
ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax[1].tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_premium.pdf")); plt.close(fig)

# drift vs position relative to the band
fig, ax = plt.subplots(figsize=(4.8, 3.0))
pos = np.where(a.x > a.u, a.x - a.u, np.where(a.x < a.l, a.x - a.l, 0.0))
# net of order flow and of the month fixed effects estimated in column (2)
th = res["thr"][0]
fitted_slopes = th["q"] * a.q + th["up"] * a.up + th["dn"] * a.dn
alpha_m = (a.y - fitted_slopes).groupby(a.month).transform("mean")
yy = (a.y - th["q"] * a.q - alpha_m).values
out_bins = np.r_[-60, -40, -25, -15, -8, -3, -1e-9, 1e-9, 3, 8, 15, 25, 40, 60]
xs_, ms_, ss_ = [], [], []
for lo_, hi_ in zip(out_bins[:-1], out_bins[1:]):
    m = (pos > lo_) & (pos <= hi_) if lo_ != -1e-9 else (pos == 0)
    if m.sum() < 200: continue
    xs_.append(np.mean(pos[m])); ms_.append(yy[m].mean())
    gsum = pd.Series(yy[m]).groupby(a.day.values[m]).sum(); ss_.append(np.sqrt(np.sum((gsum - gsum.mean() * 0) ** 2)) / m.sum())
xs_, ms_, ss_ = map(np.array, (xs_, ms_, ss_))
ax.errorbar(xs_, ms_, yerr=1.96 * ss_, fmt="o", ms=3.5, color=C1, ecolor=C3, lw=0.8, label="binned data")
gx = np.linspace(xs_.min(), xs_.max(), 300)
ax.plot(gx, -res["thr"][0]["up"] * np.maximum(gx, 0) + res["thr"][0]["dn"] * np.maximum(-gx, 0), color=C2, label="threshold model")
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color=C3, lw=0.4, ls=":")
ax.set_xlabel("signed distance outside the band (bp); 0 = inside")
ax.set_ylabel(r"$E[\Delta b_t-\hat\beta\,OFI_t]$ (bp per 5 min)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_drift.pdf")); plt.close(fig)

# event study 19 March 2025
ev = p.loc["2025-03-19 05:00":"2025-03-19 15:00"]
if len(ev) > 100 and ev.fx_mid.notna().sum() > 100:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bl = np.log(ev.p_usdt.dropna().iloc[0]); bg = np.log(ev.fx_mid.dropna().iloc[0])
    tt = ev.index.tz_localize(None)
    ax[0].plot(tt, 100 * (np.log(ev.p_usdt) - bl), color=C1, lw=0.8, label="USDT/TRY (Binance)")
    ax[0].plot(tt, 100 * (np.log(ev.fx_mid) - bg), color=C2, lw=0.8, label="USD/TRY (interbank)")
    ax[0].set_ylabel("% change since 05:00 UTC"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("(a) Prices, 19 March 2025", fontsize=10)
    ax[1].plot(tt, ev.b, color=C1, lw=0.7)
    bm = bands.get("2025-03" if BAND == "M" else "2025Q1")
    ax[1].axhspan(bm["l"], bm["u"], color=C3, alpha=0.3, label="March 2025 band")
    ax[1].set_ylabel("premium (bp)"); ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].set_title("(b) Premium", fontsize=10)
    for x_ in ax:
        x_.xaxis.set_major_locator(mdates.HourLocator(interval=2)); x_.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); x_.tick_params(axis="x", rotation=30); x_.set_xlabel("UTC")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_event.pdf")); plt.close(fig)
    bb = ev.b.dropna()
    after = bb.loc[bb.idxmin():]; back = after[after >= bm["l"]]
    NUM["event"] = dict(min_prem=float(bb.min()), t_min=str(bb.idxmin()),
                        fx_peak=float(100 * (np.log(ev.fx_mid).max() - bg)), usdt_peak=float(100 * (np.log(ev.p_usdt).max() - bl)),
                        fx_end=float(100 * (np.log(ev.fx_mid.dropna().iloc[-1]) - bg)), usdt_end=float(100 * (np.log(ev.p_usdt.dropna().iloc[-1]) - bl)),
                        minutes_to_band=float((back.index[0] - bb.idxmin()).total_seconds() / 60) if len(back) else None,
                        band=bm)

# intraday profile
w = p[p.index.dayofweek < 5].dropna(subset=["b"])
ihh = (w.index.hour + 3) % 24
med = w.b.groupby(w.index.date).transform("median")
prof = pd.DataFrame({"absdev": (w.b - med).abs().groupby(ihh).mean(), "vol": (w.vol / 1e3).groupby(ihh).mean()})
fig, ax = plt.subplots(figsize=(4.8, 2.9))
ax.bar(prof.index, prof.absdev, color=C1, alpha=0.75, label="mean |premium - daily median| (bp)")
ax.axvspan(8.5, 16.5, color=C2, alpha=0.08)
ax2 = ax.twinx(); ax2.plot(prof.index, prof.vol, color=C2, marker="o", ms=3, label="USDT volume (k per min)")
ax2.spines["top"].set_visible(False)
ax.set_xlabel("Istanbul hour"); ax.set_ylabel("bp"); ax2.set_ylabel("thousand USDT per minute")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "emp_intraday.pdf")); plt.close(fig)
NUM["intraday"] = prof.round(3).to_dict()

json.dump(NUM, open(os.path.join(TAB, "emp_numbers.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in NUM.items() if k not in ("bands", "intraday")}, indent=1, default=float))

# ------------------------------------------------------------------ hero figure (introduction)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
edges_h = np.r_[-60, -40, -25, -15, -8, -3, -1e-9, 1e-9, 3, 8, 15, 25, 40, 60]


def binned(mask):
    xs, ms, ss = [], [], []
    for lo_, hi_ in zip(edges_h[:-1], edges_h[1:]):
        m = mask & ((pos > lo_) & (pos <= hi_) if lo_ != -1e-9 else (pos == 0))
        if m.sum() < 200: continue
        xs.append(np.mean(pos[m])); ms.append(yy[m].mean())
        gsum = pd.Series(yy[m]).groupby(a.day.values[m]).sum(); ss.append(np.sqrt(np.sum(gsum ** 2)) / m.sum())
    return map(np.array, (xs, ms, ss))


allm = np.ones(len(a), bool)
xs_, ms_, ss_ = binned(allm)
ax[0].axvspan(-1.5, 1.5, color=C3, alpha=0.25)
ax[0].errorbar(xs_, ms_, yerr=1.96 * ss_, fmt="o", ms=4, color=C1, ecolor=C3, lw=0.8)
gx = np.linspace(-50, 50, 300)
ax[0].plot(gx, -th["up"] * np.maximum(gx, 0) + th["dn"] * np.maximum(-gx, 0), color=C2, lw=1.2)
ax[0].axhline(0, color="k", lw=0.4)
ax[0].text(0, 0.47, "inside\nband", ha="center", fontsize=8, color=C3)
ax[0].set_title("(a) No arbitrage inside the band,\nreversion outside it", fontsize=9.5)
ax[0].set_xlabel("signed distance outside the band (bp)")
ax[0].set_ylabel("expected 5-min change in premium (bp)")
bhm = a.bh.values == 1
cbh = res["bh"][0]
for mask, col, mk, lab, kp, km in [(bhm, C1, "o", "banks open (09-17)", cbh["up"] + cbh["up_bh"], cbh["dn"] + cbh["dn_bh"]),
                                   (~bhm, C2, "s", "banks closed", cbh["up"], cbh["dn"])]:
    xs_, ms_, ss_ = binned(mask)
    ax[1].errorbar(xs_, ms_, yerr=1.96 * ss_, fmt=mk, ms=3.5, color=col, ecolor=col, alpha=0.8, lw=0.6, label=lab)
    ax[1].plot(gx, -kp * np.maximum(gx, 0) + km * np.maximum(-gx, 0), color=col, lw=1.0)
ax[1].axvspan(-1.5, 1.5, color=C3, alpha=0.25)
ax[1].axhline(0, color="k", lw=0.4)
ax[1].legend(frameon=False, fontsize=8, loc="upper right")
ax[1].set_title("(b) Arbitrage is ~4x faster when\nbanks are open (above the band)", fontsize=9.5)
ax[1].set_xlabel("signed distance outside the band (bp)"); ax[1].set_ylim(-0.8, 0.7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "hero.pdf")); plt.close(fig)

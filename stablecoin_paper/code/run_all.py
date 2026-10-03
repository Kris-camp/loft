"""Produce every figure and table in the paper. Run: python run_all.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import arb_policy, simulate, stationary_density, fit_threshold, fit_linear

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "tables")
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
C1, C2, C3 = "#1f4e79", "#c0504d", "#7f7f7f"

# Baseline (illustrative) calibration: bp and minutes
P = dict(beta=1.0, lam=1.0, Kp=0.05, Km=0.05, cp=30.0, cm=30.0, sigma=4.0, rho_q=0.9, sd_q=1.0)

# ---------------------------------------------------------------- Figure 1
b = np.linspace(-90, 90, 721)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(b, arb_policy(b, 0.05, 0.05, 30, 30), color=C1, label=r"$K^\pm=0.05$")
ax[0].plot(b, arb_policy(b, 0.02, 0.02, 30, 30), color=C2, ls="--", label=r"$K^\pm=0.02$")
ax[0].axvspan(-30, 30, color=C3, alpha=0.12)
ax[0].set_xlabel("premium $b$ (bp)"); ax[0].set_ylabel(r"arbitrage flow $a^*(b)$")
ax[0].set_title("(a) Optimal arbitrage", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
bb = np.linspace(1, 90, 400)
for K, col, ls in [(0.05, C1, "-"), (0.02, C2, "--")]:
    ax[1].plot(bb, K * np.maximum(bb - 30, 0) / bb, color=col, ls=ls, label=rf"$\lambda K={K}$")
    ax[1].axhline(K, color=col, lw=0.6, ls=":")
ax[1].plot(bb, 0.035 * np.ones_like(bb), color=C3, lw=1, label="linear ECM")
ax[1].set_xlabel(r"$|b|$ (bp)"); ax[1].set_ylabel(r"$|\mu(b)|/|b|$ per minute")
ax[1].set_title("(b) Proportional reversion speed", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_policy.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# Stationary density: theory vs simulation (noise-only drift, asymmetric band)
cp, cm, ap, am, sig = 25.0, 45.0, 0.08, 0.03, 4.0
bsim, _ = simulate(2_000_000, beta=0.0, Kp=ap, Km=am, cp=cp, cm=cm, sigma=sig, seed=1)
x = np.linspace(-110, 80, 800)
dens, pin = stationary_density(x, ap, am, cp, cm, sig)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.hist(bsim[100_000:], bins=200, range=(-110, 80), density=True, color=C3, alpha=0.45,
        label="simulated")
ax.plot(x, dens, color=C1, lw=1.6, label="Proposition 4")
ax.axvline(cp, color=C2, lw=0.7, ls="--"); ax.axvline(-cm, color=C2, lw=0.7, ls="--")
ax.set_xlabel("premium $b$ (bp)"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_density.pdf")); plt.close(fig)
pin_sim = np.mean((bsim[100_000:] > -cm) & (bsim[100_000:] < cp))

# ---------------------------------------------------------------- Figure 3
# Capacity depletion: steady-state premium vs order flow, and impulse responses
beta, lam, Kbar, xi, chi, c = 1.0, 1.0, 0.05, 0.01, 0.002, 30.0
qcrit = xi * lam * Kbar / (chi * beta)
qs = np.linspace(0, 0.98 * qcrit, 300)
bstar_dep = c + beta * qs / (lam * (Kbar - chi * beta * qs / (xi * lam)))
bstar_fix = c + beta * qs / (lam * Kbar)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(qs, bstar_fix, color=C3, label="fixed capacity")
ax[0].plot(qs, bstar_dep, color=C1, label="endogenous capacity")
ax[0].axvline(qcrit, color=C2, ls="--", lw=0.8)
ax[0].text(qcrit * 0.96, 270, r"$q_{crit}$", ha="right", color=C2, fontsize=8)
ax[0].set_ylim(25, 300)
ax[0].set_xlabel("persistent order flow $q$"); ax[0].set_ylabel(r"steady-state premium $b^*$ (bp)")
ax[0].set_title("(a) Convex long-run price impact", fontsize=10); ax[0].legend(frameon=False, fontsize=8)


def ode(q_fun, T, deplete, dt=0.1, b0=0.0, K0=Kbar):
    n = int(T / dt); bt = np.empty(n); Kt = np.empty(n); bcur, K = b0, K0
    for i in range(n):
        t = i * dt
        a = K * max(bcur - c, 0.0)
        bcur += (beta * q_fun(t) - lam * a) * dt
        if deplete:
            K += (xi * (Kbar - K) - chi * a) * dt
        bt[i], Kt[i] = bcur, K
    return np.arange(n) * dt, bt, Kt


burst = lambda t: 4.0 if t < 60 else 0.0
for dep, col, lab in [(False, C3, "fixed capacity"), (True, C1, "endogenous capacity")]:
    tt, bt, Kt = ode(burst, 400, dep)
    ax[1].plot(tt, bt, color=col, label=lab)
ax[1].axhline(c, color=C2, lw=0.7, ls="--")
ax[1].set_xlabel("minutes"); ax[1].set_ylabel("premium (bp)")
ax[1].set_title("(b) Response to a 60-minute flow burst", fontsize=10)
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_capacity.pdf")); plt.close(fig)

# half-life of excess premium after the burst ends
def post_half_life(dep):
    tt, bt, _ = ode(burst, 2000, dep)
    i0 = np.searchsorted(tt, 60.0); ex0 = bt[i0] - c
    j = i0 + np.argmax(bt[i0:] - c <= ex0 / 2)
    return tt[j] - 60.0, bt[i0]
hl_fix, peak_fix = post_half_life(False)
hl_dep, peak_dep = post_half_life(True)

# ---------------------------------------------------------------- Figure 4
# Local information transmission (Proposition 6)
d = np.linspace(0, 150, 600)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
omega = 0.5
for c_, K_, col, ls in [(30, 0.05, C1, "-"), (60, 0.05, C2, "--"), (30, 0.01, C3, "-.")]:
    share = omega * np.maximum(1 - c_ / np.maximum(d, 1e-9), 0) * (1 - np.exp(-K_ * 30))
    ax.plot(d, share, color=col, ls=ls, label=rf"$c={c_}$, $\lambda K={K_}$")
ax.set_xlabel(r"local information shock $\delta$ (bp)")
ax.set_ylabel(r"share in global price, $h=30$ min")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_discovery.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Table 1: Monte Carlo
designs = [
    ("Symmetric", dict(P)),
    ("Asymmetric", dict(P, cp=20.0, cm=50.0, Kp=0.08, Km=0.02)),
    ("Low capacity", dict(P, Kp=0.015, Km=0.015)),
]
R, T = 200, 20_000
grid = np.arange(0.0, 80.5, 2.5)
rows = []
for name, d_ in designs:
    est = []
    for r in range(R):
        bs, qs_ = simulate(T, seed=1000 + r, **d_)
        f = fit_threshold(bs, qs_, grid, grid)
        lin = fit_linear(bs, qs_)
        est.append([f["cp"], f["cm"], f["kp"], f["km"], f["beta"], lin["k"], lin["beta"],
                    f["beta"] / max(f["kp"], 1e-6)])
    est = np.array(est)
    truth = [d_["cp"], d_["cm"], d_["lam"] * d_["Kp"], d_["lam"] * d_["Km"], d_["beta"]]
    rows.append((name, d_, truth, est))

with open(os.path.join(TAB, "tab1_montecarlo.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccc}\n\\toprule\n")
    fh.write(" & $c^+$ & $c^-$ & $\\kappa^+=\\lambda K^+$ & $\\kappa^-=\\lambda K^-$ & $\\beta$ \\\\\n\\midrule\n")
    for name, d_, truth, est in rows:
        fh.write(f"\\multicolumn{{6}}{{l}}{{\\textit{{{name} design}}}} \\\\\n")
        fmt = lambda vals: " & ".join(f"{v:.1f}" if k < 2 else f"{v:.3f}" for k, v in enumerate(vals))
        fh.write("True & " + fmt(truth) + " \\\\\n")
        med = np.median(est[:, :5], axis=0)
        fh.write("Median estimate & " + fmt(med) + " \\\\\n")
        rmse = np.sqrt(np.mean((est[:, :5] - np.array(truth)) ** 2, axis=0))
        fh.write("RMSE & " + fmt(rmse) + " \\\\\n")
        fh.write("Linear ECM $\\hat\\kappa$ & \\multicolumn{4}{c}{" + f"{np.median(est[:,5]):.4f}" + "} & " + f"{np.median(est[:,6]):.3f}" + " \\\\\n")
        fh.write("\\addlinespace\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Table 2: size-binned reversion
bs, qs_ = simulate(500_000, seed=7, **P)
y = np.diff(bs); x = bs[:-1]
bins = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90)]
with open(os.path.join(TAB, "tab2_bins.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n$|b_t|$ (bp) & " +
             " & ".join(f"[{lo},{hi})" for lo, hi in bins) + " \\\\\n\\midrule\n")
    sp, sh, nn = [], [], []
    for lo, hi in bins:
        m = (np.abs(x) >= lo) & (np.abs(x) < hi)
        # regress -db*sign(b) on |b| through the origin, controlling for q
        X = np.column_stack([np.abs(x[m]), qs_[m] * np.sign(x[m])])
        coef, *_ = np.linalg.lstsq(X, -y[m] * np.sign(x[m]), rcond=None)
        sp.append(coef[0]); nn.append(m.mean())
        hl = np.log(2) / coef[0] if coef[0] > 1e-4 else np.inf
        sh.append(hl)
    fh.write("Reversion speed per minute & " + " & ".join(f"{v:.4f}" for v in sp) + " \\\\\n")
    fh.write("Implied half-life (min) & " + " & ".join("$\\infty$" if not np.isfinite(v) or v > 1e4 else f"{v:.0f}" for v in sh) + " \\\\\n")
    fh.write("Share of observations & " + " & ".join(f"{v:.3f}" for v in nn) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(TAB, "numbers.txt"), "w") as fh:
    fh.write(f"pin_theory={pin:.4f} pin_sim={pin_sim:.4f}\n")
    fh.write(f"qcrit={qcrit:.3f}\n")
    fh.write(f"hl_fix={hl_fix:.1f} hl_dep={hl_dep:.1f} peak_fix={peak_fix:.1f} peak_dep={peak_dep:.1f} theory_hl={np.log(2)/(lam*Kbar):.1f}\n")
    for name, d_, truth, est in rows:
        imp = np.median(est[:, 7]); imp_true = d_["beta"] / (d_["lam"] * d_["Kp"])
        fh.write(f"{name}: impact multiplier median={imp:.2f} true={imp_true:.2f}; linear k median={np.median(est[:,5]):.4f}\n")
print(open(os.path.join(TAB, "numbers.txt")).read())

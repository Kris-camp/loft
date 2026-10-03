"""Model figure: the three predictions that Figure 1 tests, drawn from the model alone."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
NAVY, RED, GREY, INK, MUTE = "#1f4e79", "#c0504d", "#8c8c8c", "#222222", "#666666"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.6, "xtick.color": "#555555",
                     "xtick.major.width": 0.5, "axes.labelcolor": INK, "axes.titlesize": 9})

b = np.linspace(-60, 60, 1201)
c_open, c_closed = 12.0, 18.0        # band half-widths (bp): wider when banks are closed
K_open, K_closed = 1.0, 0.25         # capacity (model units)
lam, omega = 1.0, 0.04               # arbitrage price impact; global leg's share


def soft(x, c):
    return np.sign(x) * np.maximum(np.abs(x) - c, 0)


def bands(ax, both=True):
    if both:
        ax.axvspan(-c_closed, c_closed, color=RED, alpha=0.06, lw=0)
    ax.axvspan(-c_open, c_open, color=NAVY, alpha=0.09, lw=0)
    ax.axhline(0, color="#c8c8c8", lw=0.6, zorder=0)
    ax.set_xlim(-60, 60); ax.set_xticks([-50, -25, 0, 25, 50]); ax.set_yticks([])


fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.75), sharex=True)
Y = 45
# (a) policy: flow toward the band = -a*(b)
A = ax[0]; bands(A)
A.plot(b, -K_open * soft(b, c_open), color=NAVY, lw=1.7, label="banks open")
A.plot(b, -K_closed * soft(b, c_closed), color=RED, lw=1.7, label="banks closed")
A.set_ylim(-Y, Y)
A.set_ylabel("arbitrage flow")
A.set_title("(a) Arbitrage policy", loc="left")
A.text(0, -0.80 * Y, "band", ha="center", color=MUTE, fontsize=8)
A.legend(frameon=False, fontsize=7.8, loc="upper right", handlelength=1.3, borderaxespad=0.1)

# (b) drift of the premium
B = ax[1]; bands(B)
s = 30
B.plot(b, -lam * K_open * soft(b, c_open) / s, color=NAVY, lw=1.7, label="banks open: fast")
B.plot(b, -lam * K_closed * soft(b, c_closed) / s, color=RED, lw=1.7, label="banks closed: slow")
B.set_ylim(-Y / s, Y / s)
B.set_ylabel("expected premium change")
B.set_title("(b) Convergence speed $\\kappa=\\lambda K$", loc="left")
B.legend(frameon=False, fontsize=7.8, loc="upper right", handlelength=1.3, borderaxespad=0.1)
B.text(0, -0.80 * Y / s, "martingale", ha="center", color=MUTE, fontsize=8)

# (c) which price adjusts
C = ax[2]; bands(C, both=False)
C.plot(b, -(1 - omega) * lam * K_open * soft(b, c_open) / s, color=NAVY, lw=1.7, label="stablecoin price")
C.plot(b, omega * lam * K_open * soft(b, c_open) / s, color=GREY, lw=1.7, ls=(0, (4, 2)), label="dollar price")
C.set_ylim(-Y / s, Y / s)
C.set_ylabel("expected price change")
C.set_title("(c) Price discovery", loc="left")
C.legend(frameon=False, fontsize=7.8, loc="upper right", handlelength=1.6, borderaxespad=0.1)

for a_ in ax:
    a_.set_xlabel("premium $b$ (bp)")
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(FIG, "fig_model.pdf")); plt.close(fig)

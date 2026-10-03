"""Threshold-arbitrage model of a local stablecoin premium.

db_t = (beta*q_t - lam*a*(b_t)) dt + sigma dW_t,
a*(b) = Kp*(b - cp)_+ - Km*(-b - cm)_+.

Units: premium in basis points, time in minutes.
"""
import numpy as np
from numba import njit, prange


def arb_policy(b, Kp, Km, cp, cm):
    return Kp * np.maximum(b - cp, 0.0) - Km * np.maximum(-b - cm, 0.0)


def simulate(T, dt=1.0, beta=1.0, lam=1.0, Kp=0.05, Km=0.05, cp=30.0, cm=30.0,
             sigma=4.0, rho_q=0.9, sd_q=1.0, b0=0.0, seed=0, K_path=None):
    """Euler scheme. q is an AR(1) signed order-flow imbalance (standardized units).

    Returns arrays b (T+1,), q (T,). Observation at t uses b[t], q[t] to predict b[t+1].
    """
    rng = np.random.default_rng(seed)
    q = np.empty(T)
    e = rng.standard_normal(T)
    q = _ar1(e, rho_q, sd_q)
    w = rng.standard_normal(T)
    Kmult = np.ones(T) if K_path is None else np.asarray(K_path, dtype=float)
    b = _euler(q, w, Kmult, dt, beta, lam, Kp, Km, cp, cm, sigma, b0)
    return b, q


@njit(cache=True)
def _euler(q, w, Kmult, dt, beta, lam, Kp, Km, cp, cm, sigma, b0):
    T = q.shape[0]
    b = np.empty(T + 1)
    b[0] = b0
    sq = np.sqrt(dt)
    for t in range(T):
        x = b[t]
        a = 0.0
        if x > cp:
            a = Kmult[t] * Kp * (x - cp)
        elif x < -cm:
            a = -Kmult[t] * Km * (-x - cm)
        b[t + 1] = x + (beta * q[t] - lam * a) * dt + sigma * sq * w[t]
    return b


def stationary_density(x, alpha_p, alpha_m, cp, cm, sigma):
    """Proposition 4: stationary density when the order-flow drift has mean zero
    and is absorbed into the diffusion term. alpha = lam*K."""
    sp = np.sqrt(np.pi * sigma ** 2 / alpha_p) / 2
    sm = np.sqrt(np.pi * sigma ** 2 / alpha_m) / 2
    Z = cp + cm + sp + sm
    f = np.where(x > cp, np.exp(-alpha_p / sigma ** 2 * (x - cp) ** 2),
                 np.where(x < -cm, np.exp(-alpha_m / sigma ** 2 * (x + cm) ** 2), 1.0))
    return f / Z, (cp + cm) / Z


def fit_threshold(b, q, grid_p, grid_m):
    """Profiled least squares for
    db_{t+1} = a0 + beta*q_t - kp*(b_t-cp)_+ + km*(-b_t-cm)_+ + e.
    Returns dict of estimates and SSR surface."""
    y = np.diff(b)
    x = b[:-1]
    n = len(y)
    grid_p = np.asarray(grid_p, float); grid_m = np.asarray(grid_m, float)
    S = _ssr_grid(y, x, np.asarray(q, float), grid_p, grid_m)
    i, j = np.unravel_index(np.argmin(S), S.shape)
    cp, cm = grid_p[i], grid_m[j]
    X = np.column_stack([np.ones(n), q, -np.maximum(x - cp, 0.0), np.maximum(-x - cm, 0.0)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    ssr = r @ r
    s2 = ssr / (n - X.shape[1])
    # HAC-free conventional SEs conditional on thresholds (Hansen 2000: thresholds
    # are super-consistent, so slope inference is asymptotically unaffected).
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return dict(cp=cp, cm=cm, a0=coef[0], beta=coef[1], kp=coef[2], km=coef[3],
                se_beta=se[1], se_kp=se[2], se_km=se[3], ssr=ssr, n=n)


def fit_linear(b, q):
    """Linear error-correction benchmark: db_{t+1} = a0 + beta*q_t - k*b_t + e."""
    y = np.diff(b)
    X = np.column_stack([np.ones(len(y)), q, -b[:-1]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return dict(a0=coef[0], beta=coef[1], k=coef[2], ssr=r @ r)


@njit(cache=True)
def _ar1(e, rho, sd):
    q = np.empty(e.shape[0])
    q[0] = sd * e[0]
    s = sd * np.sqrt(1 - rho ** 2)
    for t in range(1, e.shape[0]):
        q[t] = rho * q[t - 1] + s * e[t]
    return q


@njit(cache=True, parallel=True)
def _ssr_grid(y, x, q, grid_p, grid_m):
    n = y.shape[0]
    out = np.empty((grid_p.shape[0], grid_m.shape[0]))
    for i in prange(grid_p.shape[0]):
        for j in range(grid_m.shape[0]):
            if -grid_m[j] > grid_p[i]:
                out[i, j] = np.inf
                continue
            A = np.zeros((4, 4)); v = np.zeros(4); yy = 0.0
            r = np.empty(4)
            for t in range(n):
                r[0] = 1.0; r[1] = q[t]
                r[2] = -max(x[t] - grid_p[i], 0.0); r[3] = max(-x[t] - grid_m[j], 0.0)
                for k in range(4):
                    v[k] += r[k] * y[t]
                    for l in range(4):
                        A[k, l] += r[k] * r[l]
                yy += y[t] * y[t]
            A += 1e-10 * np.eye(4)
            coef = np.linalg.solve(A, v)
            out[i, j] = yy - coef @ v
    return out

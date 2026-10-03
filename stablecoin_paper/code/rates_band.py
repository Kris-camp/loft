"""Band width and the lira-dollar interest differential (Remark on settlement latency).

Policy-rate paths (2024-2025), from central-bank announcements:
CBRT one-week repo: 42.5% (from 21 Dec 2023), 45% (25 Jan 2024), 50% (21 Mar 2024),
47.5% (26 Dec 2024), 45% (23 Jan 2025), 42.5% (6 Mar 2025), 46% (17 Apr 2025),
43% (24 Jul 2025), 40.5% (11 Sep 2025), 39.5% (23 Oct 2025), 38% (11 Dec 2025).
Fed funds target midpoint: 5.375% to 18 Sep 2024, 4.875%, 4.625% (7 Nov 2024),
4.375% (18 Dec 2024), 4.125% (17 Sep 2025), 3.875% (29 Oct 2025), 3.625% (10 Dec 2025).
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "..", "tables")
cb = [("2023-12-21", 42.5), ("2024-01-25", 45), ("2024-03-21", 50), ("2024-12-26", 47.5), ("2025-01-23", 45),
      ("2025-03-06", 42.5), ("2025-04-17", 46), ("2025-07-24", 43), ("2025-09-11", 40.5), ("2025-10-23", 39.5),
      ("2025-12-11", 38)]
fed = [("2023-07-27", 5.375), ("2024-09-19", 4.875), ("2024-11-08", 4.625), ("2024-12-19", 4.375),
       ("2025-09-18", 4.125), ("2025-10-30", 3.875), ("2025-12-11", 3.625)]


def path(ev):
    s = pd.Series({pd.Timestamp(d): v for d, v in ev})
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    return s.reindex(s.index.union(days)).ffill().reindex(days)


diff = (path(cb) - path(fed)).resample("ME").mean()
diff.index = diff.index.to_period("M").astype(str)
num = json.load(open(os.path.join(TAB, "emp_numbers.json")))
B = pd.DataFrame(num["bands"]).T
B = B.loc[[m for m in B.index if m <= "2025-12"]]
B["w"] = B.u - B.l
B["diff"] = diff.reindex(B.index).values / 100.0          # in decimal per year
X = np.column_stack([np.ones(len(B)), B["diff"].values])
y = B.w.values
b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
# Newey-West (3 lags) standard errors
n, k = X.shape; L = 3
S = sum((X[t:t + 1].T * e[t]) @ (X[t:t + 1] * e[t]) for t in range(n))
for l in range(1, L + 1):
    w = 1 - l / (L + 1)
    for t in range(l, n):
        G = (X[t:t + 1].T * e[t]) @ (X[t - l:t - l + 1] * e[t - l]); S += w * (G + G.T)
XtXi = np.linalg.inv(X.T @ X); V = XtXi @ S @ XtXi
se = np.sqrt(np.diag(V))
gamma = b[1]                       # bp of width per unit (decimal) of annual differential
tau_years = gamma / 2 / 1e4        # width = c+ + c- + 2 (i - i*) tau  (both in decimals, width in bp)
out = dict(gamma=float(gamma), se=float(se[1]), tau_hours=float(tau_years * 365 * 24),
           tau_hours_se=float(se[1] / 2 / 1e4 * 365 * 24), n=int(n), corr=float(np.corrcoef(B["diff"], y)[0, 1]),
           diff_range=[float(B["diff"].min()), float(B["diff"].max())])
json.dump(out, open(os.path.join(TAB, "emp_rates.json"), "w"), indent=1)
print(out)
print(B[["l", "u", "w", "diff"]].round(3).to_string())

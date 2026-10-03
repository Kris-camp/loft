"""Size-split signed order flow from Binance USDT/TRY aggTrades.

For each month, downloads the aggTrades archive, classifies aggressor side
(is_buyer_maker=True -> taker sold), and sums signed USDT quantity per 5-minute
interval by trade size: small (< 1,000 USDT), medium (1,000-10,000), large (>= 10,000).
Writes data_raw/ofi_size/YYYY-MM.parquet and deletes the archive.
"""
import os, io, sys, zipfile, urllib.request, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data_raw", "ofi_size")
os.makedirs(OUT, exist_ok=True)
EDGES = [0, 1_000, 10_000, np.inf]
LABELS = ["small", "mid", "large"]


def month_list(a, b):
    y, m = map(int, a.split("-")); y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield f"{y}-{m:02d}"; m += 1
        if m == 13: y, m = y + 1, 1


def process(ym, sym="USDTTRY"):
    fn = os.path.join(OUT, f"{ym}.parquet")
    if os.path.exists(fn): return "cached"
    url = f"https://data.binance.vision/data/spot/monthly/aggTrades/{sym}/{sym}-aggTrades-{ym}.zip"
    for k in range(5):
        try:
            raw = urllib.request.urlopen(url, timeout=300).read(); break
        except Exception as e:
            time.sleep(2 ** k)
    else:
        return "fail"
    z = zipfile.ZipFile(io.BytesIO(raw))
    with z.open(z.namelist()[0]) as f:
        d = pd.read_csv(f, header=None, usecols=[1, 2, 5, 6], names=["px", "qty", "ts", "bm"])
    if isinstance(d.bm.iloc[0], str):
        d["bm"] = d.bm.str.lower().eq("true")
    t = d.ts.astype("int64").values
    t = np.where(t > 1e14, t // 1000, t)
    d.index = pd.to_datetime(t, unit="ms", utc=True)
    sgn = np.where(d.bm.values, -1.0, 1.0)
    d["s"] = sgn * d.qty
    d["cls"] = pd.cut(d.qty, EDGES, labels=LABELS, right=False)
    g = d.groupby([pd.Grouper(freq="5min"), "cls"], observed=False)
    out = g.s.sum().unstack("cls").fillna(0.0)
    out.columns = [f"ofi_{c}" for c in out.columns]
    n = d.groupby([pd.Grouper(freq="5min"), "cls"], observed=False).s.size().unstack("cls").fillna(0)
    n.columns = [f"n_{c}" for c in n.columns]
    vol = d.groupby([pd.Grouper(freq="5min"), "cls"], observed=False).qty.sum().unstack("cls").fillna(0.0)
    vol.columns = [f"vol_{c}" for c in vol.columns]
    out = out.join(n).join(vol)
    out.to_parquet(fn)
    return f"ok {len(d):,} trades"


if __name__ == "__main__":
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2024-01", "2026-08")
    for ym in month_list(a, b):
        print(ym, process(ym), flush=True)

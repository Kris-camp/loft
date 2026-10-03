"""Download Binance public 1-minute klines (data.binance.vision) into data_raw/binance."""
import os, sys, io, zipfile, time, urllib.request
OUT = os.path.join(os.path.dirname(__file__), "..", "data_raw", "binance")
os.makedirs(OUT, exist_ok=True)

def months(a, b):
    y, m = map(int, a.split("-")); y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield f"{y}-{m:02d}"; m += 1
        if m == 13: y, m = y + 1, 1

def get(sym, ym):
    fn = os.path.join(OUT, f"{sym}-1m-{ym}.csv")
    if os.path.exists(fn): return "cached"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{ym}.zip"
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
                open(fn, "wb").write(z.read(z.namelist()[0])); return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404: return "404"
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    return "fail"

if __name__ == "__main__":
    jobs = [(s, "2023-06", "2026-08") for s in ["USDTTRY", "BTCTRY", "BTCUSDT"]] + \
           [(s, "2022-01", "2024-03") for s in ["USDTNGN", "BTCNGN"]]
    for sym, a, b in jobs:
        for ym in months(a, b):
            print(sym, ym, get(sym, ym), flush=True)

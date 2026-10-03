"""Download Dukascopy USD/TRY 1-minute bid/ask candles (one file per day, weekdays).
Note: Dukascopy URLs use 0-based months. Prices are scaled by 1e5."""
import os, time, datetime as dt, urllib.request, random
OUT = os.path.join(os.path.dirname(__file__), "..", "data_raw", "dukascopy")
os.makedirs(OUT, exist_ok=True)
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

def fetch(day, side, pair="USDTRY"):
    fn = os.path.join(OUT, f"{pair}_{side}_{day:%Y%m%d}.bi5")
    if os.path.exists(fn): return "cached"
    url = f"https://datafeed.dukascopy.com/datafeed/{pair}/{day.year}/{day.month-1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"
    for k in range(40):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            open(fn, "wb").write(data); return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404: open(fn, "wb").close(); return "404"
        except Exception:
            pass
        time.sleep(1 + 2 * random.random())
    return "fail"

if __name__ == "__main__":
    import sys
    PAIR = sys.argv[1] if len(sys.argv) > 1 else "USDTRY"
    d0, d1 = dt.date(2023, 6, 1), dt.date(2026, 8, 31)
    if len(sys.argv) > 3:
        d0, d1 = dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3])
    days = [d0 + dt.timedelta(n) for n in range((d1 - d0).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    # priority: 2025 (includes the March 2025 lira shock), then the rest newest-first
    days.sort(key=lambda d: (d.year != 2025, -d.toordinal()))
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(d, "BID") for d in days]  # FX bid-ask spread is ~1bp; mid = bid + half median spread
    with ThreadPoolExecutor(12) as ex:
        for (d, s), r in zip(jobs, ex.map(lambda j: fetch(*j, pair=PAIR), jobs)):
            print(d, s, r, flush=True)

"""Build minute-level panels from raw Binance klines and Dukascopy FX candles.

Output: data_raw/panel_TRY.parquet, data_raw/panel_NGN.parquet with columns
  p_usdt   : USDT price in local currency (minute VWAP, Binance)
  p_tri    : crypto-implied USDT price = BTC/local VWAP / BTC/USDT VWAP
  ofi      : taker-buy minus taker-sell USDT volume on USDT/local
  vol      : USDT volume on USDT/local
  fx_mid   : USD/local interbank mid (Dukascopy; TRY only), forward-filled <= 5 min
  fx_spread: relative bid-ask spread of FX benchmark
"""
import os, glob, lzma, struct, datetime as dt
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data_raw")
COLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ign"]


def load_binance(sym):
    fs = sorted(glob.glob(os.path.join(RAW, "binance", f"{sym}-1m-*.csv")))
    dfs = []
    for f in fs:
        d = pd.read_csv(f, header=None, names=COLS)
        t = d.open_time.astype("int64")
        unit = np.where(t > 1e14, "us", "ms")
        ts = np.where(unit == "us", t // 1000, t)
        d.index = pd.to_datetime(ts, unit="ms", utc=True)
        dfs.append(d)
    d = pd.concat(dfs)
    d = d[~d.index.duplicated()]
    vwap = np.where(d.v > 0, d.qv / d.v.replace(0, np.nan), np.nan)
    out = pd.DataFrame({"vwap": vwap, "close": d.c, "v": d.v, "tbv": d.tbv, "n": d.n}, index=d.index)
    return out


def load_fx(pair="USDTRY"):
    rows = []
    for side in ("BID", "ASK"):
        for f in sorted(glob.glob(os.path.join(RAW, "dukascopy", f"{pair}_{side}_*.bi5"))):
            if os.path.getsize(f) == 0: continue
            try:
                raw = lzma.decompress(open(f, "rb").read())
            except Exception:
                continue
            day = dt.datetime.strptime(os.path.basename(f).split("_")[2][:8], "%Y%m%d").replace(tzinfo=dt.timezone.utc)
            a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
            secs = a[:, 0].astype("int64")
            close = a[:, 2] / 1e5
            vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
            idx = pd.Timestamp(day) + pd.to_timedelta(secs, unit="s")
            rows.append(pd.DataFrame({"side": side, "px": close, "fxv": vol}, index=idx))
    fx = pd.concat(rows)
    bid = fx[fx.side == "BID"]; ask = fx[fx.side == "ASK"]
    bid = bid[~bid.index.duplicated()]; ask = ask[~ask.index.duplicated()]
    m = pd.DataFrame({"bid": bid.px, "fxv": bid.fxv})
    m["ask"] = ask.px.reindex(m.index)
    m = m[(m.fxv > 0)]  # Dukascopy pads non-trading minutes with zero-volume flat candles
    rel = ((m.ask - m.bid) / m.bid).dropna()
    half = 0.5 * (rel.median() if len(rel) else 1e-4)
    # where the ask is unavailable, the mid is the bid plus half the median relative spread
    m["fx_mid"] = np.where(m.ask.notna(), (m.bid + m.ask) / 2, m.bid * (1 + half))
    m["fx_spread"] = (m.ask - m.bid) / m.fx_mid
    return m[["fx_mid", "fx_spread"]]


def build(local, with_fx):
    u = load_binance(f"USDT{local}")
    b = load_binance(f"BTC{local}")
    g = load_binance("BTCUSDT")
    idx = u.index
    p = pd.DataFrame(index=idx)
    p["p_usdt"] = u.vwap
    p["ofi"] = 2 * u.tbv - u.v
    p["vol"] = u.v
    p["ntrades"] = u.n
    p["p_tri"] = (b.vwap / g.vwap).reindex(idx)
    if with_fx:
        fx = load_fx(f"USD{local}")
        fx = fx.reindex(idx, method="ffill", limit=5)
        p = p.join(fx)
    return p


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["TRY", "NGN"]
    for loc in which:
        p = build(loc, with_fx=(loc == "TRY"))
        p.to_parquet(os.path.join(RAW, f"panel_{loc}.parquet"))
        print(loc, p.index.min(), p.index.max(), len(p))
        print(p.describe().T.to_string())

# Who Discovers the Dollar? — Stern/Salomon Microstructure submission

- `paper.pdf` / `paper.tex`: the submission (build: `pdflatex paper.tex && pdflatex paper.tex`).

## Reproduce (Python 3.11; `pip install numpy scipy pandas pyarrow matplotlib numba pymupdf`)
1. `python code/download_binance.py` — Binance public 1-minute klines (USDT/TRY, BTC/TRY, BTC/USDT; also NGN pairs).
2. `python code/download_dukascopy.py` — Dukascopy interbank USD/TRY 1-minute candles (rate-limited; retries automatically).
3. `python code/build_panel.py TRY` — builds `data_raw/panel_TRY.parquet`.
4. `BOOT=99 python code/empirics_main.py` — Section 6 tables (`tables/emp_*.tex`) and figures (`figures/emp_*.pdf`, `figures/hero.pdf`).
5. Additional tests (each reads `data_raw/a5.parquet` written by step 4):
   - `python code/empirics_extra.py` — interbank-activity filter (Table 4) and MASAK before/after (Table 8)
   - `python code/empirics_state.py` — bank-state-specific bands and in-band beta (Table 5)
   - `python code/aggtrades_ofi.py 2024-01 2026-08 && python code/empirics_size.py` — size-split order flow (Table 6)
   - `python code/download_binance.py USDTMXN 2024-11 2026-08 && python code/download_dukascopy.py USDMXN 2024-11-01 2026-08-31 && python code/empirics_did.py` — DiD vs USDT/MXN (Table 9)
   - `python code/empirics_mech.py` — flow-side policy test, price-flow cross-equation test, daily cycle and holiday placebo, 24/7-banking counterfactual, Figure 1 (run after empirics_state.py)
   - `python code/download_binance.py USDTMXN 2024-11 2026-08 && python code/download_dukascopy.py USDMXN 2024-11-01 2026-08-31 && python code/empirics_mxn.py` — Turkey vs Mexico contrast (Table/Figure on rails that never close)
   - `python code/rates_band.py` — band width vs CBRT-Fed rate differential (not reported: not robust to a time trend)
6. Robustness: `TAG=_f1 FREQ=1min`, `TAG=_f15 FREQ=15min`, `TAG=_close PRICE=close`, `TAG=_excl EXCL=2025-03-19,2025-04-30`,
   `TAG=_q BAND=Q`, `TAG=_trim10 TRIM=0.10` (each with `BOOT=0 python code/empirics_main.py`), then `python code/robust_table.py`.
7. `python code/run_all.py` — theory figures and Monte Carlo (Section 7).

`code/estimate_real.py` runs the basic estimator on any CSV with columns
timestamp, local_price, benchmark_price, buy_volume, sell_volume (e.g., for NGN data from other sources).

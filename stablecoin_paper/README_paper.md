# Who Discovers the Dollar? — Stern/Salomon Microstructure submission

- `paper.pdf` / `paper.tex`: the submission (build: `pdflatex paper.tex && pdflatex paper.tex`).

## Reproduce (Python 3.11; `pip install numpy scipy pandas pyarrow matplotlib numba pymupdf`)
1. `python code/download_binance.py` — Binance public 1-minute klines (USDT/TRY, BTC/TRY, BTC/USDT; also NGN pairs).
2. `python code/download_dukascopy.py` — Dukascopy interbank USD/TRY 1-minute candles (rate-limited; retries automatically).
3. `python code/build_panel.py TRY` — builds `data_raw/panel_TRY.parquet`.
4. `BOOT=99 python code/empirics_main.py` — Section 6 tables (`tables/emp_*.tex`) and figures (`figures/emp_*.pdf`, `figures/hero.pdf`).
5. `python code/empirics_extra.py` — benchmark-quality test (Table 4) and MASAK Circular 29 test (Table 6).
6. Robustness: `TAG=_f1 FREQ=1min`, `TAG=_f15 FREQ=15min`, `TAG=_close PRICE=close`, `TAG=_excl EXCL=2025-03-19,2025-04-30`,
   `TAG=_q BAND=Q`, `TAG=_trim10 TRIM=0.10` (each with `BOOT=0 python code/empirics_main.py`), then `python code/robust_table.py`.
7. `python code/run_all.py` — theory figures and Monte Carlo (Section 7).

`code/estimate_real.py` runs the basic estimator on any CSV with columns
timestamp, local_price, benchmark_price, buy_volume, sell_volume (e.g., for NGN data from other sources).

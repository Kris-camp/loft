# Who Discovers the Dollar? — Stern/Salomon Microstructure submission

- `paper.pdf` / `paper.tex`: the submission.
- `code/run_all.py`: reproduces every figure and table (`pip install numpy scipy matplotlib numba`; ~3 min).
- `code/estimate_real.py data.csv TRY`: runs the paper's estimator on minute-level local USDT data
  (columns: timestamp, local_price, benchmark_price, buy_volume, sell_volume) and writes `tables/real_TRY.tex`.

Build: `pdflatex paper.tex && pdflatex paper.tex`

"""Collect robustness runs into tables/emp_robust.tex."""
import os, json
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
specs = [("", "Baseline: 5-min, VWAP, monthly bands, 5\\% trimming"), ("_f1", "1-minute frequency"),
         ("_f15", "15-minute frequency"), ("_close", "Last-trade price instead of VWAP"),
         ("_excl", "Excluding 19 Mar--30 Apr 2025"), ("_q", "Quarterly bands"), ("_trim10", "10\\% trimming")]


def st(c, s):
    t = abs(c / s)
    return "^{***}" if t > 2.576 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


with open(os.path.join(TAB, "emp_robust.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lccccc}\n\\toprule\n & $\\beta$ & $\\kappa^+$ & $\\kappa^-$ & In-band reversion & $\\kappa^+_{BH}-\\kappa^+_{off}$ \\\\\n\\midrule\n")
    for tag, lab in specs:
        f = os.path.join(TAB, f"emp_numbers{tag}.json")
        if not os.path.exists(f): continue
        d = json.load(open(f))["reg"]
        r, ri, rb = d["thr"], d["thr_in"], d["bh"]
        cells = [(r["coef"]["q"], r["se"]["q"], 3), (r["coef"]["up"], r["se"]["up"], 4), (r["coef"]["dn"], r["se"]["dn"], 4),
                 (ri["coef"]["negx_in"], ri["se"]["negx_in"], 4), (rb["coef"]["up_bh"], rb["se"]["up_bh"], 4)]
        fh.write(lab + " & " + " & ".join(f"${c:.{n}f}{st(c, s)}$" for c, s, n in cells) + " \\\\\n")
        fh.write(" & " + " & ".join(f"$({s:.{n}f})$" for c, s, n in cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(open(os.path.join(TAB, "emp_robust.tex")).read())

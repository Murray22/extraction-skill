#!/usr/bin/env python3
"""Figure 2 of the SSAC27 abstract: out-of-sample validation.

Panel A is computed here from committed tables (the ladder from l1_oos.csv, the raw-prior-rate bar
from analysis/20_naive_baseline.py). Panel B's quartile rates are 01_nhl_skill_model.py's output
(matched n=558), which needs the source database, so they are entered from its docstring.

Run from figures/:  python make_sloan_fig2.py   -> sloan_fig2.png
"""
import importlib.util, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "data", "derived")
NAVY, CYAN, ORANGE, MUTED, EDGE = "#0f2b48", "#0284c7", "#ea580c", "#64748b", "#cbd5e1"
QUARTILE_RATES = (0.41, 0.50, 0.62, 0.81)   # 01's docstring: held-out draws per hour by prior skill quartile

spec = importlib.util.spec_from_file_location("s20", os.path.join(HERE, "..", "analysis", "20_naive_baseline.py"))
s20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s20)


def ladder():
    pg = pd.read_parquet(os.path.join(DERIVED, "l1_playergames.parquet")); pg["season"] = pg.season.astype(int)
    oos = pd.read_csv(os.path.join(DERIVED, "l1_oos.csv"))
    k, _ = s20.choose_k(pg)
    m = s20.naive_predictions(pg, oos, k)
    return [s20.corr(m.drawn, m[c]) for c in ("pred_flat", "pred_ctx", "pred_model", "pred_naive")]


def main():
    r = ladder()
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": EDGE, "axes.titleweight": "bold", "axes.titlecolor": NAVY})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.1, 4.25), gridspec_kw={"width_ratios": [1.25, 1]})
    x = [0, 1, 2, 3.35]
    a.bar(x, r, width=0.62, color=[MUTED, CYAN, NAVY, ORANGE])
    for xi, ri in zip(x, r):
        a.text(xi, ri + 0.012, f"{ri:.2f}", ha="center", va="bottom", fontweight="bold", color=NAVY, fontsize=13)
    a.axvline(2.675, color=EDGE, lw=1, ls=(0, (4, 3)))
    a.set_xticks(x, ["Ice time\nonly", "+ Context\nmodel", "+ Skill\nlayer", "Raw prior\nrate alone"])
    a.set_ylim(0, 0.85); a.set_ylabel("Out-of-sample correlation", fontsize=10.5)
    a.set_title("A. Predicting held-out 2025-26 draw totals", fontsize=12.5)
    b.plot(range(4), QUARTILE_RATES, color=NAVY, lw=3, marker="o", ms=10, mfc=ORANGE, mec=NAVY, mew=1.2)
    for i, v in enumerate(QUARTILE_RATES):
        b.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontweight="bold", color=NAVY, fontsize=12)
    b.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"]); b.set_ylim(0.3, 0.95)
    b.set_xlabel("Prior skill quartile", fontsize=10.5); b.set_ylabel("Drawn per hour (held-out season)", fontsize=10.5)
    b.set_title("B. Held-out draw rate by skill quartile", fontsize=12.5)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(HERE, "sloan_fig2.png"), dpi=200)
    print("ladder r:", " / ".join(f"{v:.3f}" for v in r))


if __name__ == "__main__":
    main()

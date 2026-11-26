#!/usr/bin/env python3
"""Exploratory pilot (2026-11-24): how much of the student-teacher gap on a new scanner is the decision cutoff?

Uses only saved per-cell predictions (no GPU):
  data/pc_runs_2026-11-17/r18_crossfit_cells.csv.gz      ResNet-18 students (control / KD / KD+TGA), 10 seeds
  data/pc_runs_2026-11-17/teacher_newscanner_cells.csv.gz Optimus + LoRA teacher, view 0, its own source-val cutoff

Per run it reports: F1 with the source-validation cutoff, F1 with the best possible cutoff on the new scanner
(ORACLE: uses new-scanner labels, a diagnostic upper bound only), AUROC, and three label-free recalibrations to the
teacher, cross-fitted over halves A/B (fit on one half using teacher answers only, apply to the other half):
  platt   p' = sigmoid(a * logit(p) + b), (a, b) minimise BCE to the teacher's probabilities; decide with the teacher's cutoff
  bias    same with a = 1
  rate    student cutoff set so its positive rate matches the teacher's positive rate on the fit half
Not pre-specified, no multiplicity control: use to choose what to test properly, not as a result.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
c = pd.read_csv(D / "r18_crossfit_cells.csv.gz").merge(pd.read_csv(D / "teacher_newscanner_cells.csv.gz"), on=["holdout", "row"])
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
bce = lambda q, t: -(t * np.log(np.clip(q, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - q, 1e-7, 1))).mean()

def f1(pred, y):
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); return 2 * tp / max(2 * tp + fp + fn, 1)

def auroc(p, y):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def oracle(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return (2 * tp / (k + y.sum())).max()

rows = []
for (h, arm, s), g in c.groupby(["holdout", "arm", "seed"]):
    y = (g.y == 1).values; p = g.p_none_single.values; z = lg(p); tp_ = g.p_teacher.values
    thr = g.thr_single.values; tt = g.thr_teacher.values[0]
    out = dict(holdout=h, arm=arm, f1_source_cutoff=f1(p >= thr, y), f1_oracle_cutoff=oracle(p, y), auroc=auroc(p, y),
               f1_adabn=f1(g.p_adabn_single.values >= thr, y),
               teacher_f1=f1(tp_ >= tt, y), teacher_f1_oracle_cutoff=oracle(tp_, y), teacher_auroc=auroc(tp_, y))
    pred = {k: np.zeros(len(g), bool) for k in ("platt", "bias", "rate")}
    for fit, to in (("A", "B"), ("B", "A")):
        fi = (g.half == fit).values; pi = (g.half == to).values
        ab = minimize(lambda w: bce(sg(w[0] * z[fi] + w[1]), tp_[fi]), [1.0, 0.0], method="Nelder-Mead").x
        b = minimize(lambda w: bce(sg(z[fi] + w[0]), tp_[fi]), [0.0], method="Nelder-Mead").x[0]
        pred["platt"][pi] = sg(ab[0] * z[pi] + ab[1]) >= tt
        pred["bias"][pi] = sg(z[pi] + b) >= tt
        pred["rate"][pi] = p[pi] >= np.quantile(p[fi], 1 - (tp_[fi] >= tt).mean())
    out.update({f"f1_{k}_to_teacher": f1(v, y) for k, v in pred.items()})
    rows.append(out)
r = pd.DataFrame(rows)
m = r.groupby(["arm", "holdout"]).mean(numeric_only=True)
mm = m.groupby("arm").mean(); mm["holdout"] = "mean"; mm = mm.reset_index().set_index(["arm", "holdout"])
tab = pd.concat([m, mm]).sort_index().round(3)
tab.to_csv(D / "gap_decomposition_pilot.csv")
pd.set_option("display.width", 250); print(tab.T.to_string())

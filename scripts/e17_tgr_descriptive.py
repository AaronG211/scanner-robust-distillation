#!/usr/bin/env python3
"""Descriptive only (MULTI_TEACHER_PROTOCOL.md, 'descriptive' section; RECAL_PROTOCOL.md, 'same methods with the other
teachers'): teacher-guided recalibration (TGR) of each KD student with each available teacher, cross-fitted over the
image halves A/B, against the student's source cutoff, AdaBN and the cross-fitted oracle cutoff (labels, upper bound).

Rows: student arm (r18_kd = Optimus KD, r18_kd_<fm> = KD from <fm>, r18_control = no teacher).
Columns: TGR with teacher <fm> (the student's own teacher and the swapped ones).
F1 per run over all 50 new-scanner images, mean over seeds, then over folds. No test, no correction.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def f1(pred, y):
    tp = (pred & y).sum(); return 2 * tp / max(2 * tp + (pred & ~y).sum() + (~pred & y).sum(), 1)

def best_cut(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return p[o][np.argmax(2 * tp / (k + y.sum()))]

T = pd.read_csv(D / "teacher_newscanner_cells_multi.csv.gz")
fms = list(T.fm.unique())
cells = [pd.read_csv(D / "r18_crossfit_cells.csv.gz")]
cells += [pd.read_csv(D / f"r18_e17_{fm}_cells.csv.gz") for fm in fms if (D / f"r18_e17_{fm}_cells.csv.gz").exists()]
c = pd.concat(cells)
arms = ["r18_control", "r18_kd"] + [f"r18_kd_{fm}" for fm in fms if fm != "hopt0" and (c.arm == f"r18_kd_{fm}").any()]
c = c[c.arm.isin(arms)]

rows = []
for (h, arm, s), g in c.groupby(["holdout", "arm", "seed"]):
    g = g.sort_values("row"); y = (g.y == 1).values; p = g.p_none_single.values; z = lg(p)
    r = dict(holdout=h, arm=arm, seed=s, none=f1(p >= g.thr_single.values, y), adabn=f1(g.p_adabn_single.values >= g.thr_single.values, y))
    orc = np.zeros(len(g), bool)
    for fit, to in (("A", "B"), ("B", "A")):
        fi = (g.half == fit).values; pi = (g.half == to).values; orc[pi] = p[pi] >= best_cut(p[fi], y[fi])
    r["oracle"] = f1(orc, y)
    for fm in fms:
        t = T[(T.fm == fm) & (T.holdout == h)].set_index("row").loc[g.row.values]
        pt = t.p_teacher.values; zt = lg(np.array([t.thr_teacher.values[0]]))[0]
        pred = np.zeros(len(g), bool)
        for fit, to in (("A", "B"), ("B", "A")):
            fi = (g.half == fit).values; pi = (g.half == to).values; pred[pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= zt
        r[f"tgr_{fm}"] = f1(pred, y)
    rows.append(r)

r = pd.DataFrame(rows)
r.to_csv(D / "e17_tgr_descriptive.csv", index=False)
cols = ["none", "adabn", "oracle"] + [f"tgr_{fm}" for fm in fms]
per_fold = r.groupby(["arm", "holdout"])[cols].mean()
pd.set_option("display.width", 200)
print("New-scanner F1 (single view; mean over seeds, then folds):")
print(per_fold.groupby("arm").mean().reindex(arms).round(3).to_string()); print()
print("Per fold:"); print(per_fold.round(3).to_string())

#!/usr/bin/env python3
"""Descriptive, post hoc (2026-11-25): the few-image curve of E0.2 with two more lines, for Figure 3a.

  labelled   F1-optimal cutoff chosen with the LABELS of the cells of n images from the fit half, applied to the other
             half (what a lab would get by annotating n images on the new scanner)
  tgr        TGR from n unlabelled images (recomputed here for the ConvNeXt KD student, which E0.2 did not cover;
             for r18_kd it must reproduce e02_tgr_budget.csv, checked below)
Same image draws as E0.2 and the AdaBN budget run (pick(), identical seeding). Students: r18_kd, cnx_kd (Optimus).
Output: data/pc_runs_2026-11-17/e02_labelled_budget.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
FOLD_IDX = {"XR": 0, "S360": 1, "CS2": 2}
BUDGETS, DRAWS = [1, 2, 5, 10, 25], 10
CONFIGS = [(n, k) for n in BUDGETS for k in (range(DRAWS) if n < 25 else [0])]

def f1(pred, y):
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); return 2 * tp / max(2 * tp + fp + fn, 1)

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def best_cut(p, y):
    if y.sum() == 0:
        return np.inf                                 # no labelled positive among the chosen images: predict nothing
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return p[o][np.argmax(2 * tp / (k + y.sum()))]

def pick(imgs, fold, n, k, half):
    imgs = np.sort(imgs)
    if n >= len(imgs):
        return imgs
    return np.random.default_rng([2026, FOLD_IDX[fold], n, k, 0 if half == "A" else 1]).choice(imgs, n, replace=False)

te = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
c = pd.concat([pd.read_csv(D / "r18_crossfit_cells.csv.gz").query("arm == 'r18_kd'"),
               pd.read_csv(D / "cnx_crossfit_cells.csv.gz").query("arm == 'cnx_kd'")]).merge(te, on=["holdout", "row"])
rows = []
for (h, arm, s), g in c.groupby(["holdout", "arm", "seed"]):
    g = g.sort_values("row"); y = (g.y == 1).values; p = g.p_none_single.values; z = lg(p)
    pt = g.p_teacher.values; zt = lg(np.array([g.thr_teacher.values[0]]))[0]
    base = f1(p >= g.thr_single.values, y)
    for n, k in CONFIGS:
        pl = np.zeros(len(g), bool); pb = np.zeros(len(g), bool)
        for fit, to in (("A", "B"), ("B", "A")):
            chosen = pick(g.image_id[g.half == fit].unique(), h, n, k, fit)
            fi = (g.half == fit).values & g.image_id.isin(chosen).values; pi = (g.half == to).values
            pl[pi] = p[pi] >= best_cut(p[fi], y[fi]); pb[pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= zt
        rows.append(dict(holdout=h, arm=arm, seed=s, n_images=n, draw=k, f1_none=base, f1_labelled=f1(pl, y), f1_tgr=f1(pb, y)))
r = pd.DataFrame(rows); r.to_csv(D / "e02_labelled_budget.csv", index=False)
old = pd.read_csv(D / "e02_tgr_budget.csv").query("arm == 'r18_kd'")
m = r.query("arm == 'r18_kd'").merge(old, on=["holdout", "arm", "seed", "n_images", "draw"])
print("check vs e02_tgr_budget.csv (r18_kd TGR): max |diff| =", float(np.abs(m.f1_tgr - m.f1_tgr_bias).max()), "rows", len(m))
mean = r.groupby(["arm", "n_images", "holdout"])[["f1_none", "f1_labelled", "f1_tgr"]].mean().groupby(["arm", "n_images"]).mean()
print(mean.round(3).to_string())

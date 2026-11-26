#!/usr/bin/env python3
"""E0.2 (roadmap, exploratory): how many unlabelled new-scanner images does teacher-guided recalibration (TGR) need?

TGR: keep the student's weights; fit one bias b so that sigmoid(student logit + b) matches the teacher's probabilities
(binary cross-entropy) on the cells of n unlabelled images from the fit half; predict on the other half with the
teacher's own cutoff. Cross-fitted over halves A/B like everything else; 10 image draws per n, the same draws as the
AdaBN budget run on the PC (jobs/pc/e0_gpu_diag.py, pick()), so the two can be compared image for image.
Also reported: 'rate' (student cutoff set to reproduce the teacher's positive rate on the same cells).
No new-scanner label is used except for scoring. Students: r18_kd (Optimus KD, 10 seeds) and r18_control.

Output: data/pc_runs_2026-11-17/e02_tgr_budget.csv and e02_tgr_budget_preds.npz (per cell, same layout as the AdaBN file)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
c = pd.read_csv(D / "r18_crossfit_cells.csv.gz").merge(pd.read_csv(D / "teacher_newscanner_cells.csv.gz"), on=["holdout", "row"])
c = c[c.arm.isin(["r18_kd", "r18_control"])]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def f1(pred, y):
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); return 2 * tp / max(2 * tp + fp + fn, 1)

def fit_bias(z, pt):
    def loss(b):
        q = 1 / (1 + np.exp(-(z + b))); q = np.clip(q, 1e-7, 1 - 1e-7)
        return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

FOLD_IDX = {"XR": 0, "S360": 1, "CS2": 2}
BUDGETS, DRAWS = [1, 2, 5, 10, 25], 10
CONFIGS = [(n, k) for n in BUDGETS for k in (range(DRAWS) if n < 25 else [0])]

def pick(imgs, fold, n, k, half):
    """Deterministic image draw, identical to jobs/pc/e0_gpu_diag.py (independent of the student's seed)."""
    imgs = np.sort(imgs)
    if n >= len(imgs):
        return imgs
    return np.random.default_rng([2026, FOLD_IDX[fold], n, k, 0 if half == "A" else 1]).choice(imgs, n, replace=False)

rows, preds = [], {}
for (h, arm, s), g in c.groupby(["holdout", "arm", "seed"]):
    g = g.sort_values("row")
    P = np.zeros((len(CONFIGS), len(g)), bool)
    y = (g.y == 1).values; z = lg(g.p_none_single.values); pt = g.p_teacher.values; tt = g.thr_teacher.values[0]
    zt = lg(np.array([tt]))[0]
    base = f1(g.p_none_single.values >= g.thr_single.values, y)
    for ci, (n, k) in enumerate(CONFIGS):
        if True:
            pred_b = np.zeros(len(g), bool); pred_r = np.zeros(len(g), bool)
            for fit, to in (("A", "B"), ("B", "A")):
                chosen = pick(g.image_id[g.half == fit].unique(), h, n, k, fit)
                fi = (g.half == fit).values & g.image_id.isin(chosen).values; pi = (g.half == to).values
                b = fit_bias(z[fi], pt[fi]); pred_b[pi] = z[pi] + b >= zt
                rate = (pt[fi] >= tt).mean(); pred_r[pi] = z[pi] >= np.quantile(z[fi], 1 - rate) if 0 < rate < 1 else z[pi] >= np.inf
            P[ci] = pred_b
            rows.append(dict(holdout=h, arm=arm, seed=s, n_images=n, draw=k, f1_source_cutoff=base,
                             f1_tgr_bias=f1(pred_b, y), f1_tgr_rate=f1(pred_r, y)))
    preds[f"{h}|{arm}|{s}"] = P
r = pd.DataFrame(rows)
r.to_csv(D / "e02_tgr_budget.csv", index=False)
np.savez_compressed(D / "e02_tgr_budget_preds.npz", configs=np.array(CONFIGS), **preds)
per_fold = r.groupby(["arm", "n_images", "holdout"])[["f1_source_cutoff", "f1_tgr_bias", "f1_tgr_rate"]].mean()
mean = per_fold.groupby(["arm", "n_images"]).mean()
spread = r.groupby(["arm", "n_images", "draw"])[["f1_tgr_bias"]].mean().groupby(["arm", "n_images"]).agg(["min", "max"])
pd.set_option("display.width", 200)
print("F1 on the new scanner (mean over folds, seeds and draws):"); print(mean.round(3).to_string()); print()
print("Range of the fold-and-seed mean across image draws (bias variant):"); print(spread.round(3).to_string()); print()
print("Per fold, n = 5 images:"); print(per_fold.xs(5, level="n_images").round(3).to_string())

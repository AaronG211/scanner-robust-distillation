#!/usr/bin/env python3
"""Exploratory (not a pre-specified test): the student-teacher gap on the new scanner, layer by layer, for every teacher
whose E1.7 cells exist. ResNet-18 student, single view, all 50 new-scanner images per fold.

  S0 KD student, source cutoff
  S1 + TGR: one bias fitted to the teacher's probabilities on the other image half (cross-fitted)      -> operating point
  S2 + AdaBN, then TGR on the adapted logits (BN statistics from the other half; label-free, transductive) -> BN statistics
  S3 KD + TGA student (teacher answers on the other half's unlabelled cells during training), source cutoff -> representation
  S4 the teacher itself (view 0, its own source cutoff)
  extra: AdaBN alone (source cutoff), so the two post-hoc layers can also be credited order-free: Shapley values over
  {operating point, BN statistics}, i.e. each layer's gain averaged over the two orders.

Each step and each increment: F1 per run -> mean over seeds -> mean over folds; 95% interval from a paired bootstrap over
images (per fold) and seeds, B = 5000. Output: data/pc_runs_2026-11-17/layer_ladder.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 5000
rng = np.random.default_rng(20261127)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
STEPS = ["S0 KD", "S1 +TGR", "S2 +AdaBN+TGR", "S3 KD+TGA", "S4 teacher", "AdaBN only"]

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def tgr(z, half, pt, zt):
    out = np.zeros(len(z), bool)
    for fit, to in (("A", "B"), ("B", "A")):
        fi = half == fit; pi = half == to; out[pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= zt
    return out

T = pd.read_csv(D / "teacher_newscanner_cells_multi.csv.gz")
base = pd.read_csv(D / "r18_crossfit_cells.csv.gz")
rows = []
for fm in T.fm.unique():
    if fm == "hopt0":
        c, kd, tga = base, "r18_kd", "r18_kd_tga"
    else:
        f = D / f"r18_e17_{fm}_cells.csv.gz"
        if not f.exists():
            continue
        c, kd, tga = pd.read_csv(f), f"r18_kd_{fm}", f"r18_kd_tga_{fm}"
    per_fold = []
    for h in ("CS2", "S360", "XR"):
        th = T[(T.fm == fm) & (T.holdout == h)].set_index("row")
        gk = c[(c.holdout == h) & (c.arm == kd)]; gt = c[(c.holdout == h) & (c.arm == tga)]
        ref = gk[gk.seed == 0].sort_values("row"); y = (ref.y == 1).values; half = ref.half.values
        imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
        W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
        t = th.loc[ref.row.values]; pt = t.p_teacher.values; zt = lg(np.array([t.thr_teacher.values[0]]))[0]
        seeds = sorted(set(gk.seed) & set(gt.seed)); F = np.zeros((len(STEPS), B + 1, len(seeds)))
        for j, s in enumerate(seeds):
            a = gk[gk.seed == s].sort_values("row"); b = gt[gt.seed == s].sort_values("row")
            assert np.array_equal(a.row.values, ref.row.values) and np.array_equal(b.row.values, ref.row.values)
            preds = [a.p_none_single.values >= a.thr_single.values,
                     tgr(lg(a.p_none_single.values), half, pt, zt),
                     tgr(lg(a.p_adabn_single.values), half, pt, zt),
                     b.p_none_single.values >= b.thr_single.values,
                     pt >= t.thr_teacher.values[0],
                     a.p_adabn_single.values >= a.thr_single.values]
            for k, pred in enumerate(preds):
                cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
                tp, fp, fn = (W @ cnt).T; F[k, :, j] = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        sd = np.vstack([np.arange(len(seeds)), rng.integers(0, len(seeds), (B, len(seeds)))])
        per_fold.append(np.stack([np.take_along_axis(F[k], sd, 1).mean(1) for k in range(len(STEPS))]))
        rows += [dict(teacher=fm, fold=h, step=STEPS[k], f1=round(F[k, 0].mean(), 4)) for k in range(len(STEPS))]
    M = np.mean(per_fold, 0)                                  # steps x (B+1)
    for k in range(len(STEPS)):
        if STEPS[k] == "AdaBN only":
            rows.append(dict(teacher=fm, fold="mean", step=STEPS[k], f1=round(M[k, 0], 4),
                             lo=round(np.percentile(M[k, 1:], 2.5), 4), hi=round(np.percentile(M[k, 1:], 97.5), 4)))
            continue
        rows.append(dict(teacher=fm, fold="mean", step=STEPS[k], f1=round(M[k, 0], 4),
                         lo=round(np.percentile(M[k, 1:], 2.5), 4), hi=round(np.percentile(M[k, 1:], 97.5), 4)))
        if k and STEPS[k] != "AdaBN only":
            d = M[k] - M[k - 1]
            rows.append(dict(teacher=fm, fold="mean", step=f"{STEPS[k]} minus {STEPS[k-1]}", f1=round(d[0], 4),
                             lo=round(np.percentile(d[1:], 2.5), 4), hi=round(np.percentile(d[1:], 97.5), 4)))
    gap = M[4] - M[0]
    shap = {"Shapley: operating point (TGR)": ((M[1] - M[0]) + (M[2] - M[5])) / 2,
            "Shapley: BN statistics (AdaBN)": ((M[5] - M[0]) + (M[2] - M[1])) / 2}
    for name, v in shap.items():
        for lab, x in ((name, v), (name + ", share of gap", v / gap)):
            rows.append(dict(teacher=fm, fold="mean", step=lab, f1=round(x[0], 4),
                             lo=round(np.percentile(x[1:], 2.5), 4), hi=round(np.percentile(x[1:], 97.5), 4)))
    for k in (1, 2, 3):
        share = (M[k] - M[k - 1]) / gap
        rows.append(dict(teacher=fm, fold="mean", step=f"share of gap: {STEPS[k]}", f1=round(share[0], 3),
                         lo=round(np.percentile(share[1:], 2.5), 3), hi=round(np.percentile(share[1:], 97.5), 3)))

r = pd.DataFrame(rows); r.to_csv(D / "layer_ladder.csv", index=False)
pd.set_option("display.width", 200)
print(r[r.fold == "mean"].to_string(index=False)); print()
print(r[r.fold != "mean"].pivot_table(index=["teacher", "fold"], columns="step", values="f1")[STEPS].round(3).to_string())

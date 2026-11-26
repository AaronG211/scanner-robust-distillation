#!/usr/bin/env python3
"""Protocol T (TENT_PROTOCOL.md): TENT vs AdaBN and vs training-time adaptation, ResNet-18 students.

Inputs (data/pc_runs_2026-11-17/): e1_tent_cells.csv.gz (PC: jobs/pc/e0_gpu_diag.py tent), r18_crossfit_cells.csv.gz
(r18_kd_tga, and AdaBN for a consistency check), r18_exp2b_cells.csv.gz (r18_free_tga), teacher_newscanner_cells.csv.gz.
Usage: python3 scripts/tent_compare.py [path to tent cells]   (the optional path is for code checks on the smoke file)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 20000
rng = np.random.default_rng(20261128)
PRIMARY = (1e-3, 1)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

def stat(d):
    boot = d[1:]; p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (len(boot) + 1))
    return d[0], np.percentile(boot, 2.5), np.percentile(boot, 97.5), p

tent = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else D / "e1_tent_cells.csv.gz")
xf = pd.read_csv(D / "r18_crossfit_cells.csv.gz"); fr = pd.read_csv(D / "r18_exp2b_cells.csv.gz")
other = pd.concat([xf[xf.arm.isin(["r18_kd_tga", "r18_kd", "r18_control"])], fr[fr.arm == "r18_free_tga"]])
teacher = pd.read_csv(D / "teacher_newscanner_cells.csv.gz").set_index(["holdout", "row"])

# per (fold, arm, seed, config): predictions on all new-scanner cells (each half from the model fitted on the other half)
series = {}           # key (arm, method) -> per fold: (B+1, n_seeds) F1 array
desc = []
for (h, arm, lr, ep), g in tent.groupby(["holdout", "arm", "lr", "epochs"]):
    ap = g[g.role == "apply"]
    imgs = np.sort(ap.image_id.unique()); n = len(imgs)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
    F = {}
    for s, gs in ap.groupby("seed"):
        u = gs.sort_values("row"); y = (u.y == 1).values; ii = np.searchsorted(imgs, u.image_id.values); thr = u.thr_single.values
        preds = {"adabn": u.p_adabn.values >= thr, "tent": u.p_tent.values >= thr}
        # descriptive: TENT then TGR (bias fitted on the fit half's TENT probabilities against the Optimus teacher)
        tg = np.zeros(len(u), bool); fitrows = g[(g.role == "fit") & (g.seed == s)]
        for fit in ("A", "B"):
            fc = fitrows[fitrows.fit == fit]; tc = teacher.loc[[(h, r) for r in fc.row.values]]
            b = fit_bias(lg(fc.p_tent.values), tc.p_teacher.values); zt = lg(np.array([tc.thr_teacher.values[0]]))[0]
            m = (u.fit == fit).values; tg[m] = lg(u.p_tent.values[m]) + b >= zt
        preds["tent_tgr"] = tg
        if (lr, ep) == PRIMARY:
            for oarm in ("r18_kd_tga", "r18_free_tga"):
                o = other[(other.holdout == h) & (other.arm == oarm) & (other.seed == s)].set_index("row").loc[u.row.values]
                preds[oarm] = o.p_none_single.values >= o.thr_single.values
            a = other[(other.holdout == h) & (other.arm == arm) & (other.seed == s)].set_index("row").reindex(u.row.values)
            desc.append(dict(holdout=h, arm=arm, seed=s, adabn_max_abs_diff_vs_crossfit=np.nanmax(np.abs(a.p_adabn_single.values - u.p_adabn.values))))
        for k, pred in preds.items():
            cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
            tp, fp, fn = (W @ cnt).T; F.setdefault(k, []).append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
        desc.append(dict(holdout=h, arm=arm, seed=s, lr=lr, epochs=ep, pos_adabn=preds["adabn"].mean(), pos_tent=preds["tent"].mean()))
    for k, v in F.items():
        series.setdefault((arm, lr, ep, k), {})[h] = np.stack(v, 1)

def fold_mean(key, sd):
    per = series[key]; return np.mean([np.take_along_axis(per[h], sd[h], 1).mean(1) for h in sorted(per)], 0)

any_key = next(iter(series)); folds = sorted(series[any_key]); nS = series[any_key][folds[0]].shape[1]
sd = {h: np.vstack([np.arange(nS), rng.integers(0, nS, (B, nS))]) for h in folds}
table = pd.DataFrame([dict(arm=k[0], lr=k[1], epochs=k[2], method=k[3], **{h: series[k][h][0].mean() for h in folds},
                           mean=fold_mean(k, sd)[0]) for k in series])
pd.set_option("display.width", 220)
print("New-scanner F1 (single view, source cutoff unless stated):")
print(table.sort_values(["arm", "lr", "epochs", "method"]).round(3).to_string(index=False)); print()

rows = []
for arm, other_arm in (("r18_kd", "r18_kd_tga"), ("r18_control", "r18_free_tga")):
    if (arm, *PRIMARY, "tent") not in series:
        continue
    t = fold_mean((arm, *PRIMARY, "tent"), sd)
    for name, ref in (("AdaBN", (arm, *PRIMARY, "adabn")), (other_arm, (arm, *PRIMARY, other_arm))):
        est, lo, hi, p = stat(t - fold_mean(ref, sd))
        rows.append(dict(student=arm, contrast=f"TENT vs {name}", diff=round(est, 4), lo=round(lo, 4), hi=round(hi, 4), p=p))
out = pd.DataFrame(rows); out["holm"] = holm(out.p.values)
print(out.to_string(index=False)); print()
dd = pd.DataFrame(desc)
print("Consistency: max |AdaBN here - AdaBN in r18_crossfit_cells| =", np.nanmax(dd.adabn_max_abs_diff_vs_crossfit.values))
print("Positive rate on the new scanner (mean over folds and seeds):")
print(dd.dropna(subset=["lr"]).groupby(["arm", "lr", "epochs"])[["pos_adabn", "pos_tent"]].mean().round(3).to_string())
if len(sys.argv) == 1:
    out.to_csv(D / "tent_compare.csv", index=False); table.to_csv(D / "tent_table.csv", index=False)

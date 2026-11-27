#!/usr/bin/env python3
"""MULTI_TEACHER_PROTOCOL.md amendment 1: ConvNeXt-T students (no BatchNorm), Optimus teacher.

Pre-specified (one-contrast family): cnx_kd_tga vs cnx_kd, new-scanner F1, single view, source cutoff, cross-fitted over
all 50 images; paired bootstrap over images and seeds, B = 20000. Expectation: positive and significant.
Descriptive: cnx_control; TGR (Optimus teacher, cross-fitted) and the cross-fitted oracle cutoff for every arm; the same
ladder as scripts/layer_ladder.py (no AdaBN layer: LayerNorm has no running statistics); TTA columns.
Input: data/pc_runs_2026-11-17/cnx_crossfit_cells.csv.gz (PC: cf_eval_any.py --out cnx_crossfit)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 20000
rng = np.random.default_rng(20261129)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def best_cut(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return p[o][np.argmax(2 * tp / (k + y.sum()))]

def stat(d):
    boot = d[1:]; p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (len(boot) + 1))
    return d[0], np.percentile(boot, 2.5), np.percentile(boot, 97.5), p

c = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else D / "cnx_crossfit_cells.csv.gz")   # optional path: code checks only
T = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
assert np.allclose(c.p_none_single, c.p_adabn_single, atol=1e-5), "ConvNeXt has no BatchNorm: AdaBN columns should equal plain ones"
METHODS = ["source cutoff", "TTA", "TGR", "oracle cutoff"]
series, table = {}, []
for h in ("CS2", "S360", "XR"):
    g0 = c[c.holdout == h]; th = T[T.holdout == h].set_index("row")
    ref = g0[(g0.arm == g0.arm.iloc[0]) & (g0.seed == 0)].sort_values("row")
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    y = (ref.y == 1).values; half = ref.half.values
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
    t = th.loc[ref.row.values]; pt = t.p_teacher.values; zt = lg(np.array([t.thr_teacher.values[0]]))[0]
    for arm, ga in g0.groupby("arm"):
        F = {m: [] for m in METHODS}
        for s, g in ga.groupby("seed"):
            g = g.sort_values("row"); assert np.array_equal(g.row.values, ref.row.values)
            p = g.p_none_single.values; z = lg(p)
            tgr = np.zeros(len(g), bool); orc = np.zeros(len(g), bool)
            for fit, to in (("A", "B"), ("B", "A")):
                fi = half == fit; pi = half == to
                tgr[pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= zt; orc[pi] = p[pi] >= best_cut(p[fi], y[fi])
            preds = {"source cutoff": p >= g.thr_single.values, "TTA": g.p_none_tta.values >= g.thr_tta.values, "TGR": tgr, "oracle cutoff": orc}
            for m, pred in preds.items():
                cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
                tp, fp, fn = (W @ cnt).T; F[m].append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
        for m in METHODS:
            series.setdefault((arm, m), {})[h] = np.stack(F[m], 1)
            table.append(dict(arm=arm, method=m, fold=h, f1=series[(arm, m)][h][0].mean()))

nS = next(iter(series.values()))["XR"].shape[1]
sd = {h: np.vstack([np.arange(nS), rng.integers(0, nS, (B, nS))]) for h in ("CS2", "S360", "XR")}
mean = lambda k: np.mean([np.take_along_axis(series[k][h], sd[h], 1).mean(1) for h in sd], 0)
tb = pd.DataFrame(table).pivot_table(index=["arm", "method"], columns="fold", values="f1")
tb["mean"] = [mean(k)[0] for k in tb.index]
pd.set_option("display.width", 200)
print("ConvNeXt-T students, new-scanner F1 (mean over seeds, then folds):"); print(tb.round(3).to_string()); print()
est, lo, hi, p = stat(mean(("cnx_kd_tga", "source cutoff")) - mean(("cnx_kd", "source cutoff")))
print(f"PRE-SPECIFIED  cnx_kd_tga vs cnx_kd (source cutoff): {est:+.4f} [{lo:+.4f}, {hi:+.4f}], p = {p:.4f}")
if len(sys.argv) == 1:                            # E1.7 amendment 1 (one-contrast family, two-sided, no Holm step)
    pd.DataFrame([dict(test="E1.7a", contrast="cnx_kd_tga minus cnx_kd (source cutoff)", diff=est, lo=lo, hi=hi, p=p)]).to_csv(
        D / "cnx_e17a_contrast.csv", index=False)
for a_, m_, b_, n_ in (("cnx_kd", "TGR", "cnx_kd", "source cutoff"), ("cnx_kd_tga", "source cutoff", "cnx_kd", "TGR"),
                       ("cnx_kd", "source cutoff", "cnx_control", "source cutoff")):
    e, l, h_, _ = stat(mean((a_, m_)) - mean((b_, n_)))
    print(f"descriptive    {a_} {m_} minus {b_} {n_}: {e:+.4f} [{l:+.4f}, {h_:+.4f}]")
if len(sys.argv) == 1:
    tb.reset_index().to_csv(D / "cnx_table.csv", index=False)

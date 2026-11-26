#!/usr/bin/env python3
"""Corrected cutoff share of the student-teacher gap (reviewer point, Literature_Review_2026-11-25.md). Exploratory.

For every run: F1 with the source-validation cutoff (deployable), with a CROSS-FITTED oracle cutoff (chosen with labels
on one half of the new scanner's images, applied to the other half: an honest upper bound for "the right cutoff"),
and with the in-sample oracle cutoff (optimistic). Same for the Optimus teacher (one seed).
Gap shares: how much of the student-teacher gap at the deployable cutoff remains when BOTH use the oracle cutoff.
Bootstrap: images resampled within fold (same draw for student and teacher), seeds resampled; B = 5000.
"""
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
c = pd.read_csv(D / "r18_crossfit_cells.csv.gz").merge(pd.read_csv(D / "teacher_newscanner_cells.csv.gz"), on=["holdout", "row"])

def counts(pred, y):
    return np.array([(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()])

def best_cut(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); i = np.argmax(2 * tp / (k + y.sum()))
    return p[o][i]

def per_image_counts(g, p, cut_by_half):
    """(n_images, 3) TP/FP/FN per image for predictions p >= cut (cut may differ by half)."""
    imgs = np.sort(g.image_id.unique()); y = (g.y == 1).values
    cut = np.array([cut_by_half[h] for h in g.half.values]) if isinstance(cut_by_half, dict) else cut_by_half
    pred = p >= cut; cnt = np.zeros((len(imgs), 3))
    np.add.at(cnt, np.searchsorted(imgs, g.image_id.values), np.c_[pred & y, pred & ~y, ~pred & y]); return imgs, cnt

rng = np.random.default_rng(11); B = 5000
res = {}
for h, gh in c.groupby("holdout"):
    imgs = np.sort(gh.image_id.unique()); n = len(imgs)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
    f1 = lambda cnt: (lambda tp, fp, fn: 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))(*(W @ cnt).T)
    t0 = gh[(gh.arm == "r18_kd") & (gh.seed == 0)]; yt = (t0.y == 1).values; pt = t0.p_teacher.values
    cfx = {hf: best_cut(pt[(t0.half != hf).values], yt[(t0.half != hf).values]) for hf in ("A", "B")}   # chosen on the other half
    res.setdefault("teacher", {})[h] = {
        "source": f1(per_image_counts(t0, pt, t0.thr_teacher.values[0])[1]),
        "oracle_xfit": f1(per_image_counts(t0, pt, cfx)[1]),
        "oracle_insample": f1(per_image_counts(t0, pt, best_cut(pt, yt))[1])}
    for arm in ("r18_control", "r18_kd", "r18_kd_tga"):
        acc = {k: [] for k in ("source", "oracle_xfit", "oracle_insample")}
        for s, g in gh[gh.arm == arm].groupby("seed"):
            y = (g.y == 1).values; p = g.p_none_single.values
            cfx = {hf: best_cut(p[(g.half != hf).values], y[(g.half != hf).values]) for hf in ("A", "B")}
            acc["source"].append(f1(per_image_counts(g, p, g.thr_single.values)[1]))
            acc["oracle_xfit"].append(f1(per_image_counts(g, p, cfx)[1]))
            acc["oracle_insample"].append(f1(per_image_counts(g, p, best_cut(p, y))[1]))
        sd = np.vstack([np.arange(10), rng.integers(0, 10, (B, 10))])
        res.setdefault(arm, {})[h] = {k: np.take_along_axis(np.stack(v, 1), sd, 1).mean(1) for k, v in acc.items()}
M = {m: {k: np.mean([res[m][h][k] for h in res[m]], 0) for k in ("source", "oracle_xfit", "oracle_insample")} for m in res}
ci = lambda x: f"{x[0]:.3f} [{np.percentile(x[1:], 2.5):.3f}, {np.percentile(x[1:], 97.5):.3f}]"
print("F1 on the new scanner, mean over folds (95% CI, images + seeds):")
for m in ("teacher", "r18_control", "r18_kd", "r18_kd_tga"):
    print(f"  {m:12s} source cutoff {ci(M[m]['source'])}   oracle (cross-fitted) {ci(M[m]['oracle_xfit'])}   oracle (in-sample) {ci(M[m]['oracle_insample'])}")
for arm in ("r18_kd", "r18_control"):
    g_src = M["teacher"]["source"] - M[arm]["source"]; g_x = M["teacher"]["oracle_xfit"] - M[arm]["oracle_xfit"]
    share = 1 - g_x / g_src
    print(f"\n{arm}: teacher-student gap at source cutoffs {ci(g_src)}; with both at cross-fitted oracle cutoffs {ci(g_x)}")
    print(f"  share of the gap removed by fixing both cutoffs: {share[0]:.2f} [{np.percentile(share[1:], 2.5):.2f}, {np.percentile(share[1:], 97.5):.2f}]")

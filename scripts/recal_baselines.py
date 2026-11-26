#!/usr/bin/env python3
"""Descriptive, post hoc (added 2026-11-25 after the literature review; not part of RECAL_PROTOCOL.md or
PREVALENCE_SHIFT_PROTOCOL.md): the label-free recalibration baselines a reviewer will ask for, next to TGR, at the
natural prevalence and under the simulated prevalence shift of Protocol P (same resampling masks and bootstrap draws, so
the methods shared with scripts/prevalence_shift.py reproduce its table exactly; this is checked).

Students: r18_kd and cnx_kd (Optimus KD students; the only students with source-validation cells). Teacher: Optimus.
Everything cross-fitted over image halves A/B (fit on one half's unlabelled cells, decide on the other).

Teacher-free (reference = the student's own source-validation outputs):
  none            fixed source cutoff
  source_rate     cutoff reproducing the source positive rate                         (Protocol R)
  upa             Unsupervised Prediction Alignment (Roschewitz et al., Nat Commun 2023): piecewise-linear quantile
                  map of the target scores onto the source-validation scores (101 quantiles), then the source cutoff
  source_bias     one logit bias so the mean target probability equals the mean source-validation probability
                  (the batch-calibration idea: same mechanism as TGR, source reference instead of the teacher)
  em_prior        EM prior adjustment on raw probabilities                            (Protocol R)
  em_bcts         EM after bias-corrected temperature scaling fitted on labelled source-validation cells
                  (Alexandari et al., ICML 2020), then the prior-shift logit offset
  bbse            black-box shift estimation (Lipton et al., ICML 2018): confusion matrix at the source cutoff on
                  labelled source-validation cells, target prior from the predicted-positive rate, logit offset
Teacher-referenced (reference = the teacher's probabilities on the same unlabelled target cells):
  tgr             one bias, BCE to teacher probabilities, decide at the teacher's cutoff (Protocol R)
  teacher_rate    cutoff reproducing the teacher's positive rate                      (Protocol R)
  teacher_pl      F1-optimal cutoff against teacher pseudo-labels                     (Protocol R)
  teacher_bt      bias + temperature (two parameters), BCE to teacher probabilities, teacher's cutoff
  teacher_qmap    quantile map of the student's logits onto the teacher's logits (UPA with the teacher as reference)
Bounds: oracle (labelled cutoff from the other half), teacher_alone.

Outputs: data/pc_runs_2026-11-17/recal_baselines_table.csv (F1 by student, prevalence, method, fold, mean) and
recal_baselines_diffs.csv (TGR minus each method, paired bootstrap interval, at natural, 0.2 and 0.6).
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize, brentq

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "pc_runs_2026-11-17"
B, R = 20000, 20
FOLDS = ["CS2", "S360", "XR"]
PIS = ["natural", 0.20, 0.30, 0.40, 0.50, 0.60]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
L = lambda x: lg(np.array([x]))[0]

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(sg(z + b), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def fit_bt(z, t):
    def loss(v):
        q = np.clip(sg(v[0] * z + v[1]), 1e-7, 1 - 1e-7); return -(t * np.log(q) + (1 - t) * np.log(1 - q)).mean()
    return minimize(loss, x0=[1.0, 0.0], method="L-BFGS-B", bounds=[(0.05, 20), (-20, 20)]).x

def best_cut(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return p[o][np.argmax(2 * tp / (k + y.sum()))]

def rate_cut(p, rate):
    return np.inf if rate <= 0 else (-np.inf if rate >= 1 else np.quantile(p, 1 - rate))

def em_prior(p, pi_s, iters=200):
    pi_t = pi_s
    for _ in range(iters):
        a = (pi_t / pi_s) * p; b = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
        new = (a / (a + b)).mean()
        if abs(new - pi_t) < 1e-7:
            break
        pi_t = new
    return pi_t

def qmap(x_fit, ref, x_apply, K=101):
    lv = np.linspace(0, 1, K); qx = np.quantile(x_fit, lv) + np.arange(K) * 1e-9; qr = np.quantile(ref, lv)
    return np.interp(x_apply, qx, qr)

def bbse_prior(hv, yv, hf):
    C = np.array([[np.mean(~hv & ~yv), np.mean(~hv & yv)], [np.mean(hv & ~yv), np.mean(hv & yv)]])
    mu = np.array([np.mean(~hf), np.mean(hf)])
    try:
        w = np.clip(np.linalg.solve(C, mu), 0, None)
    except np.linalg.LinAlgError:
        w = np.ones(2)
    q = w * np.array([1 - yv.mean(), yv.mean()])
    return float(np.clip(q[1] / max(q.sum(), 1e-9), 0.01, 0.99))

ALL = ["none", "source_rate", "upa", "source_bias", "em_prior", "em_bcts", "bbse",
       "tgr", "teacher_rate", "teacher_pl", "teacher_bt", "teacher_qmap", "oracle"]

def methods(p, y, half, thr, vp, vy, pt, tt, cal):
    z = lg(p); zthr = L(thr); zt = L(tt); vz = lg(vp)
    src_rate = (vp >= thr).mean(); pi_s_raw = vp.mean(); pi_s = vy.mean(); a_c, b_c = cal
    out = {k: np.zeros(len(p), bool) for k in ALL}
    out["none"] = p >= thr
    for fit, to in (("A", "B"), ("B", "A")):
        fi = half == fit; pi = half == to
        zf, za = z[fi], z[pi]
        out["source_rate"][pi] = p[pi] >= rate_cut(p[fi], src_rate)
        out["upa"][pi] = qmap(p[fi], vp, p[pi]) >= thr
        f = lambda b: sg(zf + b).mean() - vp.mean()
        b = brentq(f, -15, 15) if f(-15) * f(15) < 0 else (-15 if f(-15) > 0 else 15)
        out["source_bias"][pi] = za + b >= zthr
        t = em_prior(p[fi], pi_s_raw); out["em_prior"][pi] = za + L(t) - L(pi_s_raw) >= zthr
        t = em_prior(sg(a_c * zf + b_c), pi_s); out["em_bcts"][pi] = a_c * za + b_c + L(t) - L(pi_s) >= a_c * zthr + b_c
        t = bbse_prior(vp >= thr, vy, p[fi] >= thr); out["bbse"][pi] = za + L(t) - L(pi_s) >= zthr
        out["tgr"][pi] = za + fit_bias(zf, pt[fi]) >= zt
        out["teacher_rate"][pi] = p[pi] >= rate_cut(p[fi], (pt[fi] >= tt).mean())
        out["teacher_pl"][pi] = p[pi] >= best_cut(p[fi], pt[fi] >= tt)
        a2, b2 = fit_bt(zf, pt[fi]); out["teacher_bt"][pi] = a2 * za + b2 >= zt
        out["teacher_qmap"][pi] = qmap(zf, lg(pt[fi]), za) >= zt
        out["oracle"][pi] = p[pi] >= best_cut(p[fi], y[fi])
    return out

def stat(d):
    return d[0], np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5)

cells = {"r18": pd.read_csv(D / "e0_cells_r18.csv.gz"), "cnx": pd.read_csv(D / "e0_cells_cnx.csv.gz")}
TE = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
STUDENTS = [("r18_kd", "r18"), ("cnx_kd", "cnx")]
point, boot, prior_log = [], {}, []
for fi_, h in enumerate(FOLDS):
    ref = cells["r18"]; ref = ref[(ref.holdout == h) & (ref.arm == "r18_kd") & (ref.seed == 0) & (ref.split == "unseen")].sort_values("row")
    rows = ref.row.values; y = (ref.y == 1).values; half = ref.half.values
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    th = TE[TE.holdout == h].set_index("row").loc[rows]; PT, TT = th.p_teacher.values, th.thr_teacher.values[0]
    masks = {"natural": [np.ones(len(rows), bool)]}                      # identical to prevalence_shift.py
    npos, nneg = y.sum(), (~y).sum()
    for k, pi in enumerate(PIS[1:]):
        g = np.random.default_rng(20261130 + 100 * fi_ + k)
        ms = []
        for _ in range(R):
            m = np.ones(len(rows), bool)
            if pi < y.mean():
                keep = int(round(pi / (1 - pi) * nneg)); drop = np.flatnonzero(y)[g.permutation(npos)[keep:]]
            else:
                keep = int(round((1 - pi) / pi * npos)); drop = np.flatnonzero(~y)[g.permutation(nneg)[keep:]]
            m[drop] = False; ms.append(m)
        masks[pi] = ms
    gb = np.random.default_rng([20261131, fi_])
    W = np.vstack([np.ones(n), gb.multinomial(n, np.full(n, 1 / n), size=B)]).astype(np.float32)
    sd = np.vstack([np.arange(10), gb.integers(0, 10, (B, 10))])
    for student, src in STUDENTS:
        c = cells[src]; c = c[(c.holdout == h) & (c.arm == student)]
        acc = {}
        for s, g in c.groupby("seed"):
            u = g[g.split == "unseen"].sort_values("row"); assert np.array_equal(u.row.values, rows)
            v = g[g.split == "val"]; vp = v.p_single.values; vy = (v.y == 1).values
            p = u.p_single.values; thr = u.thr_single.values[0]
            cal = fit_bt(lg(vp), vy.astype(float))                       # BCTS on labelled source-validation cells
            for pi, ms in masks.items():
                cnt = {}
                for m in ms:
                    preds = methods(p[m], y[m], half[m], thr, vp, vy, PT[m], TT, cal)
                    preds["teacher_alone"] = PT[m] >= TT
                    for meth, pr in preds.items():
                        a = cnt.setdefault(meth, np.zeros((n, 3)))
                        np.add.at(a, ii[m], np.c_[pr & y[m], pr & ~y[m], ~pr & y[m]])
                for meth, a in cnt.items():
                    tp, fp, fn = (W @ a.astype(np.float32)).T.astype(np.float64)
                    acc.setdefault((str(pi), meth), []).append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
        for (pi, meth), lst in acc.items():
            F = np.stack(lst, 1)
            point.append(dict(student=student, pi=pi, method=meth, fold=h, f1=F[0].mean()))
            boot.setdefault((student, pi, meth), []).append(np.take_along_axis(F, sd, 1).mean(1))
        print(f"{h} {student} done", flush=True)

tab = pd.DataFrame(point).pivot_table(index=["student", "pi", "method"], columns="fold", values="f1")
tab["mean"] = tab[FOLDS].mean(1)
series = {k: np.mean(v, 0) for k, v in boot.items()}
old = pd.read_csv(D / "prevalence_shift_table.csv").set_index(["student", "pi", "method"])
nchk = 0
for (st, pi, meth), row in tab.iterrows():
    if (st, pi, meth) in old.index:
        assert abs(row["mean"] - old.loc[(st, pi, meth), "mean"]) < 1e-9, ("MISMATCH vs prevalence_shift_table", st, pi, meth)
        nchk += 1
print(f"consistency: {nchk} (student, prevalence, method) cells reproduce prevalence_shift_table.csv exactly")

diffs = []
for st, _ in STUDENTS:
    for pi in ("natural", "0.2", "0.6"):
        for meth in ALL + ["teacher_alone"]:
            if meth == "tgr":
                continue
            e, lo, hi = stat(series[(st, pi, "tgr")] - series[(st, pi, meth)])
            diffs.append(dict(student=st, pi=pi, contrast=f"TGR minus {meth}", diff=e, lo=lo, hi=hi))
diffs = pd.DataFrame(diffs)
tab.reset_index().to_csv(D / "recal_baselines_table.csv", index=False); diffs.to_csv(D / "recal_baselines_diffs.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
wide = tab["mean"].unstack("pi")[[str(x) for x in PIS]]
wide = wide.reindex([(s, m) for s, _ in STUDENTS for m in ALL + ["teacher_alone"]])
print("\nNew-scanner F1 (mean over seeds, then folds) by target prevalence:"); print(wide.round(3).to_string())
print("\nTGR minus each method (paired bootstrap over images and seeds, 95% interval):")
print(diffs.round(4).to_string(index=False))

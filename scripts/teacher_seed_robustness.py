#!/usr/bin/env python3
"""Protocol F (TEACHER_SEED_AND_CONTROL_PROTOCOL.md): is the ConvNeXt-T student non-inferior to its H-optimus-0 teacher
for other trainings (LoRA seeds 1, 2) of that teacher? Written before the students existed.

Inputs (data/pc_runs_2026-11-17/): cnx_crossfit_tseeds_cells.csv.gz (cnx_kd_t1, cnx_kd_tga_t1, cnx_kd_t2, cnx_kd_tga_t2),
cnx_crossfit_cells.csv.gz (cnx_control, cnx_kd, cnx_kd_tga: teacher seed 0), teacher_newscanner_cells.csv.gz (teacher
seed 0, as in Protocol N), teacher_seeds_cells.csv.gz (teacher seeds 1, 2).
Usage: python3 scripts/teacher_seed_robustness.py [--check]
  --check   only the seed-0 term, to confirm it reproduces Protocol N's N1a (-0.0092, -0.0216 to +0.0028)
Output: data/pc_runs_2026-11-17/teacher_seed_robustness.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B, DELTA = 20000, 0.02
FOLDS = ["CS2", "S360", "XR"]
CHECK = "--check" in sys.argv
KS = [0] if CHECK else [0, 1, 2]
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

cnx = pd.read_csv(D / "cnx_crossfit_cells.csv.gz")
if not CHECK:
    cnx = pd.concat([cnx, pd.read_csv(D / "cnx_crossfit_tseeds_cells.csv.gz")])
t0 = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
ts = pd.read_csv(D / "teacher_seeds_cells.csv.gz"); ts = ts[ts.fm == "hopt0"]
ARMS = {0: ("cnx_kd", "cnx_kd_tga"), 1: ("cnx_kd_t1", "cnx_kd_tga_t1"), 2: ("cnx_kd_t2", "cnx_kd_tga_t2")}

S = {}                                       # key -> list over folds of (B+1,)
for fi, h in enumerate(FOLDS):
    g0 = cnx[cnx.holdout == h]
    ref = g0[(g0.arm == "cnx_kd") & (g0.seed == 0)].sort_values("row")
    rows = ref.row.values; y = (ref.y == 1).values; half = ref.half.values
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    gb = np.random.default_rng([20261201, fi])                     # identical draws to Protocol N
    W = np.vstack([np.ones(n), gb.multinomial(n, np.full(n, 1 / n), size=B)])
    sd = np.vstack([np.arange(10), gb.integers(0, 10, (B, 10))])
    def f1w(pred):
        cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
        tp, fp, fn = (W @ cnt).T; return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    def add(key, per_seed):
        S.setdefault(key, []).append(np.take_along_axis(np.stack(per_seed, 1), sd, 1).mean(1))
    def runs(arm):
        out = []
        for s in range(10):
            g = g0[(g0.arm == arm) & (g0.seed == s)].sort_values("row"); assert np.array_equal(g.row.values, rows), (h, arm, s)
            out.append(g)
        return out
    add("ctrl", [f1w(g.p_none_single.values >= g.thr_single.values) for g in runs("cnx_control")])
    for k in KS:
        if k == 0:
            t = t0[t0.holdout == h].set_index("row").loc[rows]
        else:
            t = ts[(ts.holdout == h) & (ts.seed == k)].set_index("row").loc[rows]
        pt, tt = t.p_teacher.values, t.thr_teacher.values[0]; zt = lg(np.array([tt]))[0]
        add(("teacher", k), [f1w(pt >= tt)] * 10)
        kd, tga = ARMS[k]
        add(("tga", k), [f1w(g.p_none_single.values >= g.thr_single.values) for g in runs(tga)])
        kdr = runs(kd)
        add(("kd", k), [f1w(g.p_none_single.values >= g.thr_single.values) for g in kdr])
        tg = []
        for g in kdr:
            z = lg(g.p_none_single.values); pred = np.zeros(len(rows), bool)
            for fit, to in (("A", "B"), ("B", "A")):
                fm, tm = half == fit, half == to; pred[tm] = z[tm] + fit_bias(z[fm], pt[fm]) >= zt
            tg.append(f1w(pred))
        add(("kd_tgr", k), tg)
M = {k: np.mean(v, 0) for k, v in S.items()}
ci = lambda d: (d[0], np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5))
p_ni = lambda d: ((d[1:] <= -DELTA).sum() + 1) / (B + 1)

print("F1 (mean over seeds, then folds): " + "; ".join(f"teacher s{k} {M[('teacher', k)][0]:.4f}, KD {M[('kd', k)][0]:.4f}, "
      f"KD+TGR {M[('kd_tgr', k)][0]:.4f}, KD+TGA {M[('tga', k)][0]:.4f}" for k in KS) + f"; no-teacher {M['ctrl'][0]:.4f}")
out = []
if CHECK:
    e, lo, hi = ci(M[("tga", 0)] - M[("teacher", 0)])
    print(f"CHECK seed-0 term: {e:.4f} ({lo:.4f} to {hi:.4f}); Protocol N's N1a was -0.0092 (-0.0216 to +0.0028)")
    sys.exit()
fam = []
for name, k in (("F1", 1), ("F2", 2)):
    d = M[("tga", k)] - M[("teacher", k)]; e, lo, hi = ci(d)
    fam.append(dict(test=name, contrast=f"cnx_kd_tga_t{k} minus teacher LoRA seed {k}", diff=e, lo=lo, hi=hi, p_one_sided=p_ni(d)))
for r, a in zip(fam, holm([r["p_one_sided"] for r in fam])):
    r["holm"] = a; r["non_inferior"] = bool(r["lo"] > -DELTA and a < 0.025)
d = np.mean([M[("tga", k)] - M[("teacher", k)] for k in (0, 1, 2)], 0); e, lo, hi = ci(d)
fam.append(dict(test="key secondary", contrast="mean over teacher seeds 0-2 of (KD + TGA minus its teacher); k = 0 already seen",
                diff=e, lo=lo, hi=hi, p_one_sided=p_ni(d), holm=np.nan, non_inferior=bool(lo > -DELTA and p_ni(d) < 0.025)))
fam = pd.DataFrame(fam)
desc = []
def put(q, x):
    e, lo, hi = ci(x); desc.append(dict(quantity=q, value=e, lo=lo, hi=hi))
for k in (0, 1, 2):
    adv = M[("teacher", k)] - M["ctrl"]
    put(f"teacher seed {k}: share of the teacher's advantage recovered by KD + TGA", (M[("tga", k)] - M["ctrl"]) / adv)
    put(f"teacher seed {k}: share recovered by KD", (M[("kd", k)] - M["ctrl"]) / adv)
    put(f"teacher seed {k}: share recovered by KD + TGR", (M[("kd_tgr", k)] - M["ctrl"]) / adv)
    put(f"teacher seed {k}: KD + TGR minus teacher", M[("kd_tgr", k)] - M[("teacher", k)])
    put(f"teacher seed {k}: KD + TGA minus KD", M[("tga", k)] - M[("kd", k)])
    put(f"teacher seed {k}: KD + TGA F1", M[("tga", k)]); put(f"teacher seed {k}: teacher F1", M[("teacher", k)])
put("range of KD + TGA F1 over teacher seeds (max - min)", np.ptp(np.stack([M[("tga", k)] for k in (0, 1, 2)]), 0))
put("range of teacher F1 over teacher seeds (max - min)", np.ptp(np.stack([M[("teacher", k)] for k in (0, 1, 2)]), 0))
desc = pd.DataFrame(desc)
pd.set_option("display.width", 220)
print("\nPRE-SPECIFIED family F (margin 0.02) and key secondary:"); print(fam.round(4).to_string(index=False))
print("\nDescriptive:"); print(desc.round(4).to_string(index=False))
pd.concat([fam.assign(kind="pre-specified"), desc.assign(kind="descriptive")]).to_csv(D / "teacher_seed_robustness.csv", index=False)

#!/usr/bin/env python3
"""Protocol C (TEACHER_SEED_AND_CONTROL_PROTOCOL.md): how much of the ConvNeXt-T student's retraining gain needs a
teacher? Written before the cnx_free_tga students existed.

Inputs (data/pc_runs_2026-11-17/): cnx_crossfit_free_cells.csv.gz (cnx_free_tga), cnx_crossfit_cells.csv.gz
(cnx_control, cnx_kd, cnx_kd_tga), cnx_crossfit_virchow2_cells.csv.gz (cnx_kd_virchow2, cnx_kd_tga_virchow2),
teacher_newscanner_cells.csv.gz and teacher_newscanner_cells_multi.csv.gz (for the descriptive KD + TGR rows).
Usage: python3 scripts/cnx_free_control.py [--check]
  --check   without the new file: only the existing arms, to confirm cnx_kd_tga's share (0.8825) is reproduced
Output: data/pc_runs_2026-11-17/cnx_free_control.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 20000
FOLDS = ["CS2", "S360", "XR"]
CHECK = "--check" in sys.argv
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

frames = [pd.read_csv(D / "cnx_crossfit_cells.csv.gz"), pd.read_csv(D / "cnx_crossfit_virchow2_cells.csv.gz")]
if not CHECK:
    frames.append(pd.read_csv(D / "cnx_crossfit_free_cells.csv.gz"))
cnx = pd.concat(frames)
TE = {"hopt0": pd.read_csv(D / "teacher_newscanner_cells.csv.gz")}
tm = pd.read_csv(D / "teacher_newscanner_cells_multi.csv.gz"); TE["virchow2"] = tm[tm.fm == "virchow2"]
ARMS = ["cnx_control", "cnx_kd", "cnx_kd_tga", "cnx_kd_virchow2", "cnx_kd_tga_virchow2"] + ([] if CHECK else ["cnx_free_tga"])

S = {}
for fi, h in enumerate(FOLDS):
    g0 = cnx[cnx.holdout == h]
    ref = g0[(g0.arm == "cnx_kd") & (g0.seed == 0)].sort_values("row")
    rows = ref.row.values; y = (ref.y == 1).values; half = ref.half.values
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    gb = np.random.default_rng([20261201, fi])
    W = np.vstack([np.ones(n), gb.multinomial(n, np.full(n, 1 / n), size=B)])
    sd = np.vstack([np.arange(10), gb.integers(0, 10, (B, 10))])
    def f1w(pred):
        cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
        tp, fp, fn = (W @ cnt).T; return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    def add(key, per_seed):
        S.setdefault(key, []).append(np.take_along_axis(np.stack(per_seed, 1), sd, 1).mean(1))
    gs = {}
    for arm in ARMS:
        gs[arm] = []
        for s in range(10):
            g = g0[(g0.arm == arm) & (g0.seed == s)].sort_values("row"); assert np.array_equal(g.row.values, rows), (h, arm, s)
            gs[arm].append(g)
        add(arm, [f1w(g.p_none_single.values >= g.thr_single.values) for g in gs[arm]])
    for fm, kd in (("hopt0", "cnx_kd"), ("virchow2", "cnx_kd_virchow2")):
        t = TE[fm][TE[fm].holdout == h].set_index("row").loc[rows]; pt = t.p_teacher.values; zt = lg(np.array([t.thr_teacher.values[0]]))[0]
        add(("teacher", fm), [f1w(pt >= t.thr_teacher.values[0])] * 10)
        tg = []
        for g in gs[kd]:
            z = lg(g.p_none_single.values); pred = np.zeros(len(rows), bool)
            for fit, to in (("A", "B"), ("B", "A")):
                fmask, tmask = half == fit, half == to; pred[tmask] = z[tmask] + fit_bias(z[fmask], pt[fmask]) >= zt
            tg.append(f1w(pred))
        add(f"{kd}+tgr", tg)
M = {k: np.mean(v, 0) for k, v in S.items()}
ci = lambda d: (d[0], np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5))
p2 = lambda d: min(1.0, 2 * (min((d[1:] <= 0).sum(), (d[1:] >= 0).sum()) + 1) / (B + 1))
print("F1: " + "; ".join(f"{k} {v[0]:.4f}" for k, v in M.items() if isinstance(k, str)))
if CHECK:
    x = (M["cnx_kd_tga"] - M["cnx_control"]) / (M[("teacher", "hopt0")] - M["cnx_control"])
    print(f"CHECK share recovered by cnx_kd_tga: {x[0]:.4f} (expected 0.8825)"); sys.exit()
fam = []
for name, arm in (("C1", "cnx_kd_tga"), ("C2", "cnx_kd_tga_virchow2")):
    d = M[arm] - M["cnx_free_tga"]; e, lo, hi = ci(d)
    fam.append(dict(test=name, contrast=f"{arm} minus cnx_free_tga (two-sided)", diff=e, lo=lo, hi=hi, p=p2(d)))
for r, a in zip(fam, holm([r["p"] for r in fam])):
    r["holm"] = a; r["significant"] = bool(a < 0.05)
fam = pd.DataFrame(fam)
desc = []
def put(q, x):
    e, lo, hi = ci(x); desc.append(dict(quantity=q, value=e, lo=lo, hi=hi))
shares = {}
for fm, arm in (("hopt0", "cnx_kd_tga"), ("virchow2", "cnx_kd_tga_virchow2")):
    x = (M["cnx_free_tga"] - M["cnx_control"]) / (M[arm] - M["cnx_control"]); shares[fm] = x[0]
    put(f"share of {arm}'s gain over cnx_control that needs no teacher", x)
    put(f"cnx_free_tga: share of the {fm} teacher's advantage recovered", (M["cnx_free_tga"] - M["cnx_control"]) / (M[("teacher", fm)] - M["cnx_control"]))
put("cnx_free_tga minus cnx_control", M["cnx_free_tga"] - M["cnx_control"])
put("cnx_free_tga minus cnx_kd + TGR (H-optimus-0)", M["cnx_free_tga"] - M["cnx_kd+tgr"])
put("cnx_free_tga minus cnx_kd_virchow2 + TGR (Virchow2)", M["cnx_free_tga"] - M["cnx_kd_virchow2+tgr"])
desc = pd.DataFrame(desc)
pd.set_option("display.width", 220)
print("\nPRE-SPECIFIED family C (Holm, two-sided 0.05):"); print(fam.round(4).to_string(index=False))
print("\nDescriptive:"); print(desc.round(4).to_string(index=False))
rule = all(v >= 0.5 for v in shares.values())
print("\nPre-committed wording rule (RQ3b):", "both shares >= 0.5 -> 'for both students most of the retraining gain comes from the unlabelled images'"
      if rule else "a share < 0.5 -> say it only for ResNet-18; for ConvNeXt-T the teacher contributes most of the retraining gain")
pd.concat([fam.assign(kind="pre-specified"), desc.assign(kind="descriptive")]).to_csv(D / "cnx_free_control.csv", index=False)

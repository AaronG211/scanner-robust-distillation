#!/usr/bin/env python3
"""Descriptive (not a pre-specified test): the layer ladder of both students in ONE unit, the share of the teacher's
advantage recovered = (F1(arm) - F1(no-teacher student)) / (F1(teacher) - F1(no-teacher student)), with paired
bootstrap intervals. Replaces the hand-computed ResNet shares (14/50/64/71/75%) in the story document.

Rungs (single view, new-scanner F1, cross-fitted over image halves A/B; each student uses its own teacher for TGR):
  ResNet-18, teachers hopt0 / virchow2 / uni / uni2h:
    ctrl (no-teacher student, source cutoff) | kd | kd + TGR | kd + AdaBN | kd + AdaBN + TGR |
    kd + AdaBN + TENT + TGR (Optimus only: TENT cells exist only for the Optimus KD student) | kd + TGA |
    free TGA (adaptation on the same unlabelled images without any foundation model; teacher-independent) | teacher
  ConvNeXt-T, teachers hopt0 / virchow2: ctrl | kd | kd + TGR | kd + TGA | teacher
Order-free credit for the two post-hoc layers of the ResNet student (Shapley over {threshold, BN statistics}).

Bootstrap: B = 20000, generator [20261201, fold index] per fold (folds CS2, S360, XR) drawn in the same order as
scripts/noninferiority.py, so the ConvNeXt shares reproduce that script exactly; image weights and seed resampling are
shared by every arm of a fold. Output: data/pc_runs_2026-11-17/ladder_recovered.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 20000
FOLDS = ["CS2", "S360", "XR"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def tgr(z, half, pt, zt):
    out = np.zeros(len(z), bool)
    for fit, to in (("A", "B"), ("B", "A")):
        fi = half == fit; pi = half == to; out[pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= zt
    return out

# ---- inputs ----
TM = pd.read_csv(D / "teacher_newscanner_cells_multi.csv.gz")
t1 = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
chk = TM[TM.fm == "hopt0"].merge(t1, on=["holdout", "row"], suffixes=("", "_1"))
assert len(chk) == len(t1) and np.allclose(chk.p_teacher, chk.p_teacher_1) and np.allclose(chk.thr_teacher, chk.thr_teacher_1)
r18 = {"hopt0": pd.read_csv(D / "r18_crossfit_cells.csv.gz")}
for fm in ("virchow2", "uni", "uni2h"):
    r18[fm] = pd.read_csv(D / f"r18_e17_{fm}_cells.csv.gz")
free = pd.read_csv(D / "r18_exp2b_cells.csv.gz"); free = free[free.arm == "r18_free_tga"]
cnx = {"hopt0": pd.read_csv(D / "cnx_crossfit_cells.csv.gz")}
cnx["virchow2"] = pd.read_csv(D / "cnx_crossfit_virchow2_cells.csv.gz")
CNX_FREE = D / "cnx_crossfit_free_cells.csv.gz"                  # Protocol C (TEACHER_SEED_AND_CONTROL_PROTOCOL.md), once it exists
cnx_free = pd.read_csv(CNX_FREE) if CNX_FREE.exists() else None
tent = pd.read_csv(D / "e1_tent_cells.csv.gz")
tent = tent[(tent.arm == "r18_kd") & (tent.lr == 1e-3) & (tent.epochs == 1)]

def runs(df, arm, h, rows):
    """seed -> frame sorted by row, for one arm and fold; checks the cell order."""
    out = {}
    for s, g in df[(df.holdout == h) & (df.arm == arm)].groupby("seed"):
        g = g.sort_values("row"); assert np.array_equal(g.row.values, rows), (arm, h, s); out[s] = g
    assert sorted(out) == list(range(10)), (arm, h, sorted(out))
    return out

S = {}                                               # (student, teacher, rung) -> list over folds of (B+1,) F1
for fi, h in enumerate(FOLDS):
    ref = cnx["hopt0"]; ref = ref[(ref.holdout == h) & (ref.arm == "cnx_kd") & (ref.seed == 0)].sort_values("row")
    rows = ref.row.values; y = (ref.y == 1).values; half = ref.half.values
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    gb = np.random.default_rng([20261201, fi])
    W = np.vstack([np.ones(n), gb.multinomial(n, np.full(n, 1 / n), size=B)])
    sd = np.vstack([np.arange(10), gb.integers(0, 10, (B, 10))])

    def f1w(pred):
        cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
        tp, fp, fn = (W @ cnt).T; return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)

    def add(key, per_seed):                          # per_seed: list of 10 (B+1,) arrays -> resample seeds -> mean
        S.setdefault(key, []).append(np.take_along_axis(np.stack(per_seed, 1), sd, 1).mean(1))

    T = {}
    for fm, g in TM[TM.holdout == h].groupby("fm"):
        g = g.set_index("row").loc[rows]; T[fm] = (g.p_teacher.values, g.thr_teacher.values[0])
    src = lambda g: g.p_none_single.values >= g.thr_single.values

    # ResNet-18
    ctrl_r = runs(r18["hopt0"], "r18_control", h, rows)
    free_r = runs(free, "r18_free_tga", h, rows)
    f_ctrl = [f1w(src(ctrl_r[s])) for s in range(10)]; f_free = [f1w(src(free_r[s])) for s in range(10)]
    tent_apply = tent[(tent.holdout == h) & (tent.role == "apply")]
    tent_fit = tent[(tent.holdout == h) & (tent.role == "fit")]
    for fm in ("hopt0", "virchow2", "uni", "uni2h"):
        kd_arm, tga_arm = ("r18_kd", "r18_kd_tga") if fm == "hopt0" else (f"r18_kd_{fm}", f"r18_kd_tga_{fm}")
        kd = runs(r18[fm], kd_arm, h, rows); tga = runs(r18[fm], tga_arm, h, rows)
        pt, tt = T[fm]; zt = lg(np.array([tt]))[0]
        acc = {k: [] for k in ("kd", "kd_tgr", "kd_adabn", "kd_adabn_tgr", "kd_tga", "kd_adabn_tent_tgr")}
        for s in range(10):
            a = kd[s]; z = lg(a.p_none_single.values); za = lg(a.p_adabn_single.values)
            acc["kd"].append(f1w(src(a)))
            acc["kd_tgr"].append(f1w(tgr(z, half, pt, zt)))
            acc["kd_adabn"].append(f1w(a.p_adabn_single.values >= a.thr_single.values))
            acc["kd_adabn_tgr"].append(f1w(tgr(za, half, pt, zt)))
            acc["kd_tga"].append(f1w(src(tga[s])))
            if fm == "hopt0":                        # TENT (after AdaBN), then TGR fitted on the fit half (as tent_compare.py)
                u = tent_apply[tent_apply.seed == s].set_index("row").loc[rows]
                pred = np.zeros(len(rows), bool)
                for fit in ("A", "B"):
                    fc = tent_fit[(tent_fit.seed == s) & (tent_fit.fit == fit)]
                    pos = np.searchsorted(rows, fc.row.values); assert np.array_equal(rows[pos], fc.row.values)
                    b = fit_bias(lg(fc.p_tent.values), pt[pos]); m = (u.fit == fit).values
                    pred[m] = lg(u.p_tent.values[m]) + b >= zt
                acc["kd_adabn_tent_tgr"].append(f1w(pred))
        add(("resnet18", fm, "ctrl"), f_ctrl); add(("resnet18", fm, "free_tga"), f_free)
        for k, v in acc.items():
            if v:
                add(("resnet18", fm, k), v)
        add(("resnet18", fm, "teacher"), [f1w(pt >= tt)] * 10)

    # ConvNeXt-T
    ctrl_c = runs(cnx["hopt0"], "cnx_control", h, rows)
    f_ctrl_c = [f1w(src(ctrl_c[s])) for s in range(10)]
    f_free_c = [f1w(src(g)) for g in runs(cnx_free, "cnx_free_tga", h, rows).values()] if cnx_free is not None else None
    for fm in ("hopt0", "virchow2"):
        sfx = "" if fm == "hopt0" else f"_{fm}"
        kd = runs(cnx[fm], f"cnx_kd{sfx}", h, rows); tga = runs(cnx[fm], f"cnx_kd_tga{sfx}", h, rows)
        pt, tt = T[fm]; zt = lg(np.array([tt]))[0]
        add(("convnext_t", fm, "ctrl"), f_ctrl_c)
        add(("convnext_t", fm, "kd"), [f1w(src(kd[s])) for s in range(10)])
        add(("convnext_t", fm, "kd_tgr"), [f1w(tgr(lg(kd[s].p_none_single.values), half, pt, zt)) for s in range(10)])
        add(("convnext_t", fm, "kd_tga"), [f1w(src(tga[s])) for s in range(10)])
        add(("convnext_t", fm, "teacher"), [f1w(pt >= tt)] * 10)
        if f_free_c is not None:
            add(("convnext_t", fm, "free_tga"), f_free_c)
    print(h, "done", flush=True)

M = {k: np.mean(v, 0) for k, v in S.items()}
FOLD = {k: [x[0] for x in v] for k, v in S.items()}                 # per-fold point estimates (folds in FOLDS order)
ci = lambda x: (x[0], np.percentile(x[1:], 2.5), np.percentile(x[1:], 97.5))
out = []
def put(student, fm, kind, name, x):
    e, lo, hi = ci(x); out.append(dict(student=student, teacher=fm, kind=kind, quantity=name, value=e, lo=lo, hi=hi))

for (student, fm) in sorted({k[:2] for k in M}):
    g = lambda r: M[(student, fm, r)]
    adv = g("teacher") - g("ctrl")
    rungs = [r for r in ("ctrl", "kd", "kd_tgr", "kd_adabn", "kd_adabn_tgr", "kd_adabn_tent_tgr", "kd_tga", "free_tga", "teacher")
             if (student, fm, r) in M]
    for r in rungs:
        put(student, fm, "f1", r, g(r))
        for h, v in zip(FOLDS, FOLD[(student, fm, r)]):
            out.append(dict(student=student, teacher=fm, kind="f1_fold", quantity=f"{r} | {h}", value=v, lo=np.nan, hi=np.nan))
    for r in rungs:
        if r not in ("ctrl", "teacher"):
            put(student, fm, "share", r, (g(r) - g("ctrl")) / adv)
    put(student, fm, "f1_diff", "teacher advantage (teacher - ctrl)", adv)
    put(student, fm, "f1_diff", "kd - ctrl", g("kd") - g("ctrl"))
    put(student, fm, "f1_diff", "kd_tga - kd", g("kd_tga") - g("kd"))
    put(student, fm, "f1_diff", "kd_tgr - kd", g("kd_tgr") - g("kd"))
    put(student, fm, "f1_diff", "teacher - kd_tga", g("teacher") - g("kd_tga"))
    # components in units of the teacher's advantage (they sum to 1 with the remaining gap)
    comp = {"distillation (kd - ctrl)": g("kd") - g("ctrl")}
    if student == "resnet18":
        comp["threshold, Shapley"] = ((g("kd_tgr") - g("kd")) + (g("kd_adabn_tgr") - g("kd_adabn"))) / 2
        comp["BN statistics, Shapley"] = ((g("kd_adabn") - g("kd")) + (g("kd_adabn_tgr") - g("kd_tgr"))) / 2
        comp["retraining (kd_tga - kd_adabn_tgr)"] = g("kd_tga") - g("kd_adabn_tgr")
    else:
        comp["threshold (kd_tgr - kd)"] = g("kd_tgr") - g("kd")
        comp["retraining (kd_tga - kd_tgr)"] = g("kd_tga") - g("kd_tgr")
    comp["remaining (teacher - kd_tga)"] = g("teacher") - g("kd_tga")
    for name, x in comp.items():
        put(student, fm, "component_f1", name, x); put(student, fm, "component_share", name, x / adv)
    if student == "resnet18":
        put(student, fm, "component_share", "post-hoc layers together (kd_adabn_tgr - kd)", (g("kd_adabn_tgr") - g("kd")) / adv)
        put(student, fm, "f1_diff", "kd_tga - free_tga (teacher's extra in retraining)", g("kd_tga") - g("free_tga"))
    if student == "convnext_t" and (student, fm, "free_tga") in M:
        put(student, fm, "f1_diff", "kd_tga - free_tga (teacher's extra in retraining)", g("kd_tga") - g("free_tga"))
        put(student, fm, "f1_diff", "free_tga - ctrl", g("free_tga") - g("ctrl"))
        put(student, fm, "share_of_retraining_gain", "(free_tga - ctrl) / (kd_tga - ctrl)",
            (g("free_tga") - g("ctrl")) / (g("kd_tga") - g("ctrl")))
    if student == "resnet18" and fm == "hopt0":
        put(student, fm, "f1_diff", "kd_tga - kd_adabn_tent_tgr", g("kd_tga") - g("kd_adabn_tent_tgr"))
    if student == "resnet18":
        put(student, fm, "share_of_retraining_gain", "(free_tga - ctrl) / (kd_tga - ctrl)",
            (g("free_tga") - g("ctrl")) / (g("kd_tga") - g("ctrl")))

r = pd.DataFrame(out); r.to_csv(D / "ladder_recovered.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
for kind in ("f1", "share", "component_share", "f1_diff"):
    print(f"--- {kind}"); print(r[r.kind == kind].drop(columns="kind").round(3).to_string(index=False))
# consistency with earlier scripts
exp = {("convnext_t", "hopt0", "share", "kd_tga"): 0.8825, ("convnext_t", "virchow2", "share", "kd_tga"): 0.9917,
       ("convnext_t", "hopt0", "share", "kd_tgr"): 0.8282, ("resnet18", "hopt0", "f1", "kd_tga"): 0.788,
       ("resnet18", "hopt0", "f1", "kd"): 0.725, ("resnet18", "hopt0", "f1", "kd_adabn_tent_tgr"): 0.784}
for (st, fm, kind, q), v in exp.items():
    got = r[(r.student == st) & (r.teacher == fm) & (r.kind == kind) & (r.quantity == q)].value.iloc[0]
    print("check", st, fm, kind, q, round(got, 4), "expected", v, "OK" if abs(got - v) < 6e-4 else "MISMATCH")

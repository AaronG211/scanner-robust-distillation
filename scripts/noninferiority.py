#!/usr/bin/env python3
"""Protocol N (NONINFERIORITY_PROTOCOL.md): is the ConvNeXt-T student non-inferior to its teacher on the new scanner?

Usage: python3 scripts/noninferiority.py [--teacher hopt0|virchow2]
  hopt0    family N1 (cnx_kd_tga, cnx_kd + TGR vs the Optimus teacher) + descriptives
  virchow2 family N2 (needs cnx_crossfit_virchow2_cells.csv.gz from the PC)
Inputs in data/pc_runs_2026-11-17/. Writes noninferiority_<teacher>.csv there.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
TEACHER = sys.argv[sys.argv.index("--teacher") + 1] if "--teacher" in sys.argv else "hopt0"
B, DELTA = 20000, 0.02
FOLDS = ["CS2", "S360", "XR"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def fit_bias(z, pt):
    def loss(b):
        q = np.clip(1 / (1 + np.exp(-(z + b))), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def best_cut(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return p[o][np.argmax(2 * tp / (k + y.sum()))]

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

# ---- inputs ----
cnx = pd.read_csv(D / "cnx_crossfit_cells.csv.gz")
sfx = "" if TEACHER == "hopt0" else f"_{TEACHER}"
if TEACHER == "hopt0":
    te = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
else:
    te = pd.read_csv(D / "teacher_newscanner_cells_multi.csv.gz"); te = te[te.fm == TEACHER]
    cnx = pd.concat([cnx[cnx.arm == "cnx_control"], pd.read_csv(D / f"cnx_crossfit{sfx}_cells.csv.gz")])
r18 = pd.read_csv(D / "r18_crossfit_cells.csv.gz") if TEACHER == "hopt0" else None
if r18 is None and (D / "r18_e17_cells.csv.gz").exists():
    pass                                            # ResNet arms of other teachers are covered by layer_ladder.py
ts = pd.read_csv(D / "teacher_seeds_cells.csv.gz"); ts = ts[ts.fm == TEACHER]
tta_f = D / "teacher_tta_cells.csv.gz"
tt_ = pd.read_csv(tta_f) if tta_f.exists() else None
if tt_ is not None:
    tt_ = tt_[tt_.fm == TEACHER]
KD, TGA, CTRL = f"cnx_kd{sfx}", f"cnx_kd_tga{sfx}", "cnx_control"

def f1w(pred, y, ii, n, W):
    cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
    tp, fp, fn = (W @ cnt).T; return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)

S, T = {}, {}                                       # (arm, method) -> [per-fold (B+1,)],  teacher variant -> [per-fold (B+1,)]
for fi, h in enumerate(FOLDS):
    g0 = cnx[cnx.holdout == h]
    ref = g0[(g0.arm == KD) & (g0.seed == 0)].sort_values("row")
    rows = ref.row.values; y = (ref.y == 1).values; half = ref.half.values
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    gb = np.random.default_rng([20261201, fi])
    W = np.vstack([np.ones(n), gb.multinomial(n, np.full(n, 1 / n), size=B)])
    sd = np.vstack([np.arange(10), gb.integers(0, 10, (B, 10))])
    t = te[te.holdout == h].set_index("row").loc[rows]; pt = t.p_teacher.values; thr_t = t.thr_teacher.values[0]; zt = lg(np.array([thr_t]))[0]
    T.setdefault("seed0", []).append(f1w(pt >= thr_t, y, ii, n, W))
    orc = np.zeros(len(rows), bool)
    for fit, to in (("A", "B"), ("B", "A")):
        orc[half == to] = pt[half == to] >= best_cut(pt[half == fit], y[half == fit])
    T.setdefault("oracle", []).append(f1w(orc, y, ii, n, W))
    for s_ in sorted(ts.seed.unique()):
        q = ts[(ts.holdout == h) & (ts.seed == s_)].set_index("row").loc[rows]
        T.setdefault(f"lora_seed{s_}", []).append(f1w(q.p_teacher.values >= q.thr_teacher.values[0], y, ii, n, W))
    if tt_ is not None and len(tt_):
        q = tt_[tt_.holdout == h].set_index("row").loc[rows]
        T.setdefault("tta", []).append(f1w(q.p_tta.values >= q.thr_tta.values[0], y, ii, n, W))
    groups = [(a, g0[g0.arm == a]) for a in (CTRL, KD, TGA)]
    if r18 is not None:
        groups += [(a, r18[(r18.holdout == h) & (r18.arm == a)]) for a in ("r18_kd_tga",)]
    for arm, ga in groups:
        F = {}
        for s_, g in ga.groupby("seed"):
            g = g.sort_values("row"); assert np.array_equal(g.row.values, rows), (h, arm, s_)
            p = g.p_none_single.values; z = lg(p)
            tgr = np.zeros(len(g), bool); so = np.zeros(len(g), bool)
            for fit, to in (("A", "B"), ("B", "A")):
                fi_ = half == fit; pi = half == to
                tgr[pi] = z[pi] + fit_bias(z[fi_], pt[fi_]) >= zt; so[pi] = p[pi] >= best_cut(p[fi_], y[fi_])
            for m, pred in (("source", p >= g.thr_single.values), ("tta", g.p_none_tta.values >= g.thr_tta.values), ("tgr", tgr), ("oracle", so)):
                F.setdefault(m, []).append(f1w(pred, y, ii, n, W))
        for m, lst in F.items():
            S.setdefault((arm, m), []).append(np.take_along_axis(np.stack(lst, 1), sd, 1).mean(1))
S = {k: np.mean(v, 0) for k, v in S.items()}; T = {k: np.mean(v, 0) for k, v in T.items()}

ci = lambda d: (d[0], np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5))
print(f"Teacher {TEACHER}: F1 seed 0 {T['seed0'][0]:.4f}; " + "; ".join(f"{k} {v[0]:.4f}" for k, v in T.items() if k != "seed0"))
print("Students: " + "; ".join(f"{a} {m} {S[(a, m)][0]:.4f}" for a, m in ((CTRL, "source"), (KD, "source"), (KD, "tgr"), (TGA, "source"), (TGA, "tta"), (TGA, "oracle"))))
print()
rows_o = []
fam = [("a" if TEACHER == "hopt0" else "d", f"{TGA} (source cutoff)", S[(TGA, "source")]),
       ("b" if TEACHER == "hopt0" else "e", f"{KD} + TGR", S[(KD, "tgr")])]
ps = []
for tag, name, s in fam:
    d = s - T["seed0"]; est, lo, hi = ci(d); p = ((d[1:] <= -DELTA).sum() + 1) / (B + 1); ps.append(p)
    rows_o.append(dict(test=f"N{'1' if TEACHER == 'hopt0' else '2'}{tag}", contrast=f"{name} minus teacher", diff=est, lo=lo, hi=hi, p_one_sided=p))
for r, a in zip(rows_o, holm(ps)):
    r["holm"] = a; r["non_inferior"] = bool(r["lo"] > -DELTA and a < 0.025)
if TEACHER != "hopt0":
    d = S[(TGA, "source")] - S[(KD, "source")]; est, lo, hi = ci(d); boot = d[1:]
    p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (B + 1))
    rows_o.insert(0, dict(test="N2c", contrast=f"{TGA} minus {KD} (superiority, two-sided)", diff=est, lo=lo, hi=hi, p_one_sided=np.nan, holm=p, non_inferior=np.nan))
out = pd.DataFrame(rows_o)
pd.set_option("display.width", 220)
print(f"PRE-SPECIFIED (margin {DELTA}):"); print(out.round(4).to_string(index=False)); print()

desc = []
def add(name, d):
    est, lo, hi = ci(d); desc.append(dict(quantity=name, value=est, lo=lo, hi=hi))
C = S[(CTRL, "source")]
for name, s in ((f"{KD} source", S[(KD, "source")]), (f"{KD} + TGR", S[(KD, "tgr")]), (f"{TGA} source", S[(TGA, "source")])):
    add(f"share of teacher's advantage recovered: {name}", (s - C) / (T["seed0"] - C))
for k in T:
    if k.startswith("lora_seed"):
        add(f"{TGA} source minus teacher {k}", S[(TGA, "source")] - T[k])
ks = [k for k in T if k.startswith("lora_seed")]
if ks:
    add(f"{TGA} source minus mean of teacher LoRA seeds", S[(TGA, "source")] - np.mean([T[k] for k in ks], 0))
add(f"same treatment, labelled cutoff: {TGA} oracle minus teacher oracle", S[(TGA, "oracle")] - T["oracle"])
if "tta" in T:
    add(f"same treatment, 8-view TTA: {TGA} TTA minus teacher TTA", S[(TGA, "tta")] - T["tta"])
    add("teacher TTA minus teacher single view", T["tta"] - T["seed0"])
add(f"{KD} source minus {CTRL} source", S[(KD, "source")] - C)
add(f"post hoc: {TGA} source minus {KD} + TGR", S[(TGA, "source")] - S[(KD, "tgr")])
if ("r18_kd_tga", "source") in S:
    add("r18_kd_tga source minus teacher (interval seen before; not a test)", S[("r18_kd_tga", "source")] - T["seed0"])
desc = pd.DataFrame(desc)
print("Descriptive:"); print(desc.round(4).to_string(index=False))
pd.concat([out.assign(kind="pre-specified"), desc.assign(kind="descriptive")]).to_csv(D / f"noninferiority_{TEACHER}.csv", index=False)

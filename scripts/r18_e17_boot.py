#!/usr/bin/env python3
"""E1.7 statistics (MULTI_TEACHER_PROTOCOL.md): do the Optimus findings hold for Virchow2, UNI and UNI2-h teachers?

Inputs (data/pc_runs_2026-11-17/):
  r18_crossfit_cells.csv.gz      r18_control, r18_kd, r18_kd_tga (Optimus)
  r18_exp2b_cells.csv.gz         r18_free_tga (Optimus-free adaptation)
  r18_e17_<fm>_cells.csv.gz      r18_kd_<fm>, r18_kd_tga_<fm>   (fm in virchow2, uni, uni2h; missing ones are skipped)
  teacher_newscanner_cells_multi.csv.gz   each teacher alone (view 0, its own source-val cutoff)

Pre-specified family (Holm, paired bootstrap over images and seeds, B = 20000), per teacher T:
  1. KD_T vs no teacher   2. KD+TGA_T vs KD_T   3. KD+TGA_T vs Optimus-free adaptation
Everything else printed here is descriptive.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
ap = argparse.ArgumentParser()
ap.add_argument("-B", type=int, default=20000)
ap.add_argument("--out", default=str(D / "r18_e17_boot.csv"))
a = ap.parse_args()

FMS = [fm for fm in ("virchow2", "uni", "uni2h") if (D / f"r18_e17_{fm}_cells.csv.gz").exists()]
files = ["r18_crossfit_cells.csv.gz", "r18_exp2b_cells.csv.gz"] + [f"r18_e17_{fm}_cells.csv.gz" for fm in FMS]
c = pd.concat([pd.read_csv(D / f) for f in files], ignore_index=True)
C, K, T, FR = "r18_control", "r18_kd", "r18_kd_tga", "r18_free_tga"
ARMS = [C, K, T, FR] + [x for fm in FMS for x in (f"r18_kd_{fm}", f"r18_kd_tga_{fm}")]
c = c[c.arm.isin(ARMS)]
METHODS = ["none_single", "none_tta", "adabn_single", "adabn_tta"]
seeds = sorted(c.seed.unique()); nS = len(seeds); B = a.B
rng = np.random.default_rng(20261125)

F, n_img = {}, {}
for h, g in c.groupby("holdout"):
    imgs = np.sort(g.image_id.unique()); n = len(imgs); n_img[h] = n
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
    F[h] = {}
    for arm in ARMS:
        ga = g[g.arm == arm]; F[h][arm] = {}
        assert ga.seed.nunique() == nS, (h, arm, ga.seed.nunique())
        for meth in METHODS:
            per_seed = []
            for s in seeds:
                r = ga[ga.seed == s]
                pred = r[f"p_{meth}"].values >= r["thr_" + meth.split("_")[1]].values; yy = r.y.values == 1
                cnt = np.zeros((n, 3)); np.add.at(cnt, np.searchsorted(imgs, r.image_id.values), np.c_[pred & yy, pred & ~yy, ~pred & yy])
                tp, fp, fn = (W @ cnt).T
                per_seed.append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
            F[h][arm][meth] = np.stack(per_seed, 1)
folds = sorted(F)
draw = {h: {arm: np.vstack([np.arange(nS), rng.integers(0, nS, (B, nS))]) for arm in ARMS} for h in folds}

def series(cfg):
    arm, meth = cfg
    return np.mean([np.take_along_axis(F[h][arm][meth], draw[h][arm], 1).mean(1) for h in folds], 0)

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

rows = []
for fm in FMS:
    kd, tga = f"r18_kd_{fm}", f"r18_kd_tga_{fm}"
    for name, x, y_ in ((f"{fm}: KD vs no teacher", (kd, "none_single"), (C, "none_single")),
                        (f"{fm}: KD+TGA vs KD", (tga, "none_single"), (kd, "none_single")),
                        (f"{fm}: KD+TGA vs Optimus-free adaptation", (tga, "none_single"), (FR, "none_single"))):
        d = series(x) - series(y_); boot = d[1:]
        p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (B + 1))
        rows.append({"contrast": name, "diff": round(d[0], 4), "lo": round(np.percentile(boot, 2.5), 4),
                     "hi": round(np.percentile(boot, 97.5), 4), "p": p})
out = pd.DataFrame(rows)
if len(out):
    out["holm"] = holm(out.p.values)
    out.to_csv(a.out, index=False)

pd.set_option("display.width", 250)
print(f"{len(folds)} folds, images {n_img}, {nS} seeds, B={B}; teachers with results: {FMS}\n")
tab = pd.DataFrame({(arm, m): {**{h: F[h][arm][m][0].mean() for h in folds},
                               "mean": np.mean([F[h][arm][m][0].mean() for h in folds])}
                    for arm in ARMS for m in METHODS}).T
print(tab.round(3).to_string()); print()
if len(out):
    show = out.copy(); show["p"] = show.p.map(lambda v: f"{v:.4f}"); show["holm"] = show.holm.map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False)); print()

# descriptive: teacher alone, and the cutoff share of the gap
tf = D / "teacher_newscanner_cells_multi.csv.gz"
if tf.exists():
    t = pd.read_csv(tf); lab = c.drop_duplicates(["holdout", "row"])[["holdout", "row", "y"]]
    t = t.merge(lab, on=["holdout", "row"])
    def auroc(p, y):
        r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    def f1(pred, y):
        tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum(); return 2 * tp / max(2 * tp + fp + fn, 1)
    def oracle(p, y):
        o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return (2 * tp / (k + y.sum())).max()
    rows = []
    for (fm, h), g in t.groupby(["fm", "holdout"]):
        y = (g.y == 1).values; p = g.p_teacher.values
        rows.append(dict(fm=fm, holdout=h, f1=f1(p >= g.thr_teacher.values, y), f1_oracle_cutoff=oracle(p, y), auroc=auroc(p, y)))
    tt = pd.DataFrame(rows).groupby("fm").mean(numeric_only=True)
    print("Teachers alone (all 50 new-scanner images, one seed):"); print(tt.round(3).to_string()); print()
    rows = []
    for (arm, h, s), g in c.groupby(["arm", "holdout", "seed"]):
        y = (g.y == 1).values; p = g.p_none_single.values
        rows.append(dict(arm=arm, f1=f1(p >= g.thr_single.values, y), f1_oracle_cutoff=oracle(p, y), auroc=auroc(p, y)))
    print("Students, single view (mean over folds and seeds):")
    print(pd.DataFrame(rows).groupby("arm").mean(numeric_only=True).loc[ARMS].round(3).to_string())

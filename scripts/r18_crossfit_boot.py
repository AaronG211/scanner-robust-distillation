#!/usr/bin/env python3
"""Paired bootstrap for the ResNet-18 arm x AdaBN x TTA table on cross-fitted predictions (all 50 images per fold).

Input cells file: one row per fold x arm x seed x cell with p_{none,adabn}_{single,tta}, thr_{single,tta}, y, image_id
(r18_crossfit_eval.py on the PC; every prediction comes from a model that never saw that cell's image).

Two bootstraps, both paired across arms and methods (the same image draw for everything in a fold):
  images        resample the fold's images; F1 per run -> mean over seeds -> mean over folds.
  images+seeds  additionally resample the seeds of each arm (one draw per arm and fold, shared by that arm's methods),
                so run-to-run training noise is inside the interval too.
Reported per contrast: difference, 95% percentile CI, two-sided bootstrap p, Holm-adjusted p over the listed family.

Usage: python scripts/r18_crossfit_boot.py [--exp2 | --exp2b] [--cells FILE] [--tga-arm r18_kd_tga] [--official-only] [--out FILE]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
ap = argparse.ArgumentParser()
ap.add_argument("--cells", nargs="+", default=[str(D / "r18_crossfit_cells.csv.gz")])
ap.add_argument("--exp2", action="store_true",
                help="control experiment: who answers on the new scanner (adds r18_exp2_cells.csv.gz, other contrasts)")
ap.add_argument("--exp2b", action="store_true",
                help="fully Optimus-free adaptation (adds r18_exp2_cells.csv.gz and r18_exp2b_cells.csv.gz, other contrasts)")
ap.add_argument("--tga-arm", default="r18_kd_tga")
ap.add_argument("--official-only", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("-B", type=int, default=20000)
a = ap.parse_args()

C, K, T = "r18_control", "r18_kd", a.tga_arm
P, R = "r18_kd_tgasoft", "r18_kd_tgar18"          # Optimus answers only / ResNet-18 ensemble answers, on new-scanner cells
FR = "r18_free_tga"                               # labels on training scanners + ResNet-18 ensemble answers on the new scanner: no Optimus
assert not (a.exp2 and a.exp2b)
ARMS = [C, K, T] + ([P, R] if a.exp2 or a.exp2b else []) + ([FR] if a.exp2b else [])
if a.exp2 or a.exp2b:
    a.cells = a.cells + [str(D / "r18_exp2_cells.csv.gz")]
if a.exp2b:
    a.cells = a.cells + [str(D / "r18_exp2b_cells.csv.gz")]
a.out = a.out or str(D / ("r18_exp2b_boot.csv" if a.exp2b else "r18_exp2_boot.csv" if a.exp2 else "r18_crossfit_boot.csv"))
METHODS = ["none_single", "none_tta", "adabn_single", "adabn_tta"]
c = pd.concat([pd.read_csv(f) for f in a.cells], ignore_index=True)
if a.official_only:
    c = c[c.official_test]
seeds = sorted(c.seed.unique()); nS = len(seeds)
rng = np.random.default_rng(20261117)
B = a.B

# F[fold][arm][method] -> (B+1, nS) F1 of every seed under every image draw (row 0 = the original sample)
F, n_img = {}, {}
for h, g in c.groupby("holdout"):
    imgs = np.sort(g.image_id.unique()); n = len(imgs); n_img[h] = n
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
    F[h] = {}
    for arm in ARMS:
        F[h][arm] = {}
        ga = g[g.arm == arm]
        for meth in METHODS:
            per_seed = []
            for s in seeds:
                r = ga[ga.seed == s]
                pred = r[f"p_{meth}"].values >= r["thr_" + meth.split("_")[1]].values; yy = r.y.values == 1
                cnt = np.zeros((n, 3))
                np.add.at(cnt, np.searchsorted(imgs, r.image_id.values), np.c_[pred & yy, pred & ~yy, ~pred & yy])
                tp, fp, fn = (W @ cnt).T
                per_seed.append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
            F[h][arm][meth] = np.stack(per_seed, 1)
folds = sorted(F)
# seed draws: one per replicate, arm and fold; row 0 keeps the original seeds
draw = {h: {arm: np.vstack([np.arange(nS), rng.integers(0, nS, (B, nS))]) for arm in ARMS} for h in folds}

def series(cfg, with_seeds):
    arm, meth = cfg; out = []
    for h in folds:
        x = F[h][arm][meth]
        out.append(np.take_along_axis(x, draw[h][arm], 1).mean(1) if with_seeds else x.mean(1))
    return np.mean(out, 0)

def stat(d):
    est, boot = d[0], d[1:]
    p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (len(boot) + 1))
    return est, np.percentile(boot, 2.5), np.percentile(boot, 97.5), p

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

CONTRASTS = [
    ("KD vs control", (K, "none_single"), (C, "none_single")),
    ("AdaBN on KD student", (K, "adabn_single"), (K, "none_single")),
    ("TGA on KD student", (T, "none_single"), (K, "none_single")),
    ("TGA vs AdaBN (same unlabeled data)", (T, "none_single"), (K, "adabn_single")),
    ("AdaBN on top of TGA", (T, "adabn_single"), (T, "none_single")),
    ("TTA on TGA student", (T, "none_tta"), (T, "none_single")),
    ("TTA on KD student (no adaptation)", (K, "none_tta"), (K, "none_single")),
    ("KD+TGA+TTA vs teacher-free AdaBN+TTA", (T, "none_tta"), (C, "adabn_tta")),
    ("KD+TGA+TTA vs KD+AdaBN+TTA", (T, "none_tta"), (K, "adabn_tta")),
]
if a.exp2:
    CONTRASTS = [
        ("Optimus vs ResNet-ensemble answers on new scanner", (P, "none_single"), (R, "none_single")),
        ("same, with TTA", (P, "none_tta"), (R, "none_tta")),
        ("ResNet-ensemble TGA vs KD (no adaptation)", (R, "none_single"), (K, "none_single")),
        ("Optimus answers-only TGA vs KD (no adaptation)", (P, "none_single"), (K, "none_single")),
        ("copying Optimus description on new-scanner cells", (T, "none_single"), (P, "none_single")),
        ("ResNet-ensemble TGA vs AdaBN on KD student", (R, "none_single"), (K, "adabn_single")),
    ]
if a.exp2b:                                       # fixed on 11/17 while the runs were still training
    CONTRASTS = [
        ("Optimus-free adaptation vs the team's Optimus TGA", (FR, "none_single"), (T, "none_single")),
        ("same, with TTA", (FR, "none_tta"), (T, "none_tta")),
        ("labels vs Optimus on the training scanners (ResNet answers on the new one)", (FR, "none_single"), (R, "none_single")),
        ("Optimus-free adaptation vs teacher-free control", (FR, "none_single"), (C, "none_single")),
        ("Optimus-free adaptation vs AdaBN on the control", (FR, "none_single"), (C, "adabn_single")),
        ("Optimus-free adaptation + TTA vs control + AdaBN + TTA", (FR, "none_tta"), (C, "adabn_tta")),
    ]
rows = []
for name, x, y_ in CONTRASTS:
    row = {"contrast": name}
    for label, ws in (("img", False), ("imgseed", True)):
        est, lo, hi, p = stat(series(x, ws) - series(y_, ws))
        row.update({"diff": round(est, 4), f"lo_{label}": round(lo, 4), f"hi_{label}": round(hi, 4), f"p_{label}": p})
    rows.append(row)
out = pd.DataFrame(rows)
for label in ("img", "imgseed"):
    out[f"holm_{label}"] = holm(out[f"p_{label}"].values)
out.to_csv(a.out, index=False)

print(f"{len(folds)} folds, images per fold {n_img}, {nS} seeds, B={B}" + ("  [official test images only]" if a.official_only else ""))
tab = pd.DataFrame({(arm, m): {**{h: F[h][arm][m][0].mean() for h in folds},
                               "mean": np.mean([F[h][arm][m][0].mean() for h in folds]),
                               "seed_sd": np.std(np.mean([F[h][arm][m][0] for h in folds], 0), ddof=1)}
                    for arm in ARMS for m in METHODS}).T
print(tab.round(3).to_string()); print()
pd.set_option("display.width", 250)
show = out.copy()
for k in [k for k in show if k.startswith(("p_", "holm_"))]:
    show[k] = show[k].map(lambda v: f"{v:.4f}")
print(show.to_string(index=False))

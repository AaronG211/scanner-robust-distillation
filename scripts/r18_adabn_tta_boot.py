#!/usr/bin/env python3
"""Paired image-level bootstrap for the ResNet-18 AdaBN x TTA x arm table (official new-scanner test images).

Each replicate resamples the 11 official test images of every fold with replacement (the same draw for every
arm / method / seed, so contrasts are paired). F1 uses each run's fixed validation cutoff. Per replicate:
F1 per run -> mean over 3 seeds -> mean over 3 folds. Reports the point estimate, 95% percentile CI and the
two-sided bootstrap p-value (2 x share of replicates on the other side of zero) for each contrast.

Input:  data/pc_runs_2026-11-17/r18_adabn_tta_cells.csv.gz  (from r18_adabn_tta.py on the PC)
Output: data/pc_runs_2026-11-17/r18_adabn_tta_boot.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 5000
METHODS = ["none_single", "none_tta", "adabn_single", "adabn_tta"]
c = pd.read_csv(D / "r18_adabn_tta_cells.csv.gz")
c = c[c.official_test]
configs = [(a, m) for a in ["r18_control", "r18_kd", "r18_kd_target"] for m in METHODS]
rng = np.random.default_rng(20261117)

fold_f1 = []                                                     # per fold: (B+1, n_configs) seed-averaged F1
for h, g in c.groupby("holdout"):
    imgs = np.sort(g.image_id.unique()); n = len(imgs)
    W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])   # row 0 = original sample
    per_cfg = []
    for arm, meth in configs:
        f1s = []
        for seed in range(3):
            r = g[(g.arm == arm) & (g.seed == seed)]
            thr = r["thr_tta" if meth.endswith("tta") else "thr_single"].iloc[0]
            pred = r[f"p_{meth}"].values >= thr; yy = r.y.values == 1
            ii = np.searchsorted(imgs, r.image_id.values)
            cnt = np.zeros((n, 3))
            np.add.at(cnt, ii, np.c_[pred & yy, pred & ~yy, ~pred & yy])
            tp, fp, fn = (W @ cnt).T
            f1s.append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
        per_cfg.append(np.mean(f1s, 0))
    fold_f1.append(np.stack(per_cfg, 1))
F = np.mean(fold_f1, 0)                                          # (B+1, n_configs), mean over folds
col = {cfg: i for i, cfg in enumerate(configs)}

def contrast(name, a, b):
    d = F[:, col[a]] - F[:, col[b]]; est, boot = d[0], d[1:]
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"contrast": name, "a": f"{a[0]}/{a[1]}", "b": f"{b[0]}/{b[1]}", "diff": round(est, 4),
            "ci_lo": round(np.percentile(boot, 2.5), 4), "ci_hi": round(np.percentile(boot, 97.5), 4), "p_boot": round(p, 4)}

K, T, C = "r18_kd", "r18_kd_target", "r18_control"
rows = [
    contrast("KD vs control (single)", (K, "none_single"), (C, "none_single")),
    contrast("AdaBN on KD student", (K, "adabn_single"), (K, "none_single")),
    contrast("TGA on KD student", (T, "none_single"), (K, "none_single")),
    contrast("TGA vs AdaBN (same unlabeled data)", (T, "none_single"), (K, "adabn_single")),
    contrast("AdaBN on top of TGA", (T, "adabn_single"), (T, "none_single")),
    contrast("TTA on TGA student", (T, "none_tta"), (T, "none_single")),
    contrast("TTA on KD student (no adaptation)", (K, "none_tta"), (K, "none_single")),
    contrast("best (KD+TGA+TTA) vs teacher-free AdaBN+TTA", (T, "none_tta"), (C, "adabn_tta")),
    contrast("best (KD+TGA+TTA) vs KD+AdaBN+TTA", (T, "none_tta"), (K, "adabn_tta")),
]
out = pd.DataFrame(rows); out.to_csv(D / "r18_adabn_tta_boot.csv", index=False)
print("point estimates (mean over folds and seeds):")
print(pd.Series(F[0], index=[f"{a}/{m}" for a, m in configs]).round(3).to_string()); print()
print(out.to_string(index=False))

"""Copy of r18_crossfit_eval.py that also finds ConvNeXt-T students (student_{h}_{tag}_s{seed}.pt; no BatchNorm, so the
AdaBN columns equal the plain ones). Cross-fitted evaluation of the ResNet-18 students on ALL 50 images of the new scanner (10 seeds).

Halves A/B of the new scanner's images come from crossfit.target_halves. For every cell the prediction comes from
a model that never saw the cell's image in any form:
  r18_control, r18_kd   one model per seed; it never saw the new scanner at all.
  r18_kd_tga*           two models per seed: the one adapted on half A (teacher-guided, unlabeled) scores half B,
                        the one adapted on half B scores half A.
  AdaBN (any arm)       BatchNorm statistics re-estimated on the fit half (unlabeled), predictions on the other half.
                        For r18_kd_tga the fit half is the half that model was adapted on, so data access is equal.
Cutoffs: training scanners' validation set, the model's original BatchNorm statistics, separately for single-view
and 8-view TTA. No new-scanner label is used anywhere.

Usage: python r18_crossfit_eval.py [--arms ...] [--out r18_crossfit]
       ->  results/<out>_cells.csv.gz (one row per fold x arm x seed x cell)
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from train import make_splits, best_threshold, prf, SCANNERS
from eval_extras import probs, adapt_bn
from probe_check import load
from crossfit import target_halves

HERE = Path(__file__).resolve().parent
DIST = Path("C:/data/midog/distill")
SEEDS = range(10)
CF = ("r18_kd_tga", "r18_kd_tgasoft", "r18_kd_tgar18", "r18_free_tga")
is_cf = lambda arm: "tga" in arm                        # every arm adapted on one half has "tga" in its tag (CF and new teachers)     # arms adapted on one half: weights are <arm>A / <arm>B
ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["r18_control", "r18_kd", "r18_kd_tga"])
ap.add_argument("--out", default="r18_crossfit", help="results/<out>_cells.csv.gz and results/<out>_summary.csv")
a = ap.parse_args()
VIEWS = (("single", False), ("tta", True))

d = dict(np.load(HERE / "cache" / "patches.npz")); X, y = d["X"], d["y"].astype(int)
out = []
for h in SCANNERS:
    S, _ = make_splits(d, h); unseen = S["test_unseen"]; H = target_halves(d, h)
    assert np.array_equal(np.sort(np.concatenate([H["A"], H["B"]])), unseen)
    pos = {k: np.searchsorted(unseen, v) for k, v in H.items()}
    yv = y[S["val"]]
    for seed in SEEDS:
        for arm in a.arms:
            P = {f"p_{bn}_{v}": np.full(len(unseen), np.nan) for bn in ("none", "adabn") for v, _ in VIEWS}
            T = {f"thr_{v}": np.full(len(unseen), np.nan) for v, _ in VIEWS}
            m = thr = None
            for fit, pred in (("A", "B"), ("B", "A")):
                if is_cf(arm) or m is None:
                    tag = arm + fit if is_cf(arm) else arm
                    m = load(next(f for f in (DIST / f"student_{h}_resnet18_{tag}_s{seed}.pt", DIST / f"student_{h}_{tag}_s{seed}.pt") if f.exists()))
                    thr = {v: best_threshold(probs(m, X, S["val"], tta), yv) for v, tta in VIEWS}
                ma = adapt_bn(m, X, H[fit])
                for v, tta in VIEWS:
                    T[f"thr_{v}"][pos[pred]] = thr[v]
                    P[f"p_none_{v}"][pos[pred]] = probs(m, X, H[pred], tta)
                    P[f"p_adabn_{v}"][pos[pred]] = probs(ma, X, H[pred], tta)
            df = pd.DataFrame({"holdout": h, "arm": arm, "seed": seed, "row": unseen, "ann_id": d["ann_id"][unseen],
                               "image_id": d["image_id"][unseen], "y": y[unseen],
                               "half": np.where(np.isin(unseen, H["A"]), "A", "B"),
                               "official_test": np.isin(unseen, S["test_unseen_official"]), **P, **T})
            assert not df.isna().any().any()
            out.append(df)
            print(h, arm, seed, {k[2:]: round(prf(df[k].values >= df["thr_" + k.split("_")[-1]].values, df.y.values, 0.5)[2], 3)
                                 for k in P}, flush=True)
c = pd.concat(out); c.to_csv(HERE / "results" / f"{a.out}_cells.csv.gz", index=False)
rows = []
for (h, arm, seed), g in c.groupby(["holdout", "arm", "seed"]):
    for k in [k for k in g.columns if k.startswith("p_")]:
        rows.append({"holdout": h, "arm": arm, "seed": seed, "method": k[2:],
                     "f1": prf(g[k].values >= g["thr_" + k.split("_")[-1]].values, g.y.values, 0.5)[2]})
s = pd.DataFrame(rows); s.to_csv(HERE / "results" / f"{a.out}_summary.csv", index=False)
print("\nNew-scanner F1 on all 50 images per fold, mean over 10 seeds:")
print(s.groupby(["arm", "method", "holdout"]).f1.mean().unstack().assign(mean3=lambda t: t.mean(1)).round(3).to_string())

"""AdaBN x TTA on the ResNet-18 distillation students (control / KD / KD + teacher-guided adaptation).

AdaBN uses exactly the unlabeled data that teacher-guided adaptation (--target) uses: the new scanner's
'train'-split images. Scoring is on test_unseen_official (the new scanner's 'test' images), which neither
AdaBN nor the --target student ever saw. So all four cells of the AdaBN x TGA table get the same data access.

Cutoffs come from the training scanners' validation set with the model's original BatchNorm statistics,
separately for single-view and TTA (8 rotations/flips) prediction. No new-scanner label is used anywhere.

Check: the 'none_single' row should reproduce (up to fp16-vs-fp32 rounding) distill.py's test_unseen_official_f1 for every run.

Usage: python r18_adabn_tta.py
  -> results/r18_adabn_tta_summary.csv  (one row per fold x arm x seed x method)
  -> results/r18_adabn_tta_cells.csv.gz (per-cell probabilities on all new-scanner cells, with flags)
"""
from pathlib import Path
import numpy as np, pandas as pd
from train import make_splits, best_threshold, prf, auroc, SCANNERS
from eval_extras import probs, adapt_bn
from probe_check import load

HERE = Path(__file__).resolve().parent
DIST = Path("C:/data/midog/distill")
ARMS = ["r18_control", "r18_kd", "r18_kd_target"]

d = dict(np.load(HERE / "cache" / "patches.npz")); X, y = d["X"], d["y"].astype(int)
ref = pd.read_csv(HERE / "results" / "resnet18_distill.csv")
summary, cells, check = [], [], []   # check: fp32 here vs fp16 in distill.py
for h in SCANNERS:
    S, _ = make_splits(d, h)
    unseen = S["test_unseen"]
    adapt_idx = unseen[d["split"][unseen] == "train"]            # unlabeled; same cells as --target uses
    official = S["test_unseen_official"]
    assert not np.intersect1d(adapt_idx, official).size
    for arm in ARMS:
        for seed in range(3):
            m = load(DIST / f"student_{h}_resnet18_{arm}_s{seed}.pt")
            thr = {tta: best_threshold(probs(m, X, S["val"], tta), y[S["val"]]) for tta in (False, True)}
            ma = adapt_bn(m, X, adapt_idx)
            P = {}
            for bn, model in (("none", m), ("adabn", ma)):
                for tta in (False, True):
                    meth = f"{bn}_{'tta' if tta else 'single'}"
                    P[meth] = probs(model, X, unseen, tta)
                    p = P[meth][np.isin(unseen, official)]; yy = y[official]
                    pr, rc, f1 = prf(p, yy, thr[tta])
                    summary.append({"holdout": h, "arm": arm, "seed": seed, "method": meth, "bn": bn, "tta": tta,
                                    "threshold": round(float(thr[tta]), 4), "n": len(official),
                                    "f1": round(f1, 4), "precision": round(pr, 4), "recall": round(rc, 4),
                                    "auroc": round(auroc(p, yy), 4)})
            r = ref[(ref.holdout == h) & (ref.tag == arm) & (ref.seed == seed)].iloc[0]
            got = [s for s in summary if s["holdout"] == h and s["arm"] == arm and s["seed"] == seed
                   and s["method"] == "none_single"][0]["f1"]
            check.append({"holdout": h, "arm": arm, "seed": seed, "here": got, "distill_py": r.test_unseen_official_f1})
            cells.append(pd.DataFrame({"holdout": h, "arm": arm, "seed": seed, "row": unseen,
                                       "ann_id": d["ann_id"][unseen], "image_id": d["image_id"][unseen],
                                       "y": y[unseen], "official_test": np.isin(unseen, official),
                                       "seen_unlabeled_by_adabn_and_tga": np.isin(unseen, adapt_idx),
                                       **{f"p_{k}": v for k, v in P.items()},
                                       "thr_single": thr[False], "thr_tta": thr[True]}))
            print(h, arm, seed, {s["method"]: s["f1"] for s in summary[-4:]}, flush=True)
pd.DataFrame(summary).to_csv(HERE / "results" / "r18_adabn_tta_summary.csv", index=False)
pd.concat(cells).to_csv(HERE / "results" / "r18_adabn_tta_cells.csv.gz", index=False)
s = pd.DataFrame(summary); c = pd.DataFrame(check)
print("reproduction check (none_single vs distill.py): max |diff| =", round((c.here - c.distill_py).abs().max(), 4),
      "; mean diff =", round((c.here - c.distill_py).mean(), 4))
print("\nOfficial new-scanner test F1, mean over 3 seeds:")
print(s.groupby(["arm", "method", "holdout"]).f1.mean().unstack().assign(mean3=lambda t: t.mean(1)).round(3).to_string())

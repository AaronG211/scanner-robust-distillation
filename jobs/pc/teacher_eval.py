"""Each foundation-model teacher on ALL new-scanner cells (it never adapts, so no cross-fit is needed).
View 0 of the targetall cache (no rotation, no stain jitter); cutoff = the teacher's own source-validation cutoff.
Writes results/teacher_newscanner_cells_multi.csv.gz and prints F1 / AUROC per teacher and fold."""
import sys; sys.path.insert(0, r"C:\midog\repo\axel_model")
from pathlib import Path
import numpy as np, pandas as pd
from train import make_splits, prf, auroc
A = Path(r"C:\midog\repo\axel_model"); DIST = Path(r"C:\data\midog\distill")
d = dict(np.load(A / "cache" / "patches.npz")); y = d["y"].astype(int)
cells, rows = [], []
for fm in ("hopt0", "virchow2", "uni", "uni2h"):
    for h in ("CS2", "S360", "XR"):
        f = DIST / f"teacher_{'' if fm == 'hopt0' else fm + '_'}{h}_targetall_s0.npz"
        if not f.exists():
            print("missing", f.name); continue
        S, _ = make_splits(d, h); t = np.load(f); assert np.array_equal(t["idx"], S["test_unseen"])
        p = 1 / (1 + np.exp(-t["logit"][:, 0])); thr = float(t["threshold"]); off = np.isin(t["idx"], S["test_unseen_official"])
        cells.append(pd.DataFrame({"fm": fm, "holdout": h, "row": t["idx"], "p_teacher": p, "thr_teacher": thr}))
        for name, m in (("all50", np.ones(len(p), bool)), ("official11", off)):
            pr, rc, f1 = prf(p[m], y[t["idx"]][m], thr)
            rows.append(dict(fm=fm, holdout=h, set=name, f1=f1, precision=pr, recall=rc, auroc=auroc(p[m], y[t["idx"]][m]), cutoff=thr))
pd.concat(cells).to_csv(A / "results" / "teacher_newscanner_cells_multi.csv.gz", index=False)
r = pd.DataFrame(rows); r.to_csv(A / "results" / "teacher_newscanner_summary_multi.csv", index=False)
print(r.pivot_table(index=["set", "fm"], columns="holdout", values="f1").assign(mean=lambda x: x.mean(1)).round(3).to_string())
print(r[r.set == "all50"].groupby("fm")[["auroc", "precision", "recall"]].mean().round(3).to_string())

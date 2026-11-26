"""Exploratory (not pre-registered): does a 3-seed ensemble help Axel's students, as it helped our ResNet?

For each student model and held-out scanner: average the probability of each cell over the 3 seeds (val and new-scanner
test cells), choose the cutoff with his best_threshold on the averaged val probabilities, and score F1 on the new
scanner's official test images. Compared with the mean single-seed F1 (each seed with its own cutoff, as reported).
Paired bootstrap over test cases within fold (B = 2000, rng 2030); the statistic is the mean over folds.
Writes results/axel_ensemble.csv.
"""
import numpy as np, pandas as pd
from cutoff_transfer import ROOT, FOLDS, meta, best_threshold

MODELS = ["student_final", "student_final_conf", "student_final_grl", "student_final_target"]
B = 2000
rng = np.random.default_rng(2030)


def f1w(p, y, t, w):
    pred = p >= t
    tp = np.sum(w * (pred & (y == 1))); fp = np.sum(w * (pred & (y == 0))); fn = np.sum(w * (~pred & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


img = meta().set_index("ann_id").image_id
cases = {}
rows, boots = [], {}
for name in MODELS:
    d = pd.read_csv(ROOT / "data" / "axel_predictions" / f"{name}.csv")
    for f in FOLDS:
        x = d[d.holdout == f]
        pv = x[x.set == "val"].pivot(index="ann_id", columns="seed", values="prob")
        yv = x[x.set == "val"].groupby("ann_id").label.first().loc[pv.index].values
        pt = x[x.set == "new_scanner_test"].pivot(index="ann_id", columns="seed", values="prob")
        yt = x[x.set == "new_scanner_test"].groupby("ann_id").label.first().loc[pt.index].values
        ids = img.loc[pt.index].values
        if f not in cases:
            u = np.unique(ids); cases[f] = (u, rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float))
        u, W = cases[f]; ci = np.searchsorted(u, ids)
        t_single = [best_threshold(pv[s].values, yv) for s in pv.columns]
        t_ens = best_threshold(pv.mean(1).values, yv)
        pe = pt.mean(1).values
        single = np.mean([f1w(pt[s].values, yt, t, np.ones(len(yt))) for s, t in zip(pt.columns, t_single)])
        ens = f1w(pe, yt, t_ens, np.ones(len(yt)))
        rows.append({"model": name, "fold": f, "single_mean": single, "ensemble": ens, "seeds": len(pt.columns)})
        bs = np.array([np.mean([f1w(pt[s].values, yt, t, W[b, ci]) for s, t in zip(pt.columns, t_single)]) for b in range(B)])
        be = np.array([f1w(pe, yt, t_ens, W[b, ci]) for b in range(B)])
        boots[(name, f)] = be - bs

r = pd.DataFrame(rows); r["gain"] = r.ensemble - r.single_mean
r.round(4).to_csv(ROOT / "results" / "axel_ensemble.csv", index=False)
print(r.round(3).to_string(index=False))
print("\nMean over folds (paired case bootstrap, 95% CI):")
for name in MODELS:
    d = np.mean([boots[(name, f)] for f in FOLDS], axis=0)
    g = r[r.model == name]
    print(f"  {name:22s} single {g.single_mean.mean():.3f}  ensemble {g.ensemble.mean():.3f}  "
          f"gain {g.gain.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

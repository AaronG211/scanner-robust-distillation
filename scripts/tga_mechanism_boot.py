"""Exploratory (post hoc, 2026-11-16): paired bootstrap CIs for the TGA mechanism check (tga_mechanism.py).

Axel's distilled ConvNeXt-T student (student_final) vs the same student with teacher-guided adaptation
(student_final_target), and vs the Optimus teacher, on the new scanner's official test images.
Per run: F1 at the source-val cutoff, oracle F1 (in-sample; re-chosen on every resample), AUROC.
Bootstrap: test images resampled within fold (shared by all models), student seeds resampled (shared);
statistic = mean over folds of the seed-weighted mean; B = 2000, rng 2031. No multiplicity adjustment.
Writes results/tga_mechanism_boot.csv.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cutoff_transfer import FOLDS, ROOT, meta, best_threshold
from cutoff_sensitivity import w_best_threshold

B = 2000
rng = np.random.default_rng(2031)
img = meta().set_index("ann_id").image_id


def f1w(p, y, t, w):
    pred = p >= t
    tp = np.sum(w * (pred & (y == 1))); fp = np.sum(w * (pred & (y == 0))); fn = np.sum(w * (~pred & (y == 1)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


models = {n: pd.read_csv(ROOT / "data" / "axel_predictions" / f"{n}.csv")
          for n in ["optimus_teacher", "student_final", "student_final_target"]}
seedw = rng.multinomial(3, np.full(3, 1 / 3), size=B) / 3.0
stats = {}                                    # (model, metric) -> (B,) mean over folds
point = {}
for f in FOLDS:
    ids = None
    per = {}
    for n, d in models.items():
        x = d[d.holdout == f]
        for s in sorted(x.seed.unique()):
            r = x[x.seed == s]; v = r[r.set == "val"]; t = r[r.set == "new_scanner_test"].sort_values("ann_id")
            per[(n, s)] = (t.prob.values, t.label.values.astype(int), best_threshold(v.prob.values, v.label.values.astype(int)),
                           img.loc[t.ann_id.values].values)
    u = np.unique(np.concatenate([v[3] for v in per.values()]))
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float)
    for n in models:
        seeds = sorted(s for (m, s) in per if m == n)
        vals = {k: np.empty((B, len(seeds))) for k in ("F1", "oracleF1", "AUROC")}
        pts = {k: [] for k in vals}
        for j, s in enumerate(seeds):
            p, y, t, ii = per[(n, s)]; ci = np.searchsorted(u, ii)
            one = np.ones(len(y))
            pts["F1"].append(f1w(p, y, t, one)); pts["oracleF1"].append(f1w(p, y, w_best_threshold(p, y, one), one))
            pts["AUROC"].append(roc_auc_score(y, p))
            for b in range(B):
                w = W[b, ci]
                vals["F1"][b, j] = f1w(p, y, t, w)
                vals["oracleF1"][b, j] = f1w(p, y, w_best_threshold(p, y, w), w)
                vals["AUROC"][b, j] = roc_auc_score(y, p, sample_weight=w)
        sw = seedw[:, :len(seeds)] if len(seeds) == 3 else np.ones((B, 1))
        sw = sw / sw.sum(1, keepdims=True)
        for k in vals:
            stats.setdefault((n, k), np.zeros(B)); stats[(n, k)] += (sw * vals[k]).sum(1) / len(FOLDS)
            point.setdefault((n, k), 0.0); point[(n, k)] += np.mean(pts[k]) / len(FOLDS)
    print(f"{f} done", flush=True)

rows = []
for n in models:
    for k in ("F1", "oracleF1", "AUROC"):
        rows.append({"what": f"{n} {k}", "point": point[(n, k)], "ci_lo": np.percentile(stats[(n, k)], 2.5),
                     "ci_hi": np.percentile(stats[(n, k)], 97.5)})
    g = stats[(n, "oracleF1")] - stats[(n, "F1")]
    rows.append({"what": f"{n} cutoff gap (oracleF1 - F1)", "point": point[(n, "oracleF1")] - point[(n, "F1")],
                 "ci_lo": np.percentile(g, 2.5), "ci_hi": np.percentile(g, 97.5)})
for a, b_, lab in [("student_final_target", "student_final", "TGA - student"),
                   ("student_final", "optimus_teacher", "student - teacher")]:
    for k in ("F1", "oracleF1", "AUROC"):
        d = stats[(a, k)] - stats[(b_, k)]
        rows.append({"what": f"{lab} d{k}", "point": point[(a, k)] - point[(b_, k)],
                     "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "P_le0": (d <= 0).mean()})
    gd = (stats[(a, "oracleF1")] - stats[(a, "F1")]) - (stats[(b_, "oracleF1")] - stats[(b_, "F1")])
    rows.append({"what": f"{lab} d(cutoff gap)",
                 "point": (point[(a, "oracleF1")] - point[(a, "F1")]) - (point[(b_, "oracleF1")] - point[(b_, "F1")]),
                 "ci_lo": np.percentile(gd, 2.5), "ci_hi": np.percentile(gd, 97.5), "P_le0": (gd <= 0).mean()})
r = pd.DataFrame(rows); r.round(4).to_csv(ROOT / "results" / "tga_mechanism_boot.csv", index=False)
print(r.round(3).to_string(index=False))

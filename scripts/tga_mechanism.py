"""Exploratory (post hoc, 2026-11-16): does teacher-guided adaptation (TGA) improve ranking, or mainly the cutoff?
New scanner's official test images; F1 at the source-val cutoff, oracle F1 (in-sample, optimistic) and AUROC,
seeds averaged within fold, then folds averaged. Axel's teacher / student / TGA student vs our ResNet all
(as trained; AdaBN-cf + TTA). No CIs. Writes results/tga_mechanism.csv.
"""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from cutoff_transfer import best_threshold, prf, FOLDS, ROOT, meta

rows = []
for name in ["optimus_teacher", "student_final", "student_final_target"]:
    d = pd.read_csv(ROOT / "data/axel_predictions" / f"{name}.csv")
    for f in FOLDS:
        for s in sorted(d[d.holdout == f].seed.unique()):
            x = d[(d.holdout == f) & (d.seed == s)]; v = x[x.set == "val"]; t = x[x.set == "new_scanner_test"]
            thr = best_threshold(v.prob.values, v.label.values)
            rows.append(dict(model=name, fold=f, seed=s, F1=prf(t.prob.values, t.label.values, thr)[2],
                             oracleF1=prf(t.prob.values, t.label.values, best_threshold(t.prob.values, t.label.values))[2],
                             AUROC=roc_auc_score(t.label, t.prob)))
m = meta().set_index("ann_id")
for s in range(5):
    for f in FOLDS:
        p = pd.read_csv(ROOT / "results/preds" / f"{f}_all_seed{s}.csv"); t = p[p.set == "test_out_all"].sort_values("row")
        cf = pd.read_csv(ROOT / "results/preds_tta" / f"{f}_all_seed{s}_cf.csv").set_index("row").loc[t.row]
        thr = json.load(open(ROOT / "results" / f"{f}_all_seed{s}.json"))["threshold"]
        thr_tta = json.load(open(ROOT / "results/preds_tta" / f"{f}_all_seed{s}.json"))["thr_tta"]
        off = m.loc[t.ann_id, "split"].values == "test"; y = t.y.values[off]
        for mode, pp, th in [("resnet_all_plain", t.p.values[off], thr), ("resnet_all_adabncf_tta", cf.p_adabncf_tta.values[off], thr_tta)]:
            rows.append(dict(model=mode, fold=f, seed=s, F1=prf(pp, y, th)[2],
                             oracleF1=prf(pp, y, best_threshold(pp, y))[2], AUROC=roc_auc_score(y, pp)))
r = pd.DataFrame(rows); r.round(4).to_csv(ROOT / "results" / "tga_mechanism.csv", index=False)
g = r.groupby(["model", "fold"])[["F1", "oracleF1", "AUROC"]].mean().groupby("model").mean()
print(g.assign(cutoff_gap=g.oracleF1 - g.F1).round(3).to_string())

#!/usr/bin/env python3
"""Descriptive check (2026-11-25): why the label-shift estimators in recal_baselines.py fail. For each Optimus KD student,
fold and seed, the new-scanner prevalence estimated from ALL unlabelled new-scanner cells by EM (raw probabilities),
EM after bias-corrected temperature scaling, and BBSE, next to the true prevalence, the student's own positive rate at its
source cutoff, and the teacher's positive rate. Output: data/pc_runs_2026-11-17/prior_estimates_check.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
exec((HERE / "recal_baselines.py").read_text().split("cells = {")[0])      # reuse the method definitions unchanged
D = HERE.parent / "data" / "pc_runs_2026-11-17"
TE = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
rows = []
for st, f in (("r18_kd", "e0_cells_r18.csv.gz"), ("cnx_kd", "e0_cells_cnx.csv.gz")):
    c = pd.read_csv(D / f)
    for h in FOLDS:
        for s in range(10):
            g = c[(c.holdout == h) & (c.arm == st) & (c.seed == s)]; u = g[g.split == "unseen"].sort_values("row"); v = g[g.split == "val"]
            p = u.p_single.values; vp = v.p_single.values; vy = (v.y == 1).values; thr = u.thr_single.values[0]
            t = TE[TE.holdout == h].set_index("row").loc[u.row.values]; a, b = fit_bt(lg(vp), vy.astype(float))
            rows.append(dict(student=st, fold=h, seed=s, true=(u.y == 1).mean(), source_val_true=vy.mean(),
                             student_pos_rate=(p >= thr).mean(), em=em_prior(p, vp.mean()), em_bcts=em_prior(sg(a * lg(p) + b), vy.mean()),
                             bbse=bbse_prior(vp >= thr, vy, p >= thr), teacher_pos_rate=(t.p_teacher.values >= t.thr_teacher.values[0]).mean()))
r = pd.DataFrame(rows); r.to_csv(D / "prior_estimates_check.csv", index=False)
print(r.groupby(["student", "fold"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())

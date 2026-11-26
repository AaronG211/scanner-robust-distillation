"""EXPLORATORY (CUTOFF_PROTOCOL.md, Amendment 2): the cutoff methods in the TTA modes of our grid runs.

Source = source validation under TTA (results/preds_valtta, t_src = stored thr_tta). Target = the new scanner under
TTA (p_tta) or AdaBN-cf + TTA (p_adabncf_tta). Cross-fitted case halves; bootstrap as in the confirmatory script
(cases within folds, seeds resampled, cutoffs re-estimated), B = 1000, unadjusted CIs.
Writes results/cutoff_tta.csv.
"""
import json
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, RES, PT, meta, target_halves, evaluate
from cutoff_confirm import Run

GRID = {"baseline": "nonstain", "adv": "nonstain", "stain": "stain", "stain_adv": "stain", "all": "stain"}
METHODS = ["SRC", "PRM", "LMM", "EF1"]


def load(fold, tag, s, mode):
    v = pd.read_csv(RES / "preds_valtta" / f"{fold}_{tag}_seed{s}.csv")
    j = json.loads((PT / f"{fold}_{tag}_seed{s}.json").read_text())
    t = pd.read_csv(PT / f"{fold}_{tag}_seed{s}.csv")[["row", "image_id", "y", "p_tta"]]
    if mode == "adabncf_tta":
        cf = pd.read_csv(PT / f"{fold}_{tag}_seed{s}_cf.csv")[["row", "p_adabncf_tta"]]
        t = t.merge(cf, on="row", validate="one_to_one").drop(columns="p_tta").rename(columns={"p_adabncf_tta": "p_tta"})
    t = t.rename(columns={"p_tta": "p_plain"})
    m = meta()
    t["ann_id"] = m.ann_id.values[t.row.values]; t["split"] = m.split.values[t.row.values]
    t["halfA"] = target_halves(t.image_id.values)
    return {"src_p": v.p_val_tta.values.astype(float), "src_y": v.y.values.astype(int),
            "t_src": float(j["thr_tta"]), "tgt": t.reset_index(drop=True)}


def main(B=1000):
    rng = np.random.default_rng(2029); rows = []
    for mode in ("tta", "adabncf_tta"):
        runs = [Run("ours", f, tag, s, grp, load(f, tag, s, mode)) for f in FOLDS for tag, grp in GRID.items() for s in range(5)]
        cases = {}
        for f in FOLDS:
            ids = np.unique(np.concatenate([R.t.image_id.values for R in runs if R.fold == f]))
            cases[f] = (ids, rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B).astype(float))
        sw = rng.multinomial(5, np.full(5, 0.2), size=B) / 5.0
        vals = {}
        for i, R in enumerate(runs):
            ids, Wc = cases[R.fold]; cidx = np.searchsorted(ids, R.t.image_id.values)
            for m in METHODS:
                vals[(i, m)] = np.array([R.f1("plain", m, "crossfit", Wc[b, cidx]) for b in range(B)])
        for grp_name, grp in (("non-stain grid", "nonstain"), ("stain grid", "stain")):
            keys = sorted({(R.fold, R.name) for R in runs if R.group == grp})

            def agg(m):
                tot = 0
                for f, n in keys:
                    idx = [i for i, R in enumerate(runs) if R.fold == f and R.name == n]
                    w = sw[:, [runs[i].seed for i in idx]]; w = w / w.sum(1, keepdims=True)
                    tot = tot + sum(w[:, k] * vals[(i, m)] for k, i in enumerate(idx))
                return tot / len(keys)

            def pt(m):
                return np.mean([np.mean([evaluate({"src_p": runs[i].sp, "src_y": runs[i].sy, "t_src": runs[i].ts,
                                                   "tgt": runs[i].t}, "plain", m, "crossfit")[2]
                                         for i, R in enumerate(runs) if R.fold == f and R.name == n]) for f, n in keys])
            base = agg("SRC"); f_src = pt("SRC")
            for m in METHODS + ["ORACLE"]:
                r = {"subset": grp_name, "mode": mode, "method": m, "F1": pt(m)}
                if m in METHODS:
                    d = agg(m) - base
                    r.update({"dF1_vs_SRC": r["F1"] - f_src, "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)})
                rows.append(r)
    d = pd.DataFrame(rows)
    d.round(4).to_csv(ROOT / "results" / "cutoff_tta.csv", index=False)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

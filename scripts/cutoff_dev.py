"""DEVELOPMENT analysis (exploratory, before the protocol is frozen): label-free cutoff methods on phase-1 runs.

Phase-1 tags (3 seeds) are not part of the confirmatory set (phase-2 grid tags, 5 seeds; Axel's models).
Also checks that SRC reproduces the stored F1 of every run (sanity check of the loaders).
Writes results/cutoff_dev_runs.csv.
"""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, METHODS, load_ours, evaluate, ROOT

DEV_TAGS = ["macenko", "macenko_stain", "stain_ema", "stain_he", "stain_mix", "stain_s002", "stain_s010",
            "stain_scale02", "stain_soft"]


def main():
    rows = []
    for tag in DEV_TAGS:
        for f in FOLDS:
            for s in range(3):
                run = load_ours(f, tag, s)
                if run is None:
                    print("missing", tag, f, s); continue
                for mode in ("plain", "adabncf"):
                    for split in ("crossfit", "official"):
                        for m in METHODS + ["ORACLE"]:
                            r = evaluate(run, mode, m, split)
                            if r is None:
                                continue
                            rows.append({"tag": tag, "fold": f, "seed": s, "mode": mode, "split": split,
                                         "method": m, "precision": r[0], "recall": r[1], "f1": r[2], "cut": r[3]})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "cutoff_dev_runs.csv", index=False)
    pd.set_option("display.width", 200)
    for split in ("crossfit", "official"):
        x = d[d.split == split]
        g = x.groupby(["mode", "method"]).agg(f1=("f1", "mean"), prec=("precision", "mean"), rec=("recall", "mean"))
        src = x[x.method == "SRC"].set_index(["tag", "fold", "seed", "mode"]).f1
        ora = x[x.method == "ORACLE"].set_index(["tag", "fold", "seed", "mode"]).f1
        out = []
        for (mode, m), gg in x.groupby(["mode", "method"]):
            k = gg.set_index(["tag", "fold", "seed", "mode"]).f1
            dlt = k - src.loc[k.index]; gap = ora.loc[k.index] - src.loc[k.index]
            out.append({"mode": mode, "method": m, "dF1_mean": dlt.mean(), "share_runs_improved": (dlt > 0).mean(),
                        "gap_closed": dlt.sum() / gap.sum(), "worst_dF1": dlt.min()})
        g = g.join(pd.DataFrame(out).set_index(["mode", "method"]))
        print(f"\n=== split {split}: mean over {x.groupby(['tag','fold','seed']).ngroups} runs ===")
        print(g.round(3).to_string())
        print("per fold dF1 vs SRC:")
        pf = x.pivot_table(index=["mode", "method"], columns="fold", values="f1", aggfunc="mean")
        print((pf - pf.xs("SRC", level="method").reindex(pf.index, level="mode").values).round(3).to_string())


if __name__ == "__main__":
    main()

"""EXPLORATORY (CUTOFF_PROTOCOL.md): how much of AdaBN's F1 gain is a cutoff shift, and how much better ranking?

For each of our runs (cross-fitted halves, all 50 new-scanner cases):
  gain        = F1(adabncf, SRC) - F1(plain, SRC)
  ranking     = F1(adabncf, ORACLE) - F1(plain, ORACLE)        F1 gain at the best cutoff of each
  cutoff      = gain - ranking = cutoff loss of plain - cutoff loss of adabncf
                (cutoff loss = F1(ORACLE) - F1(SRC), what a perfect cutoff would add)
  PRM share   = share of plain's cutoff loss that PRM recovers without labels
plus AUROC (plain vs adabncf). ORACLE uses target labels and is in-sample optimistic: it is a reference, not a method.

Reads results/cutoff_point_runs.csv; writes results/cutoff_decomposition.csv.
"""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, load_ours

TAGS = ["baseline", "adv", "stain", "stain_adv", "all", "all_no_grl", "all_no_soft", "all_no_stain", "all_stain_default"]


def auroc(p, y):
    r = pd.Series(p).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    P = pd.read_csv(ROOT / "results" / "cutoff_point_runs.csv")
    P = P[(P.source == "ours") & (P.split == "crossfit")]
    W = P.pivot_table(index=["name", "fold", "seed"], columns=["mode", "method"], values="f1")
    rows = []
    for (tag, f, s), r in W.iterrows():
        run = load_ours(f, tag, s); t = run["tgt"]
        rows.append({"tag": tag, "fold": f, "seed": s,
                     "F1_plain_SRC": r[("plain", "SRC")], "F1_adabn_SRC": r[("adabncf", "SRC")],
                     "F1_plain_ORACLE": r[("plain", "ORACLE")], "F1_adabn_ORACLE": r[("adabncf", "ORACLE")],
                     "F1_plain_PRM": r[("plain", "PRM")],
                     "AUROC_plain": auroc(t.p_plain.values, t.y.values), "AUROC_adabn": auroc(t.p_adabncf.values, t.y.values)})
    d = pd.DataFrame(rows)
    d["gain"] = d.F1_adabn_SRC - d.F1_plain_SRC
    d["ranking"] = d.F1_adabn_ORACLE - d.F1_plain_ORACLE
    d["cutoff"] = d.gain - d.ranking
    d["cutoff_loss_plain"] = d.F1_plain_ORACLE - d.F1_plain_SRC
    d["cutoff_loss_adabn"] = d.F1_adabn_ORACLE - d.F1_adabn_SRC
    d["PRM_recovered"] = d.F1_plain_PRM - d.F1_plain_SRC
    d["dAUROC"] = d.AUROC_adabn - d.AUROC_plain
    g = d.groupby("tag")[["F1_plain_SRC", "F1_adabn_SRC", "gain", "ranking", "cutoff", "cutoff_loss_plain",
                          "cutoff_loss_adabn", "PRM_recovered", "AUROC_plain", "AUROC_adabn", "dAUROC"]].mean()
    g = g.loc[[t for t in TAGS if t in g.index]]
    g["cutoff_share_of_gain"] = g.cutoff / g.gain
    g["PRM_share_of_cutoff_loss"] = g.PRM_recovered / g.cutoff_loss_plain
    g.round(4).to_csv(ROOT / "results" / "cutoff_decomposition.csv")
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())
    print("\nper held-out scanner (grid tags):")
    grid = d[d.tag.isin(TAGS[:5])]
    print(grid.groupby(["fold"])[["gain", "ranking", "cutoff", "cutoff_loss_plain", "PRM_recovered", "dAUROC"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

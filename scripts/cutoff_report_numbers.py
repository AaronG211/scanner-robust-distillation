"""Numbers requested by the adversarial review of the cutoff report (all exploratory / post hoc unless noted).

1. Familiar-scanner oracle gap (no scanner change): our runs' test_in cells, per-case-half oracle as in crossfit.
2. Development extras that were run interactively and not saved: MixStyle (`mix`) and frozen Phikon probes, plain mode.
3. Axel's models weighted per model (seeds averaged within model, then mean over fold x model), as in H4:
   mechanism (PRM / PRM-op), audit (k = 20, 50), referral error by scanner.
4. Official split and precision / recall for our grid (secondary outcomes named in the protocol).
5. Referral: how many runs certify no cutoff.
Writes results/cutoff_report_numbers.txt (printed).
"""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, RES, load_ours, evaluate, best_threshold, prf, target_halves

out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))


def familiar_gap():
    rows = []
    for tag, grp in [("baseline", "nonstain"), ("adv", "nonstain"), ("stain", "stain"), ("stain_adv", "stain"), ("all", "stain")]:
        for f in FOLDS:
            for s in range(5):
                d = pd.read_csv(RES / "preds" / f"{f}_{tag}_seed{s}.csv"); fam = d[d.set == "test_in"]
                run = load_ours(f, tag, s); ts = run["t_src"]
                p, y, img = fam.p.values, fam.y.values.astype(int), fam.image_id.values
                A = target_halves(img); cut = np.empty(len(p))
                for h in (A, ~A):
                    cut[h] = best_threshold(p[h], y[h])
                pred = p >= cut
                tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
                pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
                orc = 2 * pr * rc / (pr + rc) if pr + rc else 0
                rows.append({"group": grp, "tag": tag, "fold": f, "seed": s, "src": prf(p, y, ts)[2], "oracle": orc,
                             "cases": len(np.unique(img))})
    d = pd.DataFrame(rows)
    g = d.groupby(["group", "fold", "tag"])[["src", "oracle"]].mean().groupby("group").mean()
    P("1. FAMILIAR-scanner oracle gap (test_in, per-half oracle; mean seeds->fold x tag):")
    P((g.assign(gap=g.oracle - g.src)).round(4).to_string()); P("   familiar cases per fold:", d.groupby("fold").cases.first().to_dict())


def dev_extras():
    rows = []
    for tag in ["mix", "phikon", "phikon_stain", "phikon_um056", "phikon_um112"]:
        for f in FOLDS:
            for s in range(3):
                run = load_ours(f, tag, s)
                if run is None:
                    continue
                for m in ["SRC", "PRM", "LMM", "SLD", "EF1", "ORACLE"]:
                    r = evaluate(run, "plain", m, "crossfit")
                    rows.append({"tag": tag, "fold": f, "seed": s, "method": m, "f1": r[2]})
    d = pd.DataFrame(rows); d.to_csv(RES / "cutoff_dev_extra_runs.csv", index=False)
    pv = d.pivot_table(index="tag", columns="method", values="f1", aggfunc="mean")
    P("\n2. DEV extras (plain, crossfit; saved to results/cutoff_dev_extra_runs.csv): dF1 vs SRC")
    P(pv.sub(pv.SRC, axis=0).round(3).assign(SRC_F1=pv.SRC.round(3)).to_string())


def mw(df, cols, by=None):
    """Model-weighted means: seeds averaged within (fold, model), then mean over models (and folds unless in by)."""
    g = df.groupby(["fold", "model"] if "model" in df else ["fold", "name"])[cols].mean().reset_index()
    if by == "fold":
        return g.groupby("fold")[cols].mean()
    return g[cols].mean()


def axel_model_weighted():
    m = pd.read_csv(RES / "cutoff_mechanism_runs.csv"); a = m[m.source == "axel"].rename(columns={"name": "model"})
    a = a.assign(d_PRM=a.F1_PRM - a.F1_SRC, d_PRMop=a["F1_PRM-op"] - a.F1_SRC)
    P("\n3a. Axel mechanism, model-weighted: overall", mw(a, ["d_PRM", "d_PRMop"]).round(4).to_dict())
    P("    by fold:\n" + mw(a, ["d_PRM", "d_PRMop"], "fold").round(4).to_string())
    t = a[a.model == "optimus_teacher"].groupby("fold")[["d_PRM", "d_PRMop"]].mean()
    P("    teacher by fold:\n" + t.round(4).to_string(), "\n    teacher overall:", t.mean().round(4).to_dict())
    au = pd.read_csv(RES / "cutoff_audit.csv"); x = au[au.source == "axel"].rename(columns={"name": "model"})
    for k in (20, 50, 100):
        P(f"3b. Axel audit k={k} (model-weighted):", mw(x[x.k == k], ["dF1_PRM_labelfree", "dF1_DIRECT_k", "dF1_PRM_k"]).round(4).to_dict())
    r = pd.read_csv(RES / "referral_point_runs.csv"); ra = r[r.source == "axel"].rename(columns={"name": "model"})
    for meth in ("none", "STD"):
        P(f"3c. Axel referral error by fold ({meth}, model-weighted):", mw(ra[ra.method == meth], ["err_new"], "fold").err_new.round(4).to_dict())
    over = ra[ra.method == "none"]
    P("    Axel runs with err_new > 0.10:", int((over.err_new > 0.10).sum()), "of", len(over),
      over.assign(o=over.err_new > 0.10).groupby("fold").o.sum().to_dict())


def official_and_pr():
    pt = pd.read_csv(RES / "cutoff_point_runs.csv"); x = pt[(pt.source == "ours") & (pt.group.isin(["nonstain", "stain"]))]
    for split in ("crossfit", "official"):
        xs = x[(x["mode"] == "plain") & (x.split == split)]
        g = xs.groupby(["group", "fold", "name", "method"]).f1.mean().groupby(["group", "method"]).mean().unstack()
        P(f"\n4. our grid, plain, {split}: dF1 vs SRC", g.sub(g.SRC, axis=0).round(3).to_dict(orient="index"))
    pr = x[(x["mode"] == "plain") & (x.split == "crossfit")].groupby(["group", "method"])[["precision", "recall", "f1"]].mean()
    P("   precision / recall, plain crossfit:\n" + pr.round(3).to_string())


def referral_counts():
    r = pd.read_csv(RES / "referral_point_runs.csv")
    o = r[(r.source == "ours") & (r.method == "none")]
    P("\n5. our grid runs certifying no cutoff:", int(np.isinf(o.certified_conf_cutoff).sum()), "of", len(o),
      o.assign(z=np.isinf(o.certified_conf_cutoff)).groupby("name").z.sum().to_dict())
    dec = o[~np.isinf(o.certified_conf_cutoff)]
    P("   deciding runs with err_new > 0.10:", int((dec.err_new > 0.10).sum()), "of", len(dec))
    cov = r[r.method == "none"].groupby(["source", "name"]).cov_new.mean()
    P("   mean coverage per model (none):", cov.round(3).to_dict())


if __name__ == "__main__":
    familiar_gap(); dev_extras(); axel_model_weighted(); official_and_pr(); referral_counts()
    (ROOT / "results" / "cutoff_report_numbers.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))

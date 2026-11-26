"""POST-HOC EXPLORATORY: label-free prevalence from a BatchNorm "helper" model, handed to models without BatchNorm.

Idea: after AdaBN, label-shift EM estimates the new-scanner prevalence well (cutoff_mechanism.py), because AdaBN
removes most of the score shift. A small BN model can therefore supply a LABEL-FREE prevalence estimate pi_hat,
which PRM then uses: rate = r_s * pi_hat / pi_S ("PRM-helper"). No new-scanner label is used anywhere.

Helper: our ResNet all (5 seeds), AdaBN-cf probabilities on the new scanner, Platt on its own source validation,
EM prior on the unlabelled cells of the new scanner's official 'train' images; pi_hat = mean over the 5 seeds.
Targets: Axel's five models (official split: estimate on 'train' images, evaluate on 'test' images).
Writes results/cutoff_helper_prev.csv.
"""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, load_ours, load_axel, fit_platt, logit, sigmoid, prf
from cutoff_mechanism import sld_prior, prm_cut

AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]


def helper_prevalence(f, tag="all"):
    ests = []
    for s in range(5):
        run = load_ours(f, tag, s); t = run["tgt"]
        a, b = fit_platt(run["src_p"], run["src_y"])
        m = t.split.values == "train"
        q = sigmoid(a * logit(t.p_adabncf.values[m]) + b)
        ests.append(sld_prior(q, run["src_y"].mean()))
    return float(np.mean(ests))


def main():
    rows = []
    for f in FOLDS:
        pi_hat = helper_prevalence(f)
        for mdl in AXEL:
            for s in range(3):
                run = load_axel(mdl, f, s)
                if run is None:
                    continue
                t = run["tgt"]; tr, te = t.split.values == "train", t.split.values == "test"
                p, y = t.p_plain.values.astype(float), t.y.values.astype(int)
                sp, sy, ts = run["src_p"], run["src_y"], run["t_src"]; r_s = np.mean(sp >= ts); pi_s = sy.mean()
                cuts = {"SRC": ts, "PRM": prm_cut(p[tr], r_s),
                        "PRM-helper": prm_cut(p[tr], min(1.0, r_s * pi_hat / pi_s)),
                        "PRM-op": prm_cut(p[tr], min(1.0, r_s * y[tr].mean() / pi_s))}
                r = {"model": mdl, "fold": f, "seed": s, "pi_source": pi_s, "pi_helper": pi_hat, "pi_true_train": y[tr].mean()}
                for k, c in cuts.items():
                    r[f"F1_{k}"] = prf(p[te], y[te], c)[2]
                rows.append(r)
    d = pd.DataFrame(rows)
    d.round(5).to_csv(ROOT / "results" / "cutoff_helper_prev.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.groupby("fold")[["pi_source", "pi_helper", "pi_true_train"]].mean().round(3).to_string())
    f1 = [c for c in d.columns if c.startswith("F1_")]
    g = d.groupby(["model"])[f1].mean(); print((g.sub(g.F1_SRC, axis=0)).round(3).to_string())
    g = d.groupby(["fold"])[f1].mean(); print((g.sub(g.F1_SRC, axis=0)).round(3).to_string())
    # mean over models (fold x model, seeds averaged), as in the report
    gm = d.groupby(["fold", "model"])[f1].mean().mean()
    print("all Axel models:", (gm - gm.F1_SRC).round(4).to_dict())


if __name__ == "__main__":
    main()

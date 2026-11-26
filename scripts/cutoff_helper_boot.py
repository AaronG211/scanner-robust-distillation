"""POST-HOC EXPLORATORY: bootstrap CIs for PRM-helper on Axel's models (cases of the new scanner resampled; the helper
prevalence (5 ResNet-all seeds, AdaBN-cf + EM on 'train'-image cells) and every cutoff re-estimated on each resample;
Axel's seeds resampled). B = 1000. Writes results/cutoff_helper_boot.csv."""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, load_ours, load_axel, fit_platt, logit, sigmoid
from cutoff_confirm import w_f1

AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]

def w_em(q, w, pi_s):
    pi = pi_s; W = w.sum()
    for _ in range(300):
        w1, w0 = pi / pi_s * q, (1 - pi) / (1 - pi_s) * (1 - q)
        new = np.sum(w * w1 / (w1 + w0)) / W
        if abs(new - pi) < 1e-9: break
        pi = new
    return pi

def w_prm(p, w, r):
    m = w > 0; p, w = p[m], w[m]
    o = np.argsort(-p, kind="stable"); ps, cw = p[o], np.cumsum(w[o])
    return float(ps[np.searchsorted(cw, max(1, int(round(r * cw[-1]))))])

def main(B=1000):
    rng = np.random.default_rng(2031)
    sw = rng.multinomial(3, np.full(3, 1/3), size=B) / 3.0
    res = {k: np.zeros(B) for k in ("PRM", "PRM-helper")}; pts = {k: 0.0 for k in res}; n = 0
    for f in FOLDS:
        helpers = []
        for s in range(5):
            run = load_ours(f, "all", s); t = run["tgt"]; a, b = fit_platt(run["src_p"], run["src_y"])
            helpers.append((t.ann_id.values, sigmoid(a * logit(t.p_adabncf.values) + b), t.split.values == "train", run["src_y"].mean()))
        axel = {m: [load_axel(m, f, s) for s in range(3)] for m in AXEL}
        ref = axel["student_final"][0]["tgt"]; img_of = dict(zip(ref.ann_id, ref.image_id))
        cases = np.unique(ref.image_id.values)
        Wc = rng.multinomial(len(cases), np.full(len(cases), 1/len(cases)), size=B).astype(float)
        def cw(ann):  # case weights per cell for replicate b
            return np.searchsorted(cases, np.array([img_of[a] for a in ann]))
        hidx = [cw(h[0]) for h in helpers]
        for mdl in AXEL:
            runs = [r for r in axel[mdl] if r is not None]
            per = {k: np.zeros((B, len(runs))) for k in res}; ptv = {k: [] for k in res}
            for j, run in enumerate(runs):
                t = run["tgt"]; p, y = t.p_plain.values.astype(float), t.y.values.astype(int)
                tr, te = t.split.values == "train", t.split.values == "test"
                r_s = np.mean(run["src_p"] >= run["t_src"]); pi_s = run["src_y"].mean(); ci = cw(t.ann_id.values)
                for b in range(-1, B):
                    w = np.ones(len(p)) if b < 0 else Wc[b, ci]
                    pi_hat = np.mean([w_em(h[1][h[2]], (np.ones(len(h[1])) if b < 0 else Wc[b, hi])[h[2]], h[3]) for h, hi in zip(helpers, hidx)])
                    src = w_f1(p[te], y[te], run["t_src"], w[te])
                    for k, rate in (("PRM", r_s), ("PRM-helper", min(1.0, r_s * pi_hat / pi_s))):
                        v = w_f1(p[te], y[te], w_prm(p[tr], w[tr], rate), w[te]) - src
                        if b < 0: ptv[k].append(v)
                        else: per[k][b, j] = v
            for k in res:
                if len(runs) == 1: res[k] += per[k][:, 0]
                else:
                    ww = sw / sw.sum(1, keepdims=True); res[k] += (ww * per[k]).sum(1)
                pts[k] += np.mean(ptv[k])
            n += 1
    rows = []
    for k in res:
        d = res[k] / n
        rows.append({"method": k, "dF1_vs_SRC": pts[k] / n, "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5),
                     "P(harm)": float((d < 0).mean())})
    d = res["PRM-helper"] / n - res["PRM"] / n
    rows.append({"method": "PRM-helper - PRM", "dF1_vs_SRC": (pts["PRM-helper"] - pts["PRM"]) / n,
                 "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "P(harm)": float((d < 0).mean())})
    out = pd.DataFrame(rows); out.round(4).to_csv(ROOT / "results" / "cutoff_helper_boot.csv", index=False); print(out.round(4).to_string(index=False))

if __name__ == "__main__":
    main()

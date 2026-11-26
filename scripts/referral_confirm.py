"""CONFIRMATORY family B (CUTOFF_PROTOCOL.md, Amendment 1): does label-free score standardisation (STD) lower the
realised new-scanner error of a referral layer certified on familiar scanners?

Per run: Platt on source validation; Learn-then-Test cutoff c certified on familiar test cells (risk 0.10,
delta 0.10; fixed); new-scanner cells scored without (none) and with STD (logits standardised to the
source-validation logit mean / SD, moments from unlabelled new-scanner estimation cells), then the same Platt map
and the same c. Outcome: error among cells decided alone (H7: Axel's models, official test images;
H8: our grid tags, plain, all 50 cases, cross-fitted halves). Coverage reported alongside.

Writes results/referral_confirmatory.csv and results/referral_point_runs.csv.
"""
import argparse
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, RES, load_ours, load_axel, fit_platt, sigmoid, logit
from cutoff_referral import certify

GRID = ["baseline", "adv", "stain", "stain_adv", "all"]
AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]


class RRun:
    def __init__(self, source, fold, name, seed, run, fam_p, fam_y):
        self.source, self.fold, self.name, self.seed = source, fold, name, seed
        sp, sy = run["src_p"], run["src_y"]
        self.a, self.b = fit_platt(sp, sy)
        zs = logit(sp); self.mu_s, self.sd_s = zs.mean(), zs.std()
        qf = sigmoid(self.a * logit(fam_p) + self.b)
        self.c = certify(np.maximum(qf, 1 - qf), (qf > 0.5).astype(int) != fam_y)
        t = run["tgt"]
        self.img = t.image_id.values; self.y = t.y.values.astype(int); self.z = logit(t.p_plain.values.astype(float))
        if source == "ours":
            A = t.halfA.values; self.pairs = [(A, ~A), (~A, A)]; self.evalmask = np.ones(len(t), bool)
        else:
            tr, te = t.split.values == "train", t.split.values == "test"
            self.pairs = [(tr, te)]; self.evalmask = te

    def outcome(self, method, w):
        zz = self.z.copy()
        if method == "STD":
            for est, ev in self.pairs:
                we = w[est]; ze = self.z[est]
                mu = np.average(ze, weights=we); sd = np.sqrt(np.average((ze - mu) ** 2, weights=we))
                zz[ev] = self.mu_s + self.sd_s * (self.z[ev] - mu) / sd
        q = sigmoid(self.a * zz + self.b)
        conf = np.maximum(q, 1 - q); err = (q > 0.5).astype(int) != self.y
        m = self.evalmask; dec = (conf >= self.c) & m
        wd = (w * dec).sum(); we = (w * m).sum()
        return ((w * dec * err).sum() / wd if wd > 0 else np.nan), wd / we


def load():
    runs = []
    for f in FOLDS:
        for tag in GRID:
            for s in range(5):
                d = pd.read_csv(RES / "preds" / f"{f}_{tag}_seed{s}.csv"); fam = d[d.set == "test_in"]
                runs.append(RRun("ours", f, tag, s, load_ours(f, tag, s), fam.p.values, fam.y.values.astype(int)))
        for m in AXEL:
            d = pd.read_csv(ROOT / "data" / "axel_predictions" / f"{m}.csv")
            for s in sorted(d[d.holdout == f].seed.unique()):
                g = d[(d.holdout == f) & (d.seed == s)]; fam = g[g.set == "familiar_test"]
                runs.append(RRun("axel", f, m, int(s), load_axel(m, f, int(s)), fam.prob.values, fam.label.values.astype(int)))
    return runs


def agg(runs, sel, arr, seedw):
    keys = sorted({(R.fold, R.name) for R in runs if sel(R)}); tot = 0.0
    for f, n in keys:
        rs = [(i, R) for i, R in enumerate(runs) if R.fold == f and R.name == n and sel(R)]
        if len(rs) == 1:
            tot = tot + arr[rs[0][0]]
        else:
            sw = seedw[rs[0][1].source][:, [R.seed for _, R in rs]]
            sw = sw / sw.sum(1, keepdims=True).clip(min=1e-12)
            stack = np.stack([arr[i] for i, _ in rs], 1)
            ok = ~np.isnan(stack)
            tot = tot + np.nansum(sw * stack, 1) / np.where((sw * ok).sum(1) > 0, (sw * ok).sum(1), np.nan)
    return tot / len(keys)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    runs = load(); print(len(runs), "runs")
    rng = np.random.default_rng(2028)
    cases = {}
    for f in FOLDS:
        ids = np.unique(np.concatenate([R.img for R in runs if R.fold == f]))
        cases[f] = (ids, rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=a.B).astype(float))
    seedw = {"ours": rng.multinomial(5, np.full(5, 0.2), size=a.B) / 5.0,
             "axel": rng.multinomial(3, np.full(3, 1 / 3), size=a.B) / 3.0}
    E = {m: {} for m in ("none", "STD")}; C = {m: {} for m in ("none", "STD")}; pt = []
    for i, R in enumerate(runs):
        ids, Wc = cases[R.fold]; cidx = np.searchsorted(ids, R.img)
        for m in ("none", "STD"):
            e, c = np.empty(a.B), np.empty(a.B)
            for b in range(a.B):
                e[b], c[b] = R.outcome(m, Wc[b, cidx])
            E[m][i], C[m][i] = e, c
            e0, c0 = R.outcome(m, np.ones(len(R.y)))
            pt.append({"source": R.source, "fold": R.fold, "name": R.name, "seed": R.seed, "method": m,
                       "certified_conf_cutoff": R.c, "err_new": e0, "cov_new": c0})
    P = pd.DataFrame(pt); P.round(5).to_csv(ROOT / "results" / "referral_point_runs.csv", index=False)

    def point(src, m, col):
        x = P[(P.method == m) & (P.source == src)]
        return x.groupby(["fold", "name"])[col].mean().mean()

    tests = [("H7", "STD - none, Axel's no-BatchNorm models (official test images)", "axel"),
             ("H8", "STD - none, our grid tags, plain (all 50 cases, cross-fitted)", "ours")]
    rows = []
    for hid, label, src in tests:
        sel = (lambda src: lambda R: R.source == src)(src)
        d = agg(runs, sel, E["STD"], seedw) - agg(runs, sel, E["none"], seedw)
        dc = agg(runs, sel, C["STD"], seedw) - agg(runs, sel, C["none"], seedw)
        d = d[~np.isnan(d)]
        Pn = (d <= 0).mean(); p_raw = max(2 * min(Pn, 1 - Pn), 2 / len(d))
        en, es = point(src, "none", "err_new"), point(src, "STD", "err_new")
        cn, cs = point(src, "none", "cov_new"), point(src, "STD", "cov_new")
        bn = agg(runs, sel, E["none"], seedw); bs = agg(runs, sel, E["STD"], seedw)
        rows.append({"id": hid, "comparison": label, "err_none": en, "err_STD": es, "d_err": es - en,
                     "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "p_raw": p_raw,
                     "err_none_ci": f"[{np.nanpercentile(bn, 2.5):.3f}, {np.nanpercentile(bn, 97.5):.3f}]",
                     "err_STD_ci": f"[{np.nanpercentile(bs, 2.5):.3f}, {np.nanpercentile(bs, 97.5):.3f}]",
                     "cov_none": cn, "cov_STD": cs, "d_cov": cs - cn,
                     "d_cov_ci": f"[{np.percentile(dc, 2.5):.3f}, {np.percentile(dc, 97.5):.3f}]",
                     "P(err_none > 0.10)": float((bn > 0.10).mean()), "P(err_STD > 0.10)": float((bs > 0.10).mean())})
    c = pd.DataFrame(rows)
    order = np.argsort(c.p_raw.values); adj = np.empty(len(c)); mx = 0.0
    for rank, k in enumerate(order):
        mx = max(mx, min(1.0, (len(c) - rank) * c.p_raw.values[k])); adj[k] = mx
    c["p_holm"] = adj; c["significant_holm"] = c.p_holm < 0.05
    c.round(4).to_csv(ROOT / "results" / "referral_confirmatory.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(c.round(4).T.to_string())
    print("\nper model (point, mean over seeds then folds):")
    print(P.groupby(["source", "name", "method"])[["err_new", "cov_new"]].mean().unstack("method").round(3).to_string())
    print("\nper held-out scanner:")
    print(P.groupby(["source", "fold", "method"])[["err_new", "cov_new"]].mean().unstack("method").round(3).to_string())


if __name__ == "__main__":
    main()

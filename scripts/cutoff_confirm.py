"""CONFIRMATORY analysis for CUTOFF_PROTOCOL.md (frozen 2026-11-16T08:20Z): H1-H6 with Holm, plus exploratory tables.

Bootstrap (protocol): cases resampled within each fold, with multiplicities shared by every run and method of that
fold; a resampled case keeps its half / split and contributes to both estimation and evaluation; cutoffs are
re-estimated on every resample (weighted versions of the frozen methods, checked against them at unit weights);
seed indices resampled and shared across tags and methods. B = 2000, rng 2027.

Writes results/cutoff_confirmatory.csv, results/cutoff_exploratory.csv, results/cutoff_point_runs.csv.
Usage: .venv/bin/python scripts/cutoff_confirm.py [--B 2000]
"""
import argparse, time
import numpy as np, pandas as pd
from cutoff_transfer import (FOLDS, ROOT, load_ours, load_axel, evaluate, fit_platt, logit, sigmoid, best_threshold)

GRID_NONSTAIN, GRID_STAIN = ["baseline", "adv"], ["stain", "stain_adv", "all"]
ABL = ["all_no_grl", "all_no_soft", "all_no_stain", "all_stain_default"]
AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]
BOOT_METHODS = ["SRC", "PRM", "LMM", "SLD", "EF1"]


# ------------------------------------------------------------------ weighted versions of the frozen methods
def w_cutoff(method, r_s, zs_mu, zs_sd, t_src, platt, pi_s, p, w):
    """Cutoff from estimation cells p with integer bootstrap weights w (w = 1: identical to cutoff_transfer)."""
    if method == "SRC":
        return t_src
    m = w > 0
    p, w = p[m], w[m]
    if method == "PRM":
        if r_s <= 0:
            return np.inf
        o = np.argsort(-p, kind="stable"); ps, cw = p[o], np.cumsum(w[o])
        K = max(1, int(round(r_s * cw[-1])))
        return float(ps[np.searchsorted(cw, K)])
    if method == "LMM":
        z = logit(p); mu = np.average(z, weights=w); sd = np.sqrt(np.average((z - mu) ** 2, weights=w))
        return float(sigmoid(mu + sd * (logit(t_src) - zs_mu) / zs_sd))
    a, b = platt
    q = sigmoid(a * logit(p) + b)
    if method == "SLD":
        pi = pi_s; W = w.sum()
        for _ in range(200):
            w1, w0 = pi / pi_s * q, (1 - pi) / (1 - pi_s) * (1 - q)
            new = np.sum(w * w1 / (w1 + w0)) / W
            if abs(new - pi) < 1e-8:
                break
            pi = new
        R = (pi * (1 - pi_s)) / (pi_s * (1 - pi))
        q_star = sigmoid(a * logit(np.array([t_src]))[0] + b)
        zq = np.log(q_star / (1 - q_star)) - np.log(R)
        return float(sigmoid((zq - b) / a))
    if method == "EF1":
        o = np.argsort(-p, kind="stable"); ps, qs, ws = p[o], q[o], w[o]
        tp, k = np.cumsum(ws * qs), np.cumsum(ws)
        ef1 = 2 * tp / (k + (ws * qs).sum())
        return float(ps[int(np.argmax(ef1))])
    raise ValueError(method)


def w_f1(p, y, cut, w):
    pred = p >= cut
    tp = np.sum(w * (pred & (y == 1))); fp = np.sum(w * (pred & (y == 0))); fn = np.sum(w * (~pred & (y == 1)))
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


class Run:
    def __init__(self, source, fold, name, seed, group, run):
        self.source, self.fold, self.name, self.seed, self.group = source, fold, name, seed, group
        self.sp, self.sy, self.ts, self.t = run["src_p"], run["src_y"], run["t_src"], run["tgt"]
        self.r_s = float(np.mean(self.sp >= self.ts))
        zs = logit(self.sp); self.zs_mu, self.zs_sd = zs.mean(), zs.std()
        self.platt = fit_platt(self.sp, self.sy); self.pi_s = self.sy.mean()
        self.y = self.t.y.values.astype(int); self.A = self.t.halfA.values
        self.tr, self.te = self.t.split.values == "train", self.t.split.values == "test"
        self.p = {m: self.t[f"p_{m}"].values.astype(float) for m in ("plain", "adabncf")
                  if f"p_{m}" in self.t.columns and not self.t[f"p_{m}"].isna().any()}

    def f1(self, mode, method, split, wcell):
        if mode not in self.p:
            return np.nan
        p, y = self.p[mode], self.y
        args = (self.r_s, self.zs_mu, self.zs_sd, self.ts, self.platt, self.pi_s)
        if split == "crossfit":
            A = self.A; cut = np.empty(len(p))
            for est, ev in ((A, ~A), (~A, A)):
                cut[ev] = w_cutoff(method, *args, p[est], wcell[est])
            return w_f1(p, y, cut, wcell)
        est, ev = self.tr, self.te
        c = w_cutoff(method, *args, p[est], wcell[est])
        return w_f1(p[ev], y[ev], c, wcell[ev])


def load_all():
    runs = []
    for f in FOLDS:
        for tag in GRID_NONSTAIN + GRID_STAIN + ABL:
            grp = "nonstain" if tag in GRID_NONSTAIN else "stain" if tag in GRID_STAIN else "abl"
            for s in range(5):
                r = load_ours(f, tag, s)
                if r is None:
                    raise SystemExit(f"missing run {f} {tag} {s}")
                runs.append(Run("ours", f, tag, s, grp, r))
        for m in AXEL:
            for s in range(3):
                r = load_axel(m, f, s)
                if r is not None:
                    runs.append(Run("axel", f, m, s, "axel", r))
    return runs


def check_weighted(runs):
    """Weighted methods at unit weights must equal the frozen cutoff_transfer.evaluate."""
    worst = 0.0
    for R in runs:
        modes = ["plain"] + (["adabncf"] if R.source == "ours" else [])
        splits = ["crossfit", "official"] if R.source == "ours" else ["official"]
        for mode in modes:
            for split in splits:
                for m in BOOT_METHODS:
                    ref = evaluate({"src_p": R.sp, "src_y": R.sy, "t_src": R.ts, "tgt": R.t}, mode, m, split)
                    if ref is None:
                        continue
                    got = R.f1(mode, m, split, np.ones(len(R.t)))
                    worst = max(worst, abs(got - ref[2]))
    print(f"weighted-vs-frozen check: max |dF1| = {worst:.2e}")
    assert worst < 1e-9, "weighted methods do not reproduce the frozen ones"


# ------------------------------------------------------------------ bootstrap
def bootstrap(runs, B, rng):
    """F1[(run index, mode, split, method)] -> array (B,), and seed weights per source."""
    cases = {}
    for f in FOLDS:
        ids = np.unique(np.concatenate([R.t.image_id.values for R in runs if R.fold == f]))
        cases[f] = (ids, rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B).astype(float))
    seedw = {"ours": rng.multinomial(5, np.full(5, 0.2), size=B) / 5.0,
             "axel": rng.multinomial(3, np.full(3, 1 / 3), size=B) / 3.0}
    jobs = []
    for i, R in enumerate(runs):
        if R.source == "ours":
            for mode in ("plain", "adabncf"):
                for m in BOOT_METHODS:
                    if m == "SLD" and not (mode == "plain" and R.group in ("nonstain", "stain")):
                        continue               # SLD bootstrapped only where it is tested (H5)
                    jobs.append((i, mode, "crossfit", m))
        else:
            for m in BOOT_METHODS:
                jobs.append((i, "plain", "official", m))
    out = {j: np.empty(B) for j in jobs}
    t0 = time.time()
    by_run = {}
    for j in jobs:
        by_run.setdefault(j[0], []).append(j)
    for n, (i, js) in enumerate(by_run.items()):
        R = runs[i]; ids, Wc = cases[R.fold]
        cidx = np.searchsorted(ids, R.t.image_id.values)
        for b in range(B):
            w = Wc[b, cidx]
            for j in js:
                out[j][b] = R.f1(j[1], j[3], j[2], w)
        if n % 40 == 0:
            print(f"  bootstrap: {n + 1}/{len(by_run)} runs, {time.time() - t0:.0f}s", flush=True)
    return out, seedw


def aggregate(runs, vals, seedw, sel, mode, split, method):
    """Mean over folds, names and (resampled) seeds of the per-run F1 arrays for runs passing sel."""
    tot = 0.0; n_groups = 0
    keys = sorted({(R.fold, R.name) for R in runs if sel(R)})
    for f, name in keys:
        rs = [(i, R) for i, R in enumerate(runs) if R.fold == f and R.name == name and sel(R)]
        src = rs[0][1].source
        if len(rs) == 1:                             # single-seed model (the teacher): weight 1
            tot = tot + vals[(rs[0][0], mode, split, method)]
        else:
            sw = seedw[src][:, [R.seed for _, R in rs]]
            sw = sw / sw.sum(1, keepdims=True).clip(min=1e-12)
            tot = tot + sum(sw[:, k] * vals[(i, mode, split, method)] for k, (i, _) in enumerate(rs))
        n_groups += 1
    return tot / n_groups


def point(runs, sel, mode, split, method):
    vals = {}
    for i, R in enumerate(runs):
        if sel(R):
            vals[(i, mode, split, method)] = R.f1(mode, method, split, np.ones(len(R.t)))
    keys = sorted({(R.fold, R.name) for R in runs if sel(R)})
    return float(np.mean([np.mean([vals[(i, mode, split, method)] for i, R in enumerate(runs)
                                   if R.fold == f and R.name == n and sel(R)]) for f, n in keys]))


def point_oracle(runs, sel, mode, split):
    """ORACLE F1 (reference only), averaged like point(): over seeds, then over folds x names."""
    keys = sorted({(R.fold, R.name) for R in runs if sel(R)}); out = []
    for f, n in keys:
        v = [evaluate({"src_p": R.sp, "src_y": R.sy, "t_src": R.ts, "tgt": R.t}, mode, "ORACLE", split)[2]
             for R in runs if R.fold == f and R.name == n and sel(R)]
        out.append(np.mean(v))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    runs = load_all()
    print(f"{len(runs)} runs loaded")
    check_weighted(runs)
    rng = np.random.default_rng(2027)
    vals, seedw = bootstrap(runs, a.B, rng)

    nonstain = lambda R: R.group == "nonstain"
    stain = lambda R: R.group == "stain"
    grid = lambda R: R.group in ("nonstain", "stain")
    axel = lambda R: R.group == "axel"
    tests = [
        ("H1", "PRM - SRC, non-stain models, plain, crossfit", nonstain, ("plain", "crossfit", "PRM"), ("plain", "crossfit", "SRC")),
        ("H2", "PRM - SRC, stain models, plain, crossfit", stain, ("plain", "crossfit", "PRM"), ("plain", "crossfit", "SRC")),
        ("H3", "PRM - SRC, grid, AdaBN-cf, crossfit", grid, ("adabncf", "crossfit", "PRM"), ("adabncf", "crossfit", "SRC")),
        ("H4", "PRM - SRC, Axel's no-BatchNorm models, official", axel, ("plain", "official", "PRM"), ("plain", "official", "SRC")),
        ("H5", "SLD - SRC, grid, plain, crossfit", grid, ("plain", "crossfit", "SLD"), ("plain", "crossfit", "SRC")),
        ("H6", "(plain + PRM) - (AdaBN-cf + SRC), non-stain, crossfit", nonstain, ("plain", "crossfit", "PRM"), ("adabncf", "crossfit", "SRC")),
    ]
    rows = []
    for hid, label, sel, x, z in tests:
        bx = aggregate(runs, vals, seedw, sel, *x); bz = aggregate(runs, vals, seedw, sel, *z)
        dlt = bx - bz
        px, pz = point(runs, sel, *x), point(runs, sel, *z)
        P = (dlt <= 0).mean()
        p_raw = max(2 * min(P, 1 - P), 2 / a.B)
        rows.append({"id": hid, "comparison": label, "F1_first": px, "F1_second": pz, "dF1": px - pz,
                     "ci_lo": np.percentile(dlt, 2.5), "ci_hi": np.percentile(dlt, 97.5), "p_raw": p_raw,
                     "p_floor": p_raw == 2 / a.B})
    c = pd.DataFrame(rows)
    order = np.argsort(c.p_raw.values); m = len(c); adj = np.empty(m); run_max = 0.0
    for rank, k in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * c.p_raw.values[k])); adj[k] = run_max
    c["p_holm"] = adj; c["significant_holm"] = c.p_holm < 0.05
    c.round(4).to_csv(ROOT / "results" / "cutoff_confirmatory.csv", index=False)

    # exploratory: every method x subset x mode with bootstrap CI (unadjusted), plus oracle points
    ex = []
    subsets = [("non-stain grid", nonstain), ("stain grid", stain), ("ablation tags", lambda R: R.group == "abl"),
               ("Axel models", axel)] + [(f"Axel: {n}", (lambda n: lambda R: R.name == n)(n)) for n in AXEL]
    for sname, sel in subsets:
        is_axel = sname.startswith("Axel")
        for mode in (["plain"] if is_axel else ["plain", "adabncf"]):
            split = "official" if is_axel else "crossfit"
            base = aggregate(runs, vals, seedw, sel, mode, split, "SRC")
            for meth in BOOT_METHODS + ["ORACLE"]:
                pt = point(runs, sel, mode, split, meth) if meth != "ORACLE" else point_oracle(runs, sel, mode, split)
                row = {"subset": sname, "mode": mode, "split": split, "method": meth, "F1": pt}
                if meth != "ORACLE" and all((i, mode, split, meth) in vals for i, R in enumerate(runs) if sel(R)):
                    d = aggregate(runs, vals, seedw, sel, mode, split, meth) - base
                    row.update({"dF1_vs_SRC": pt - point(runs, sel, mode, split, "SRC"),
                                "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)})
                ex.append(row)
    e = pd.DataFrame(ex)
    e.round(4).to_csv(ROOT / "results" / "cutoff_exploratory.csv", index=False)

    # per-run points (every split, mode, method) for figures and per-scanner tables
    pr = []
    for R in runs:
        rr = {"src_p": R.sp, "src_y": R.sy, "t_src": R.ts, "tgt": R.t}
        for mode in (["plain"] if R.source == "axel" else ["plain", "adabncf"]):
            for split in (["official"] if R.source == "axel" else ["crossfit", "official"]):
                for meth in BOOT_METHODS + ["ORACLE"]:
                    r = evaluate(rr, mode, meth, split)
                    if r is not None:
                        pr.append({"source": R.source, "group": R.group, "fold": R.fold, "name": R.name, "seed": R.seed,
                                   "mode": mode, "split": split, "method": meth, "precision": r[0], "recall": r[1],
                                   "f1": r[2], "cut": r[3]})
    pd.DataFrame(pr).round(5).to_csv(ROOT / "results" / "cutoff_point_runs.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print("\nCONFIRMATORY (Holm over 6):"); print(c.round(4).to_string(index=False))
    print("\nEXPLORATORY:"); print(e.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

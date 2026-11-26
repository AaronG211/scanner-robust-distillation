"""EXPLORATORY (Amendment 2): paired bootstrap of (TTA + PRM) - (AdaBN-cf + TTA + SRC) on the grid, with the SAME case
and seed resamples for both arms (cutoff_tta.py resamples each mode separately and cannot give this paired CI).
B = 1000, rng 2030. The "ResNet all only" row was added post hoc. Writes results/cutoff_tta_vs_adabn.txt."""
import numpy as np
from cutoff_tta import load, GRID
from cutoff_confirm import Run
from cutoff_transfer import FOLDS, ROOT, evaluate

def main(B=1000):
    rng = np.random.default_rng(2030)
    R1 = {(f, t, s): Run("ours", f, t, s, g, load(f, t, s, "tta")) for f in FOLDS for t, g in GRID.items() for s in range(5)}
    R2 = {(f, t, s): Run("ours", f, t, s, g, load(f, t, s, "adabncf_tta")) for f in FOLDS for t, g in GRID.items() for s in range(5)}
    cases = {f: np.unique(R1[(f, "baseline", 0)].t.image_id.values) for f in FOLDS}
    Wc = {f: rng.multinomial(len(cases[f]), np.full(len(cases[f]), 1 / len(cases[f])), size=B).astype(float) for f in FOLDS}
    sw = rng.multinomial(5, np.full(5, .2), size=B) / 5.
    D, PT, A1, A2 = {}, {}, {}, {}
    for k in R1:
        f = k[0]; ci = np.searchsorted(cases[f], R1[k].t.image_id.values); one = np.ones(len(R1[k].t))
        D[k] = np.array([R1[k].f1("plain", "PRM", "crossfit", Wc[f][b, ci]) - R2[k].f1("plain", "SRC", "crossfit", Wc[f][b, ci]) for b in range(B)])
        A1[k], A2[k] = R1[k].f1("plain", "PRM", "crossfit", one), R2[k].f1("plain", "SRC", "crossfit", one)
        PT[k] = A1[k] - A2[k]
    lines = []
    for name, tags in (("non-stain", ["baseline", "adv"]), ("stain", ["stain", "stain_adv", "all"]), ("ResNet all only (post hoc)", ["all"])):
        tot = 0; pt = a1 = a2 = 0; n = 0
        for f in FOLDS:
            for t in tags:
                w = sw / sw.sum(1, keepdims=True); tot = tot + sum(w[:, s] * D[(f, t, s)] for s in range(5))
                pt += np.mean([PT[(f, t, s)] for s in range(5)]); a1 += np.mean([A1[(f, t, s)] for s in range(5)]); a2 += np.mean([A2[(f, t, s)] for s in range(5)]); n += 1
        d = tot / n
        lines.append(f"(TTA + PRM) - (AdaBN-cf + TTA), {name}: {pt/n:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                     f"   TTA+PRM {a1/n:.4f}  AdaBN-cf+TTA {a2/n:.4f}")
    (ROOT / "results" / "cutoff_tta_vs_adabn.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()

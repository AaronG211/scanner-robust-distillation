"""Teacher seed variance (descriptive; MULTI_TEACHER_PROTOCOL.md amendment 2). Runs on the PC from C:\\midog.

For every foundation model, fold and LoRA seed whose weights exist (C:/data/midog/fm/models/<prefix>_lora_<h>_plain_e5_s<seed>.pt):
view 0 (centre crop, no rotation, no stain jitter: identical to view 0 of distill_teacher.py) on the training scanners'
validation cells and on ALL new-scanner cells; cutoff = best F1 cutoff on the validation cells (the teacher's own
source cutoff). Seed 0 is checked against the existing targetall caches (same numbers up to bfloat16 noise).
Writes results/teacher_seeds_cells.csv.gz and results/teacher_seeds_summary.csv.
"""
import sys; sys.path.insert(0, r"C:\midog\repo\axel_model")
from pathlib import Path
import numpy as np, pandas as pd, torch
from train import make_splits, prf, auroc, best_threshold
from fm_lora import FMNet, FM, DEV, CROP

A = Path(r"C:\midog\repo\axel_model"); DIST = Path(r"C:\data\midog\distill")
d = dict(np.load(FM / "patches256.npz")); X, y = d["X"], d["y"].astype(int)   # teachers use the 256 px patches (as fm_lora.py / distill_teacher.py)
o = (X.shape[1] - CROP) // 2
fms = sys.argv[1:] or ["hopt0", "virchow2", "uni", "uni2h"]

@torch.no_grad()
def logits(model, idx, bs=64):
    mean, std = model.mean.to(DEV), model.std.to(DEV); out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(X[idx[s:s + bs], o:o + CROP, o:o + CROP]).to(DEV).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.mitosis(model.features(((x - mean) / std).to(torch.bfloat16))).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)

cells, rows = [], []
for fm in fms:
    model = FMNet(adversarial=False, checkpointing=False, fm=fm).to(DEV).eval()
    for h in ("CS2", "S360", "XR"):
        S, _ = make_splits(d, h); u = S["test_unseen"]; off = np.isin(u, S["test_unseen_official"])
        for seed in range(10):
            f = FM / "models" / f"{model.prefix}_lora_{h}_plain_e5_s{seed}.pt"
            if not f.exists():
                continue
            sd = torch.load(f, map_location=DEV, weights_only=True)
            missing = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("scanner.")}, strict=False).missing_keys
            assert not [k for k in missing if ".A" in k or ".B" in k or k.startswith("mitosis")], "LoRA / head weights missing"
            pv = 1 / (1 + np.exp(-logits(model, S["val"]))); thr = best_threshold(pv, y[S["val"]])
            pu = 1 / (1 + np.exp(-logits(model, u)))
            check = cut_cache = np.nan
            if seed == 0:
                c = DIST / f"teacher_{'' if fm == 'hopt0' else fm + '_'}{h}_targetall_s0.npz"
                if c.exists():
                    t = np.load(c); assert np.array_equal(t["idx"], u)
                    check = float(np.abs(1 / (1 + np.exp(-t["logit"][:, 0])) - pu).max()); cut_cache = float(t["threshold"])
            cells.append(pd.DataFrame({"fm": fm, "holdout": h, "seed": seed, "row": u, "p_teacher": pu, "thr_teacher": thr}))
            for name, m in (("all50", np.ones(len(u), bool)), ("official11", off)):
                pr, rc, f1 = prf(pu[m], y[u][m], thr)
                rows.append(dict(fm=fm, holdout=h, seed=seed, set=name, f1=f1, precision=pr, recall=rc,
                                 auroc=auroc(pu[m], y[u][m]), cutoff=thr, val_f1=prf(pv, y[S["val"]], thr)[2],
                                 max_abs_diff_vs_cache=check, cutoff_in_cache=cut_cache))
            print(fm, h, seed, f"F1 {rows[-2]['f1']:.3f}", f"cache check {check:.4f}", f"cutoff {thr:.4f} (cache {cut_cache:.4f})", flush=True)
    del model; torch.cuda.empty_cache()
pd.concat(cells).to_csv(A / "results" / "teacher_seeds_cells.csv.gz", index=False)
r = pd.DataFrame(rows); r.to_csv(A / "results" / "teacher_seeds_summary.csv", index=False)
a = r[r.set == "all50"]
print(a.pivot_table(index=["fm", "seed"], columns="holdout", values="f1").assign(mean=lambda x: x.mean(1)).round(3).to_string())

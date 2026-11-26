"""Optimus-free teacher for the control experiment: who should answer on the new scanner's unlabeled cells?

The teacher here is the ensemble of the label-only ResNet-18 runs (r18_control, never trained with Optimus), with
BatchNorm statistics re-estimated (AdaBN, no labels) on the same half of the new scanner's images that the student
adapts on. It answers on exactly the views the Optimus teacher answered on (same 8 rotations/flips, same stored
stain jitter), so distill.py --target_teacher r18ens differs from the Optimus run only in who gives the answer.
The ensemble answer is the mean logit of the runs. Labels are read only for the printed diagnostic.

Usage: python r18_teacher_cache.py --holdout XR --half A   ->  C:/data/midog/distill/teacher_XR_r18ensA_s0.npz
"""
import argparse
from pathlib import Path
import numpy as np, torch
from train import auroc, MEAN, STD, CROP
from eval_extras import adapt_bn, DEV
from probe_check import load
from distill_teacher import view, K
from crossfit import target_halves

HERE = Path(__file__).resolve().parent
DIST = Path("C:/data/midog/distill")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True, choices=["XR", "S360", "CS2"])
    ap.add_argument("--half", required=True, choices=["A", "B"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    d = dict(np.load(HERE / "cache" / "patches.npz")); X, y = d["X"], d["y"].astype(int)
    idx = target_halves(d, a.holdout)[a.half]
    full = np.load(DIST / f"teacher_{a.holdout}_targetall_s0.npz")
    keep = np.isin(full["idx"], idx); assert np.array_equal(full["idx"][keep], idx)
    sa, sb, opt = full["stain_a"][keep], full["stain_b"][keep], full["logit"][keep]
    o = (X.shape[1] - CROP) // 2; mean, std = MEAN.to(DEV), STD.to(DEV)
    logit = np.zeros((len(idx), K), np.float32)
    for seed in a.seeds:
        m = adapt_bn(load(DIST / f"student_{a.holdout}_resnet18_r18_control_s{seed}.pt"), X, idx)
        for v in range(K):
            for s in range(0, len(idx), a.bs):
                sl = slice(s, s + a.bs)
                x = torch.from_numpy(X[idx[sl], o:o + CROP, o:o + CROP]).to(DEV).permute(0, 3, 1, 2).float() / 255
                x = view(x, v, torch.from_numpy(sa[sl, v]).to(DEV), torch.from_numpy(sb[sl, v]).to(DEV))
                logit[sl, v] += m((x - mean) / std)[0].float().cpu().numpy() / len(a.seeds)
    sig = lambda z: 1 / (1 + np.exp(-z))
    for name, z in (("r18 ensemble + AdaBN", logit), ("Optimus + LoRA", opt)):
        p = sig(z)
        print(f"{name:22s} on {a.holdout} half {a.half} (diagnostic only): AUROC view0 {auroc(p[:, 0], y[idx]):.3f}, "
              f"AUROC mean of views {auroc(p.mean(1), y[idx]):.3f}, mean |p - label| {np.abs(p - y[idx, None]).mean():.3f}, "
              f"share of p > 0.5 {(p > 0.5).mean():.3f} (true share {y[idx].mean():.3f})")
    print(f"agreement of the two teachers on the side of 0.5: {((logit > 0) == (opt > 0)).mean():.3f}")
    out = DIST / f"teacher_{a.holdout}_r18ens{a.half}_s0.npz"
    np.savez(out, idx=idx, logit=logit, stain_a=sa, stain_b=sb); print("saved", out)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""After jobs/verify_concurrent.sh: compare the thr_plain / thr_tta of the re-computed adv TTA/AdaBN sidecars
(results/verify_20261107) with the stored ones (results/preds_tta); replace any that differ (old copy kept in
results/corrupt_mps_20261107). The thresholds come from source-val passes run during the concurrent-MPS window."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V, T, K = ROOT / "results/verify_20261107", ROOT / "results/preds_tta", ROOT / "results/corrupt_mps_20261107"
n = 0
for f in ["CS2", "S360", "XR"]:
    for s in range(3):
        new, old = V / f"{f}_adv_seed{s}.json", T / f"{f}_adv_seed{s}.json"
        if not new.exists():
            sys.exit(f"missing {new}")
        a, b = json.loads(new.read_text()), json.loads(old.read_text())
        if any(a[k] != b[k] for k in ("thr_plain", "thr_tta")):
            K.mkdir(exist_ok=True)
            shutil.move(old, K / old.name)
            shutil.copy(new, old)
            n += 1
            print("replaced thresholds", old.name, {k: (b[k], a[k]) for k in ("thr_plain", "thr_tta")})
print(f"adv threshold sidecars: 9 compared, {n} replaced")

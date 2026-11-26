#!/bin/zsh
# Re-compute every output that was produced while several MPS processes shared the GPU on 2026-11-07
# (TENT for the default tags, TTA/AdaBN + cross-fitted AdaBN for adv) and replace any that differ.
# Random-crop AdaBN outputs were already re-verified by their own integrity run (logs/adabn_random_verify.log).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
V=results/verify_20261107
mkdir -p $V
echo "[$(date +%H:%M:%S)] verify TENT (all default tags)"
$PY scripts/eval_tent.py --out-prefix verify_
$PY scripts/compare_preds.py --new results/preds_tta --pattern '*_tent.csv' --strip verify_ --replace
echo "[$(date +%H:%M:%S)] verify adv TTA / AdaBN"
$PY scripts/eval_tta_adabn.py --tags adv --seeds 0 1 2 --out-dir $V
$PY scripts/eval_adabn_crossfit.py --tags adv --seeds 0 1 2 --out-dir $V
$PY scripts/compare_preds.py --new $V --pattern '*_adv_seed?.csv' --replace
$PY scripts/compare_preds.py --new $V --pattern '*_adv_seed?_cf.csv' --replace
echo "[$(date +%H:%M:%S)] VERIFY_DONE"

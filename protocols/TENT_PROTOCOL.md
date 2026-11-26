# Protocol T: test-time entropy minimisation (TENT) as a teacher-free baseline

Written 2026-11-25 04:40 CT, before any TENT result exists. Motivated by the literature check
(`ai_health/Literature_Review_2026-11-25.md`: reviewers will ask for TENT / SHOT-type baselines) and by the exploratory
layer ladder (`scripts/layer_ladder.py`), which puts the BatchNorm layer at 17% to 32% of the student-teacher gap.
Question: does a standard test-time adaptation method without any teacher get what AdaBN gets, or what TGA gets?

## Setting

ResNet-18 students `r18_kd` (Optimus KD) and `r18_control` (labels only), 10 seeds, leave-one-scanner-out, same image halves
A/B as every other cross-fitted result. New-scanner F1, single view, over all 50 images per fold; per run, mean over
seeds, then folds. Cutoff: the student's own source-validation cutoff (no recalibration), so the comparison with AdaBN is
like for like.

## Method (fixed now)

For each half (fit = A, apply to B; and the reverse):
1. AdaBN on the fit half (BatchNorm running statistics re-estimated, as everywhere else).
2. TENT (Wang et al., ICLR 2021): only the BatchNorm affine parameters are trainable; BatchNorm layers use batch
   statistics, everything else in eval mode; loss = mean binary entropy of sigmoid(logit) on fit-half cells; Adam,
   batch 64, shuffled with a seed derived from (fold, seed, fit half); no augmentation.
   **Primary configuration: lr 1e-3, 1 epoch.**
3. Re-estimate the running statistics on the fit half with the tuned affine parameters; predict the other half.

Sensitivity (descriptive only, not tested): lr 1e-4 / 1 epoch, lr 1e-3 / 3 epochs, lr 1e-2 / 1 epoch.

## Pre-specified family (4 contrasts, Holm; paired bootstrap over images and seeds, B = 20000)

1. `r18_kd`: TENT vs AdaBN
2. `r18_kd`: TENT vs KD + TGA (`r18_kd_tga`, source cutoff)
3. `r18_control`: TENT vs AdaBN
4. `r18_control`: TENT vs the Optimus-free adaptation (`r18_free_tga`, source cutoff)

Expectations written down now: (1) and (3) within ±0.01 and not significant (entropy minimisation with a binary,
imbalanced head is not expected to add much beyond the statistics); (2) and (4) negative and significant (training-time
adaptation with soft targets beats test-time entropy). Any contrast that breaks an expectation is reported as such.

## Descriptive only

TENT followed by TGR (one bias to the Optimus teacher's probabilities on the fit half); the sensitivity configurations;
per-scanner tables; the positive rate after TENT (entropy minimisation can collapse toward one class).

## Amendments

(none yet)

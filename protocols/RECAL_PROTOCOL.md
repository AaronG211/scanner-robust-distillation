# Protocol R: teacher-guided recalibration (TGR) against label-free recalibration baselines

Written 2026-11-25 (night), before the per-cell source-validation predictions (`e0_cells_r18`) and the ConvNeXt-T
students (`cnx_kd`, `cnx_control`) exist. Motivated by the literature check (`ai_health/Literature_Review_2026-11-25.md`):
UPA (Roschewitz et al. 2023), EM prior adjustment and teacher pseudo-label cutoffs are the baselines a reviewer will ask
for, and TGR is only interesting where AdaBN cannot be used (no BatchNorm, very few images).

## Setting

New-scanner F1, single view, cross-fitted: every method is fitted on the unlabelled cells of one half of the new
scanner's images and applied to the other half; F1 per run over all 50 images; mean over seeds, then folds.
Teacher: the Optimus + LoRA teacher (view 0 of its targetall cache, its own source-validation cutoff t_T).
Students: `r18_kd` (ResNet-18, BatchNorm) and `cnx_kd` (ConvNeXt-T, LayerNorm, so AdaBN does not apply), 10 seeds each.

## Methods (all label-free on the new scanner)

| Name | Definition |
|---|---|
| none | student's source-validation cutoff |
| TGR | one bias b on the student logit, b minimises BCE(sigmoid(z + b), teacher probability) on the fit half; predict z + b >= logit(t_T) |
| teacher-rate | student cutoff on the fit half so that its positive rate equals the teacher's positive rate (p_T >= t_T) there |
| teacher-PL | student cutoff maximising F1 against the teacher's hard labels (p_T >= t_T) on the fit half |
| source-rate (UPA-style) | student cutoff on the fit half so that its positive rate equals its own positive rate on the source validation set at its source cutoff (assumes prevalence is preserved) |
| EM prior | Saerens et al. (2002) EM on the fit half's student posteriors; source prior = mean student posterior on the source validation set; shift the logit by logit(pi_target) - logit(pi_source); keep the source cutoff |
| AdaBN | BatchNorm statistics re-estimated on the fit half (ResNet only; existing cross-fit cells) |
| oracle (reference) | cutoff chosen with labels on the fit half: an upper bound, not a method |

## Pre-specified family 1 (9 contrasts, Holm; paired bootstrap over images and seeds, B = 20000)

ResNet KD student: TGR vs none; TGR vs source-rate; TGR vs EM prior; TGR vs teacher-PL; TGR vs AdaBN.
ConvNeXt KD student: TGR vs none; TGR vs source-rate; TGR vs EM prior; TGR vs teacher-PL.

Expectations: TGR > none for both students; TGR > source-rate and TGR > EM prior for both (the new scanners differ in
prevalence and the students are not calibrated); no directional expectation for TGR vs teacher-PL; on the ResNet student
with the full fit half, TGR < AdaBN (pilot: 0.762 vs 0.772).

## Pre-specified family 2 (3 contrasts, Holm): very few images

ResNet KD student, TGR vs AdaBN with only n = 1, 2, 5 unlabelled images of the fit half (the same 10 image draws for both;
`scripts/e02_tgr_budget.py` and `jobs/pc/e0_gpu_diag.py budget`). Expectation: TGR >= AdaBN at n <= 2.

## Descriptive only

TGR cost (teacher forward passes on n images); all methods on the no-teacher students (`r18_control`, `cnx_control`);
per-scanner tables; the same methods with the Virchow2 / UNI / UNI2-h teachers once E1.7 is done.

Context already known when this was written (exploratory, not part of the test): on the ResNet KD student TGR gives
0.762, teacher-rate 0.762, oracle cross-fitted cutoff 0.762, AdaBN 0.772, none 0.725.

## Amendments

(none yet)

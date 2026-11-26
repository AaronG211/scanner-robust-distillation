# Protocol P: does the teacher help set the cutoff when the new scanner's mitosis prevalence also shifts?

Written 2026-11-25 12:39 CT, before any prevalence-resampled number was computed. CPU only (Mac), existing per-cell
predictions; no new training, no new data. Replaces the shelved second-dataset test (`REPLICATION_PROTOCOL_DRAFT.md`):
the user chose to stay on MIDOG++ breast, so prevalence shift is simulated by resampling cells of the new scanner.

## Why

On the natural breast scanners (positive rate 0.35 to 0.43) TGR did not beat "source-rate" on the ResNet KD student
(+0.002, n.s.) but did on ConvNeXt (+0.019, Holm 0.001) (`RECAL_PROTOCOL.md`, family 1). Source-rate assumes the new
scanner has the same positive rate as the source; a fixed student cutoff ("none") lets the positive rate follow the
data but is hurt by the scanner shift; TGR ties the student to the teacher's fixed decision rule, so if the teacher is
robust to the scanner it should follow both the scanner shift and the prevalence shift. The other MIDOG++ tumour types
span positive rates 0.27 to 0.63, so this is the regime a deployed model would meet.

## Setting (identical to `RECAL_PROTOCOL.md` except for the resampling)

- Students: `r18_kd` (ResNet-18) and `cnx_kd` (ConvNeXt-T), Optimus teacher, 10 seeds; per-cell probabilities from
  `data/pc_runs_2026-11-17/e0_cells_r18.csv.gz` and `e0_cells_cnx.csv.gz` (split "unseen" = all 50 new-scanner images,
  split "val" = source validation).
- Teacher: Optimus + LoRA, view 0, its own source-validation cutoff (`teacher_newscanner_cells.csv.gz`).
- Methods, exactly as in `scripts/recal_compare.py`: none, TGR, source-rate, teacher-rate, teacher-PL, EM prior, oracle.
  AdaBN is not included (its statistics depend on the class mix of the fit half and cannot be recomputed without the GPU).
- Cross-fitting over the image halves A/B as before: fit on the retained unlabelled cells of one half, score the
  retained cells of the other.

## Prevalence resampling

Target prevalence grid pi in {0.20, 0.30, 0.40, 0.50, 0.60} (covers the MIDOG++ range 0.27 to 0.63), plus "natural"
(no resampling). For each fold (held-out scanner) and pi: if pi is below the scanner's natural positive rate, keep all
negatives and a uniform random subset of round(pi / (1 - pi) * N_neg) positives; otherwise keep all positives and
round((1 - pi) / pi * N_pos) negatives. R = 20 independent draws per (fold, pi), drawn once (generator seeded
20261130 + 100 * fold index + pi index, folds ordered CS2, S360, XR) and shared by every student, seed, method and
teacher. Resampling is over all cells of the new scanner, before the A/B split, so the unlabelled fit half has the
shifted prevalence too.

F1 for one (student, seed, method, pi): per-image counts (TP, FP, FN) summed over the R draws, then F1 of the pooled
counts. Mean over seeds, then over the three folds.

## Inference

Paired bootstrap over images (multinomial weights over the 50 images of each fold) and seeds (resampled per fold),
B = 20000, two-sided percentile p as in `recal_compare.py`. Bootstrap weights drawn once per fold from a generator
seeded [20261131, fold index] and reused for every pi, student and method (so the numbers do not depend on which other
inputs are present). Draws of the resampling are held fixed (conditioned on).

## Pre-specified family (8 contrasts, Holm)

At the two ends of the grid, pi = 0.20 and pi = 0.60 (0.12 to 0.25 below and 0.15 to 0.28 above every fold's
source-validation positive rate of 0.32, 0.42, 0.45):

| # | Student | pi | Contrast | Expectation |
|---|---|---|---|---|
| 1 | r18_kd | 0.20 | TGR vs source-rate | TGR better, significant |
| 2 | r18_kd | 0.60 | TGR vs source-rate | TGR better, significant |
| 3 | cnx_kd | 0.20 | TGR vs source-rate | TGR better, significant |
| 4 | cnx_kd | 0.60 | TGR vs source-rate | TGR better, significant |
| 5 | r18_kd | 0.20 | TGR vs none | TGR better, significant |
| 6 | r18_kd | 0.60 | TGR vs none | TGR better, significant |
| 7 | cnx_kd | 0.20 | TGR vs none | TGR better, significant |
| 8 | cnx_kd | 0.60 | TGR vs none | TGR better, significant |

Any contrast that breaks its expectation is reported as such.

## Consistency check (must pass before results are read)

At "natural" the script must reproduce the family-1 F1 means of `recal_compare.py` (r18_kd: none 0.725, source-rate
0.760, TGR 0.762; cnx_kd: none 0.765, source-rate 0.782, TGR 0.801) to 3 decimals.

## Descriptive only (no test, no correction)

- Every method at every pi, both students, and the no-teacher students (`r18_control`, `cnx_control`).
- TGR minus source-rate and TGR minus none against (pi minus the fold's source-validation positive rate): a figure.
- The oracle cutoff and the teacher alone at each pi.
- TGR with the Virchow2, UNI and UNI2-h teachers as calibrators (`teacher_newscanner_cells_multi.csv.gz`); UNI's
  cutoff did not transfer on natural breast, so it is expected to fail here too.
- Sensitivity: a second set of R = 20 draws (generator offset + 50).

## Context already known when this was written

Natural-prevalence results above (family 1, final). Nothing at any resampled pi has been computed or looked at.

## Amendments

(none yet)

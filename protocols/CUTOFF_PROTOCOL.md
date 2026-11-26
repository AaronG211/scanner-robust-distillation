# Pre-registered protocol: label-free decision-cutoff transfer under scanner shift

Frozen 16 November 2026, before any confirmatory run was analysed. Author: Claude (research assistant for Aaron Gao).
Any later change is added as a dated amendment at the end; nothing above the amendments is edited.

## Question

A mitosis classifier picks its F1 cutoff on labelled source-scanner validation cells. On a new scanner the score
distribution moves. How much F1 is lost because the cutoff lands in the wrong place? Can a cutoff chosen from
*unlabelled* new-scanner cells recover that loss? In particular, can it stand in for AdaBN in models that have no
BatchNorm (Optimus and other ViT foundation models; ConvNeXt with LayerNorm)?

## Development evidence that shaped this protocol (exploratory; not reused below)

Development set: phase-1 runs (tags `macenko`, `macenko_stain`, `stain_ema`, `stain_he`, `stain_mix`, `stain_s002`,
`stain_s010`, `stain_scale02`, `stain_soft`, plus `mix` and the frozen Phikon probes; 3 seeds × 3 folds). Script
`scripts/cutoff_dev.py`, output `results/cutoff_dev_runs.csv`.

- For stain-augmented models the cutoff gap is small. An oracle cutoff on the target labels adds only ≈ +0.03 F1.
  PRM / LMM / EF1 recover ≈ 0–20 % of it, and add nothing on top of cross-fitted AdaBN.
- For models without stain augmentation (MixStyle `mix`; frozen Phikon probes) the oracle gap is larger
  (≈ +0.03 to +0.06), and PRM / LMM recover about half of it (≈ +0.015 to +0.03).
- Label-shift EM (SLD) is harmful (−0.06 to −0.12 F1), as expected when the shift is covariate, not label, shift.
- The method chosen as primary is **PRM** (non-parametric, no fitted model). LMM is the secondary method.

## Data (confirmatory)

1. **Our ResNet-18 grid, phase 2:** tags `baseline`, `adv` ("non-stain" models) and `stain`, `stain_adv`, `all`
   ("stain" models); 5 seeds × 3 held-out scanners = 75 runs. Modes: `plain` (as trained) and `adabncf`
   (cross-fitted AdaBN). Predictions from `results/preds/` and `results/preds_tta/*_cf.csv`.
2. **Axel's models without BatchNorm** (`data/axel_predictions/`): Optimus + LoRA teacher (1 seed) and four
   ConvNeXt-T students (`student_final`, `_conf`, `_grl`, `_target`; 3 seeds), × 3 held-out scanners.
3. Exploratory only: the ablation tags `all_no_grl`, `all_no_soft`, `all_no_stain`, `all_stain_default`.

## Methods (code: `scripts/cutoff_transfer.py`, frozen with this protocol)

`SRC` (source cutoff, status quo), `PRM` (positive-rate matching), `LMM` (logit moment matching), `SLD`
(label-shift EM), `EF1` (plug-in expected F1). `ORACLE` (F1-optimal on the evaluation labels) is a reference only.
Source quantities always come from source validation under source BatchNorm.

## Splits on the new scanner

- **Our runs:** `crossfit` (primary). The cutoff estimated on case half A's unlabelled cells is applied to half B,
  and vice versa. These are the same halves as cross-fitted AdaBN; F1 is computed over all 50 cases.
  `official` (secondary): estimate on the scanner's `train` images, evaluate on its `test` images.
- **Axel's models:** `official` only, because `student_final_target` adapted on the `train` images.

## Metric and statistics

- **Metric:** F1 on the held-out scanner (primary); precision and recall as secondary.
- **Bootstrap:** paired two-level bootstrap, B = 2000, rng seed 2027.
  - Cases are resampled within each fold. A resampled case keeps its half, and its cells go into both the
    estimation and the evaluation part. Cutoffs are re-estimated on every resample.
  - Seed indices are resampled and shared across tags and methods.
  - The statistic is the mean over folds, tags and seeds of ΔF1 (method − comparator) on the same resample.
- **p-values:** p = max(2·min(P(Δ ≤ 0), P(Δ ≥ 0)), 2/B). Holm adjustment over the six tests below.
- **Null results** are reported as "no effect larger than the CI bound", not as "no effect".

## Confirmatory family (Holm over 6)

| # | Comparison | Data | Expected from development |
|---|---|---|---|
| H1 | PRM − SRC | non-stain models (`baseline`, `adv`), `plain`, crossfit | positive |
| H2 | PRM − SRC | stain models (`stain`, `stain_adv`, `all`), `plain`, crossfit | ≈ 0 |
| H3 | PRM − SRC | all five grid tags, `adabncf`, crossfit | ≈ 0 or negative |
| H4 | PRM − SRC | Axel's five models (no BatchNorm), official split | open |
| H5 | SLD − SRC | all five grid tags, `plain`, crossfit | negative |
| H6 | (`plain` + PRM) − (`adabncf` + SRC) | non-stain models, crossfit | negative: AdaBN does more than move the cutoff |

## Exploratory analyses (described, not tested)

- LMM and EF1 on every subset above; the `official` split for our runs.
- Oracle-gap decomposition of AdaBN's gain into cutoff vs ranking (AUROC).
- Per held-out scanner results.
- Whether the referral layer (Platt on val, Learn-then-Test at 10 % risk on familiar test cells) keeps its error
  bound on the new scanner, with and without a label-free cutoff shift.

## Amendments

### Amendment 1 (16 November 2026, written before the analysis it describes)

**What had been looked at.** The exploratory referral analysis (`scripts/cutoff_referral.py`, results
`results/cutoff_referral_runs.csv`) showed that a referral layer certified on familiar scanners often has a higher
realised error on the new scanner than its 10 % target. This was clearest for Axel's no-BatchNorm models
(mean 9–12 %; every run with XR held out exceeded 10 %).

**What had not been looked at.** Any label-free fix for the referral layer.

**New confirmatory family B (Holm over 2, separate from family A = H1–H6).**

Method `STD` (score standardisation before the referral layer):
- Standardise the new-scanner logits to the source-validation logit mean and SD, as in LMM.
- The moments are estimated on unlabelled new-scanner cells: cross-fitted case halves for our runs, the official
  `train` images for Axel's models.
- Then apply the unchanged source Platt map and the unchanged certified confidence cutoff.

| # | Comparison | Data | Metric |
|---|---|---|---|
| H7 | STD − none | Axel's five models, official test images | realised error among decided cells (lower = better) |
| H8 | STD − none | our five grid tags, `plain`, all 50 cases (crossfit) | the same |

- **Coverage:** the share of cells decided alone is reported beside each test as a co-primary descriptive
  outcome. It is not tested, because a lower error bought with much lower coverage is not a free gain.
- **Statistics:** as in the main protocol, plus four more rules.
  - Cases are resampled within folds and seeds are resampled; B = 2000, rng seed 2028.
  - The certified cutoff is fixed (computed once on familiar cells, which are not resampled).
  - STD moments are re-estimated on every resample.
  - The error is averaged over runs as the mean of the per-run error rates.

### Amendment 2 (16 November 2026, before the analysis it describes; exploratory only, no tests)

`scripts/val_tta_probs.py` recomputes source-validation probabilities under TTA for the 75 grid runs. It checks
that `best_threshold` on them reproduces every stored `thr_tta` exactly. These probabilities let the cutoff methods
also run in the `tta` and `adabncf_tta` modes. Those results are exploratory: they are described with
unadjusted bootstrap CIs and are not added to either confirmatory family.

### Amendment 3 (16 November 2026, written after the confirmatory results; a transparency log, changes no test)

**Post-hoc analyses run after families A and B were computed.** All are hypothesis-generating.

| Analysis | Script |
|---|---|
| Oracle-prior and EM-prevalence diagnostics | `cutoff_mechanism.py` |
| Data amount | `cutoff_mechanism.py` |
| Labelled audit | `cutoff_audit.py` |
| Helper-prevalence PRM | `cutoff_helper_prev.py`, `cutoff_helper_boot.py` |
| Familiar-scanner oracle reference, development extras, re-weighted numbers | `cutoff_report_numbers.py` |
| Paired TTA comparison | `cutoff_tta_paired.py` |

The development extras (MixStyle and the Phikon probes) had been run interactively before the freeze but were not
saved. They are now saved in `results/cutoff_dev_extra_runs.csv`.

**Deviations of the implementation from the wording above.**

1. **p-values.** They use P(Δ > 0) = 1 − P(Δ ≤ 0) instead of P(Δ ≥ 0). With averaged F1 differences, ties at 0
   have probability ≈ 0, so this changes nothing in practice.
2. **Family B averages.** The error is averaged per (fold, model) cell: seeds first, then cells, which gives each
   model equal weight. Amendment 1 says "the mean of the per-run error rates".
   - Runs in which Learn-then-Test certifies no cutoff (coverage 0; 20 of our 75 grid runs) have an undefined
     error. They are excluded from the error mean but kept, as zeros, in the coverage mean.
   - The plain per-run means would be 8.0 % (ours) and 10.8 % (Axel's).
3. **What the bootstrap holds fixed.** The CIs are conditional on the source-validation quantities (t_src, the
   source positive rate, Platt, the source moments) and on the precomputed AdaBN-cf BatchNorm statistics. Only the
   target-side label-free cutoffs and moments are re-estimated on each resample.

### Amendment 4 (16 November 2026, ~21:57Z, written before the analysis it describes; sensitivity analysis)

**Why.** The family-A CIs hold the source-validation quantities fixed (Amendment 3, deviation 3). They therefore
ignore the uncertainty in where the source cutoff t_src and the source positive rate land. H1 and H2 passed with
Holm p = 0.032, so this matters for them; it cannot turn a non-significant test into a significant one.

**What.** `scripts/cutoff_sensitivity.py` repeats family A (H1–H6) with a *full-pipeline* bootstrap:
- Target cases resampled within fold and seeds resampled, exactly as before.
- In addition, the source-validation cases are resampled. Runs with the same set of validation cases share the
  same draw. On every resample t_src is re-chosen with `best_threshold` on the weighted source cells; the source
  positive rate (PRM), and for H5 the Platt map and the source prevalence, are re-estimated.
- B = 2000, rng seed 2029. Same p-value rule (P(Δ > 0) form, Amendment 3) and Holm over 6.
- Still held fixed: the trained weights (training randomness enters only through the seed resampling) and the
  cross-fitted AdaBN BatchNorm statistics (H3, H6).

**Decision rule, fixed now.** A family-A result counts as robust if it is Holm-significant in both the conditional
(original) and the full-pipeline analysis. If H1 or H2 loses significance here, the report must say that the PRM
gain is not robust to source-validation uncertainty. The point estimates do not change.

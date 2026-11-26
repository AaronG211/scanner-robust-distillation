# Protocol X (DRAFT, not frozen): replication on the other MIDOG++ domains

Status: draft written 2026-11-25 08:20 CT. No image outside the breast subset has been downloaded or seen. Freeze (hash in
`jobs/pc/protocol_hashes.txt`) only after the team agrees and before any download.

**SHELVED 2026-11-25 10:58 CT (user decision): stay on breast only; no second dataset is downloaded. Kept for reference
only.**

Why: every finding so far rests on one dataset (MIDOG++ human breast, three scanners, 50 images each). The strongest
new signal (the teacher adds little at every layer; TGR does not beat matching the source positive rate) has to be
tested where it could fail.

## What MIDOG++ offers (from the annotation file already on the PC; no images read)

| Domain | Scanner | Lab | Images | Mitotic | Imposter | Positive rate |
|---|---|---|---|---|---|---|
| human breast (current) | XR / S360 / CS2 | UMC Utrecht | 150 | 1721 | 2714 | 0.35 to 0.43 |
| human melanoma | Hamamatsu XR | UMC Utrecht | 49 | 1150 | 925 | 0.55 |
| human neuroendocrine | Hamamatsu XR | UMC Utrecht | 55 | 639 | 1761 | 0.27 |
| canine mast cell tumour | Aperio CS2 | FU Berlin | 50 | 2327 | 1366 | 0.63 |
| canine lung | 3D Histech | VMU Vienna | 44 | 855 | 951 | 0.47 |
| canine lymphoma | 3D Histech | VMU Vienna | 55 | 3959 | 4257 | 0.48 |
| canine soft tissue sarcoma | 3D Histech | AMC New York / VMU Vienna | 100 | 1286 | 2375 | 0.35 |

The positive rates differ much more than across the breast scanners (0.27 to 0.63). This is exactly where "match the
source positive rate" should break and the teacher should matter, if it matters at all.

Download: 353 more images of the same format (about 117 MB each as uint8 arrays, about 41 GB). PC has about 1 TB free.

## Design (proposed)

Source: all 150 breast images (the three scanners), the same patch extraction and splits code.
New domains: each of the six other tumour types, one at a time; all images of that domain are unlabelled for
adaptation and labelled for scoring, cross-fitted over halves A/B as now.

Teachers: H-optimus-0 and Virchow2 (the two whose cutoffs transferred on breast; licences as before, Virchow2 stays local).
Students: ResNet-18, 10 seeds. Arms: no teacher, KD, KD + TGA, teacher-free adaptation, plus the post-hoc layers
(TGR, source-rate, AdaBN, oracle cutoff), all exactly as on breast.

## Contrasts to pre-specify (Holm within the family; paired bootstrap over images and seeds, B = 20000)

Per teacher, averaged over the six domains:

1. KD vs no teacher. Expect: small, not significant.
2. KD + TGA vs KD. Expect: positive, significant.
3. KD + TGA vs teacher-free adaptation. Expect: between 0 and +0.03.
4. TGR vs source-rate on the KD student. Expect: TGR better, and the gain grows with the gap between the domain's
   positive rate and the source's (descriptive regression over the six domains, n = 6).
5. TGR vs AdaBN with 1 unlabelled image. Expect: TGR better.

Descriptive: the ladder and Shapley shares per domain; human vs canine domains; teacher alone per domain.

## Cost (estimate from breast timings)

Optimus LoRA about 9 min, Virchow2 about 6 min (one source, so one teacher each); teacher caches about 20 min;
students: 6 domains × (KD shared, TGA 2 halves) × 10 seeds × about 1 min, about 2 to 3 h per teacher. Total about one
night of GPU.

## Open decisions for the team

- Download the full MIDOG++ (public on figshare; I believe CC BY 4.0, to be checked before download) to the PC?
- Source = breast only, or leave-one-domain-out over all seven (seven times the teacher cost)?
- Is detection-level evaluation needed for MICCAI, or is cell classification enough?

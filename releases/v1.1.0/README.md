# PHYLLO: PlantSeg Integrity Audit and Reproducibility Records — v1.1-C

## Purpose and scope

This resource provides derived quality-control, candidate near-duplicate, group-aware partition, human-validation and reference-evaluation records for PlantSeg v7. It does not redistribute the original photographs or segmentation masks, correct their annotations, or establish an independently verified duplicate-free dataset.

**Release status:** This document accompanies a release candidate. The v1.1-C partition is **provisional**. The repository and archive version, DOI, licensing clearance and public-file inventory must be confirmed before this document is used as a published release README.

## Source dataset

PlantSeg v7: https://doi.org/10.5281/zenodo.17719108. The source archive has 7,774 images and corresponding masks. The source archive MD5 recorded in the study is `9358a66dff88cdd15c4fe009763c40a3`. Users must obtain the original images and masks separately and comply with the source-record and image-level terms.

A complete mask-value consistency check identified 12 masks whose disease values differ from the class indicated by source metadata (eight originally in training and four in test). These masks were excluded without changing source files. The retained population is 7,762 images.

## Distinguish the evaluation populations

| Population / partition | Train | Validation | Test | Purpose |
| --- | ---: | ---: | ---: | --- |
| Original source | 5,367 | 846 | 1,561 | Source dataset membership |
| Original after mask QC | 5,359 | 846 | 1,557 | Fixed-cohort comparisons |
| Corrected v1.0 | 5,432 | 778 | 1,552 | Historical hard-band component partition |
| Corrected v1.1-C | 5,434 | 776 | 1,552 | Provisional all-candidate-edge containment |

The v1.0 and v1.1-C test lists are **not identical**. Comparisons between their scores are descriptive, not paired causal estimates. The authoritative partition is defined by its released lists and checksums, not by rebuilding it in a different Python environment.

## Candidate detection and grouping

The frozen detector used 64-bit perceptual hashes and CLIP ViT-B/32 image embeddings, with exact comparisons of all 7,774 images. Hard-band candidates satisfy pHash Hamming distance at most 3 **or** CLIP cosine similarity at least 0.98. Soft-band candidates are not hard-band candidates and satisfy pHash distance 4–9 **or** cosine similarity from 0.95 (inclusive) to 0.98 (exclusive).

The original candidate census contains 788 hard-band and 1,480 soft-band pairs. After mask exclusions, 773 hard-band and 1,474 soft-band edges connect retained images. The v1.1-C grouping policy places all 2,247 retained candidate edges within partitions, irrespective of review labels; 6,360 connected components were assigned. The crossing audit reports zero crossing candidate edges. This is containment **within the detector bands only**; undetected photographic duplicates may still cross partitions. Many soft-band candidates are not grouping-relevant, so grouping can alter class composition.

## Review evidence and limitations

The historical initial review sampled 100 hard-band and 100 soft-band pairs under a binary duplicate/non-duplicate definition. Its labels must not be retrospectively interpreted using the later C1–C5/U taxonomy.

A separate stratified sample of 200 retained soft-band candidate pairs was evaluated using the C1–C5/U taxonomy. The design-weighted estimate of grouping-relevant (C1–C3) candidates is 25.31% (primary 95% interval 19.67–30.96%). The sampling-design CSV provides pair identifiers, original image keys, strata and design weights; final decisions and aggregate estimates are separately recorded. The reviewers' independence rests on declarations and structural checks, not direct observation; adjudication differed from the original procedure. Public release of reviewer-level records remains subject to consent and disclosure review.

An earlier 860-item preliminary screen involved AI assistance and did not constitute independent human review. Its classifications were not used to construct the partition or calculate the 200-pair precision and agreement estimates. **Completed preliminary-screen classification files are not included in the currently verified candidate inventory.** No blank reviewer templates should be presented as completed records.

## Reference evaluation and statistical interpretation

The released per-image evaluation contributions support re-aggregation of metrics without rerunning trained models. The two v1.1-C reference evaluations are single-seed (seed 0) terminal-checkpoint baselines on 1,552 test images:

| Architecture | Disease-only mIoU (points) | Image bootstrap 95% CI (points) |
| --- | ---: | --- |
| SegNeXt MSCAN-L | 33.22 | 30.76–36.13 |
| DeepLabV3+ ResNet-101 | 23.51 | 21.49–26.09 |

The fixed-cohort comparison uses identical test membership across default, sanitized and size-control conditions. H1 (hard-band cross-split candidate prevalence) and H3 (background-inclusive metric difference) met their prespecified criteria. H2 did **not** meet its prespecified causal support criterion; its result is inconclusive, not proof of no effect. The frozen design and results are distinguished in `protocol/Prespecified_Hypotheses_and_Analysis.md`.

The release candidate includes a 20-row `eval/run_index.csv` for 18 historical and two v1.1-C evaluations. The index distinguishes fixed-cohort, v1.0, sensitivity and v1.1-C populations. Some historical per-run hardware or configuration-hash metadata were not established; `not_recorded` must not be interpreted as a measured value.

## Data dictionary and file navigation

The release candidate organizes records under `qc/`, `graph/`, `review/round1/`, `review/validation_v1.1/`, `splits/v1.0/`, `splits/v1.1-C/`, `splits/fixed_cohort/`, `splits/sensitivity/`, `eval/`, `statistics/`, `tables/`, `protocol/`, `code/`, `configs/` and `environment/`. Exact files and checksums are defined by `MANIFEST.sha256`. The source-to-destination mapping is verified separately before publication.

`tables/revision/Supplementary_Table_S1.csv` reports class-wise training, validation and test counts for all 115 disease classes under the original QC-filtered partition, v1.0 and v1.1-C. The original image key preserves the **source** split prefix even when its corrected partition changes.

## Reproduction guidance

1. Download PlantSeg v7 separately and verify the source archive checksum and applicable licences.
2. Verify the deposited records with `sha256sum -c MANIFEST.sha256` from the deposit root.
3. Use the supplied partition lists as authoritative; do not infer a new partition from the original split prefixes.
4. Recalculate reported metrics from the supplied per-image NPZ contributions using the relevant published analysis scripts and frozen bootstrap conventions (10,000 replicates). Do not combine different test cohorts or claim paired inference across v1.0 and v1.1-C.
5. For retraining, obtain separately licensed pretrained weights and configure local dataset, output and scheduler paths. Python 3.11 was used to build the authoritative v1.1-C lists; environment and GPU compatibility must be checked locally.

**Verification status:** The release-candidate files passed a 192/192 SHA-256 check in the private staging directory. This is not a claim that every public reproduction command has been independently executed against the final deposited archive. The final archive must undergo a separate end-to-end reproduction check.

## Limitations

Detector bands do not measure recall; source specimen identities and acquisition sessions are unavailable. Candidate containment can chain unrelated images and alter class composition. The 200-pair validation estimates precision for the retained soft band, not detector recall. Model scores are conditional on trained checkpoints and may vary with hardware or nondeterministic kernels. Only two reference architectures and limited seeds were evaluated. The v1.1-C partition remains provisional.

## Licences, citation and availability

Original PHYLLO software is covered by the MIT licence where the authors own the relevant rights; PlantSeg-derived scientific records are declared CC BY-NC 4.0, subject to original PlantSeg and image-level terms. External dependencies and pretrained weights retain their own terms. See `LICENSE-CODE`, `LICENSE-DATA` and `LICENSE_SCOPE_v1.1-C.md`. Journal compatibility and the rights to distribute each record category must be cleared before public release.

**Corrected release DOI:** PENDING. **Corrected GitHub release tag:** PENDING. Historical deposit: https://doi.org/10.5281/zenodo.23137404. The final citation should use the published Zenodo metadata, not provisional placeholders.

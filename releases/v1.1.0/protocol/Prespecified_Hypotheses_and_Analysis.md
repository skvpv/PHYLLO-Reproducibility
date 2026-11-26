# Prespecified hypotheses and analysis criteria

## Study and provenance

This document describes the prespecified scientific hypotheses and analysis criteria for the PlantSeg image-integrity and segmentation benchmark audit. The study design was locked on **16 September 2026**, before the reported evidence runs. The authoritative frozen scientific block is identified by SHA-256 `594dd656d4e0bcc999c3ecea201928b663d5426806c1ee3092b5fd18c82a201f`. This hash identifies the frozen content under its original hashing convention; it is **not** the SHA-256 of this explanatory document. The criteria were fixed internally and **were not publicly preregistered**.

The original frozen register is preserved in the private study archive. The present document summarizes its scientific content without changing the hypotheses or claiming that subsequent results were known at the time of the lock.

## H1 — Hard-band candidate leakage prevalence

**Frozen statement.** The released v7 PlantSeg split contains cross-split near-duplicate image leakage (test images with at least one near-duplicate in train) at a non-negligible rate.

**Detector and population.** All 7,774 images were screened using 64-bit ImageHash pHash (RGB, hash size 8, high-frequency factor 4) and L2-normalized OpenAI CLIP ViT-B/32 image embeddings with canonical pretrained weights and preprocessing. Candidate pairs were identified by exact blocked comparisons, without approximate nearest-neighbour retrieval. Hard-band edges satisfy pHash Hamming distance <= 3 **or** CLIP cosine similarity >= 0.98. Soft-band edges satisfy pHash distance 4–9 **or** CLIP cosine similarity >= 0.95 and < 0.98, excluding hard-band pairs. Hard-band connected components define primary duplicate clusters. Detector sensitivity was prespecified over pHash thresholds 3, 5, 7 and 9 and CLIP thresholds 0.95, 0.96, 0.97, 0.98 and 0.99.

**Primary outcome and decision rule.** The fraction of the 1,561 original test images having at least one direct hard-band candidate edge to an original training image; the prespecified criterion is >= 1%. Report the deterministic census proportion with a 95% Wilson interval. The blinded initial manual precision audit sampled 100 hard- and 100 soft-band pairs with random seed 20260917, used binary ratings and resolved disagreements by consensus. Detector candidates are not automatically confirmed photographic duplicates.

## H2 — Fixed-cohort training comparison

**Frozen statement.** Leakage-sanitized training (quarantining from train and val every image whose near-duplicate cluster intersects a fixed original v7 test cohort) reduces disease-only mIoU on that identical fixed cohort relative to default training, beyond the effect of a size-matched random-removal control.

**Design.** The fixed test cohort contains 1,557 images, obtained after excluding four inconsistent test masks. Eight inconsistent training masks were excluded uniformly from primary arms. DEFAULT-QC retains the QC-filtered original training and validation pools; SANITIZED additionally removes every train/validation image in a hard-band component touching the fixed test cohort; SIZE-CONTROL removes the same counts from the respective pools using one deterministic class-distribution-matched removal list drawn outside test-intersecting hard components (NumPy PCG64 seed 20260917, L1 class-count matching, lexicographic tie-breaking). The same fixed test images and resampling indices are used across arms. The 1,561-image raw-mask and relabelled-mask evaluations are sensitivity analyses, not primary H2 evidence.

**Outcomes.** Disease-only mIoU is computed over labels 1–115. Define delta_leak = mIoU(DEFAULT-QC) - mIoU(SANITIZED), delta_size = mIoU(DEFAULT-QC) - mIoU(SIZE-CONTROL), and delta_attr = delta_leak - delta_size, in mIoU points.

**Decision rule.** H2 is supported for an architecture only if delta_leak >= 0.5 mIoU points with its 95% paired-bootstrap interval excluding zero, **and** delta_attr > 0 with its 95% paired-bootstrap interval excluding zero. Holm–Bonferroni correction applies across evaluated architectures. At least 10,000 paired image-level bootstrap replicates were prespecified. For the three-seed primary architecture, each replicate uses one shared vector of 1,557 sampled image indices for all arms and seeds, recomputes each seed's mIoU from aggregated per-image contributions, then averages the per-seed contrasts; predictions are not ensembled. The second architecture uses one training seed and the same paired bootstrap logic without seed averaging.

**Interpretation limitation.** Failure to meet this criterion is inconclusive, not evidence of no leakage effect. The design isolates the specified hard-band sanitization contrast under its controls; it does not identify the effect of removing all genuine duplicates.

## H3 — Background-inclusive metric difference

**Frozen statement.** The released benchmark metric (116-label mIoU including background, ignore_index=255 unused) systematically exceeds the disease-only mIoU (labels 1–115); background inclusion inflates the reported aggregate.

**Outcome and decision rule.** For the same DEFAULT-QC predictions on the fixed 1,557-image cohort, delta_bg = mIoU(labels 0–115) - mIoU(labels 1–115). The prespecified criterion is delta_bg > 0 with a 95% paired-bootstrap interval excluding zero for both evaluated architectures. The planned analysis uses at least 10,000 paired image-level bootstrap replicates and Holm–Bonferroni correction across architectures.

## Observed results and interpretation

The manuscript reports that H1 met its threshold, H2 did not meet its specified statistical decision rule, and H3 met its criterion. These are **post-experiment findings**, not entries in the original frozen hypothesis block. Their numerical estimates, confidence intervals, adjusted p-values, and full uncertainty conventions must be verified against the released statistical outputs and cited separately. No frozen hypothesis, threshold, cohort, seed, or analysis decision is revised by this summary.

## Scope and reuse

The corrected candidate-containment partition v1.1-C is a separately identified derived resource; it does not retrospectively change the H1–H3 design or fixed-cohort results. The original source images and masks are not redistributed in this document. Dataset and derived-record licensing must be checked before public release.

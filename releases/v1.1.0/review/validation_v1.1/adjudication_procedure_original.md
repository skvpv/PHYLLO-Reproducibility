# PHYLLO CR-PHYLLO-R3-01: validation-sample rating and adjudication procedure (v2)

**Scope:** the 200-pair stratified validation sample of frozen soft detector candidates
(`validation_packet_CR-PHYLLO-R3-01`).

**Purpose:** to estimate soft-band precision and inter-rater agreement. The sample does not
decide grouping: under Option C every soft candidate is contained whatever its rating.

## 1. Roles
| Role | Who | Must not |
|---|---|---|
| Packet custodian | the researcher who holds `sealed/` | rate pairs; show the key, strata or scores to raters |
| Rater 1, Rater 2 | two different people | use AI tools; consult each other or anyone else; see any AI or AI-assisted label; have taken part in the Round-3 AI-assisted review |
| Adjudicator | a third person; or both raters jointly, recorded as `joint_adjudication: yes` | use AI tools; see AI labels; see the key before adjudication is complete |

**Why prior Round-3 participants are excluded:** 107 of the 200 sampled pairs lie in the frame
from which the Round-3 AI-assisted records were drawn, and 88 cross the v1.0 partition. Anyone
who took part in that round has been exposed to AI-assisted labels for many of these pairs.

## 2. Issue
1. Run the full-image packet on HPC2 with `--expect-key-sha256` and `--expect-sheets-sha256`
   set to the selection-only checksums (§4 of the execution plan). Release only if the gate is
   `CLEAR`, or `CLEAR_WITH_WARNINGS` after reading every warning.
2. Visually spot-check about 10 random review copies for watermarks, credits or URLs.
3. Send each rater only their own `reviewer_N_package/` together with
   `provenance_declaration_TEMPLATE.txt`, and give them the sealed-key SHA-256 for their
   declaration.
4. Keep `sealed/` with the custodian only.
5. Record the issue date.

## 3. Rating
- Raters follow `RUBRIC.md` (C1–C5 or U) and `REVIEWER_ADDENDUM_v1.0.md`.
- One category per pair. Notes are optional and should be the rater's own words.
- Do not reorder, add or delete rows, and do not edit the `image_1` or `image_2` columns.
- **U is a legitimate answer:** use it when identity cannot be decided from the images. It is
  never converted to C4 or C5.
- Time the first 20 ratings, as a check on the 45 s/rating assumption.
- When finished, complete and sign the provenance declaration and return the sheet and the
  declaration to the custodian, not to the other rater.

## 4. Independence check (before any scoring)
Put the two returned sheets and the two declarations in a new folder, then run:

    python3 -I code/check_review_independence.py --packet validation_packet_CR-PHYLLO-R3-01 \
        --returned <returned_dir> --out <returned_dir>_independence

- **FAIL:** do not score. Record the reason. Re-rating needs a new rater and a fresh sheet; the
  failed sheets are archived, not deleted.
- **PASS_WITH_WARNINGS:** the researcher reads each warning and records a decision before
  scoring.
- **INVESTIGATION_REQUIRED (I1: categories identical on 100% of pairs):** perfect agreement is
  not by itself evidence of dependence, so it is neither accepted nor rejected automatically. Do
  not score.
  1. Someone who is not a rater investigates provenance: separate sessions, file timestamps or
     submission e-mails, the declarations, independent notes, and if needed a short interview.
  2. They complete `provenance_investigation_record_TEMPLATE.json`, bound to the SHA-256 of both
     returned sheets.
  3. Rerun the check with `--investigation-record` into a new folder.
     - **CLEARED** → PASS_WITH_WARNINGS (needs your warning acceptance).
     - **NON_INDEPENDENT** → FAIL.
- Copied sheets, the other rater's row order, identical notes and declared AI use or consultation
  FAIL whatever the investigation finds.

## 5. Adjudication
1. Copy the sealed key and both sheets into a scoring folder and run scorer v3 (unchanged)
   with `--scores`. Status `ADJUDICATION_REQUIRED` lists the pairs needing adjudication: those
   where the grouping decisions differ, or where either rater gave U. The list carries no key
   information.
2. The adjudicator sees the two images, the two categories and both raters' notes, but not
   the key, strata, scores or AI labels. They write `adjudication.csv` (`pair_id, category,
   note`) and sign `provenance_declaration_adjudicator.txt`.
3. **Category disagreements with the same grouping decision:** adjudication is optional, and the
   pre-specified default is not to adjudicate them. Kappa is reported for both the grouping
   decision and the category.
4. **Residual U:** if the adjudicator also returns U, the pair stays U, and scorer v3 remains
   `ADJUDICATION_REQUIRED` for those pairs by design.
   - **D-VAL-1 is resolved** by the prespecified estimator `code/estimate_soft_precision.py`
     (`PRESPECIFIED_ESTIMATOR_D-VAL-1.md`).
   - It runs only when every remaining item is residual U.
   - It reports U separately, with a lower bound (U counted as not duplicate) and an upper bound
     (U counted as duplicate). U is never relabelled.
   - Scorer v3 is not modified.
5. Re-run the independence check including `adjudication.csv`, then re-run scorer v3 into a new
   output folder.
6. Run the estimator:

       python3 -I code/estimate_soft_precision.py --packet <packet> --review-dir <scoring folder> \
           --scored-dir <scorer output> --independence-dir <independence output> --out <new folder>

## 6. What may be reported (AC6)
- Provenance class `HUMAN_DOUBLE` only if the independence check passed.
- The D-VAL-1 lower and upper bounds of design-weighted soft-band precision, with intervals and
  the uncertainty envelope, and per-stratum Wilson intervals.
- Agreement per stratum: Cohen's κ, percent agreement and PABAK, for grouping and for category.
- Residual U reported separately by stratum. The resolved-only estimate is labelled as
  conditional on U being missing at random.
- The estimated number of genuine soft duplicates, as 1,474 × precision with its interval.

**Not reportable:**
- a list of "confirmed duplicates removed";
- detector recall;
- pooling with round-1 (binary rubric) or with the AI-assisted Round-3 records.

The flags in the sealed key (`round1_binary_consensus`, `v10_crossing`,
`in_round3_default_target_frame`) support sensitivity analyses only.

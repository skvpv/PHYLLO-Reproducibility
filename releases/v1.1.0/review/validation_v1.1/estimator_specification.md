# D-VAL-1: prespecified estimator for soft-candidate precision and agreement (v1)

**Status:** prespecified on 2026-11-12, **before any rating is collected**. It is implemented as
`code/estimate_soft_precision.py` v1. The tool's SHA-256 is fixed in `VERIFIED_TOOLS.json`, and
the acceptance checker (AC5) accepts only that version.

Any change after ratings are collected needs a change request, and both the original and the
revised analysis must be reported.

## 1. Unit and outcome
- **Unit:** one sampled soft detector candidate pair. There are n = 200 sampled pairs, drawn from
  a frame of N = 1,474 candidate pairs in 4 strata.
- **Outcome:** each pair is classified by the **unchanged scorer v3 decision rule** (imported, not
  re-implemented):

| Outcome | Rule |
|---|---|
| **D** | final grouping decision *grouped* (C1–C3), by rater consensus or by adjudication |
| **N** | final grouping decision *not grouped* (C4/C5) |
| **U** | residual unresolved: a rater gave U, or the raters disagreed on grouping, **and** adjudication also returned U |

**U is never converted into a verified positive or negative.**

**When the estimator may run:** only when every disagreement has been adjudicated. The only items
allowed to remain outstanding are residual U, which scorer v3 keeps in ADJUDICATION_REQUIRED by
design.

## 2. Primary quantities
Notation for stratum h:
- N_h = population size;
- n_h = sample size;
- W_h = N_h / N;
- d_h = number of D pairs;
- u_h = number of residual U pairs;
- m_h = n_h − u_h.

| Quantity | Formula | Status |
|---|---|---|
| Lower bound p_L | Σ W_h d_h / n_h | **primary** (a bound: U counted as not duplicate) |
| Upper bound p_U | Σ W_h (d_h + u_h) / n_h | **primary** (a bound: U counted as duplicate) |
| Resolved-only p_R | Σ W_h d_h / m_h | secondary; assumes U is missing at random within each stratum |
| Uncertainty envelope | [CI_low(p_L), CI_high(p_U)] | reported with the bounds |
| Genuine duplicates among the 1,474 candidates | 1,474 × p (for p_L, p_U, p_R) | point estimate and 95% CI |

**95% confidence intervals:**
- **Primary:** stratified normal approximation with finite-population correction. The variance is
  Σ W_h² (1 − n_h/N_h) p_h(1 − p_h)/(n_h − 1), and the interval is truncated to [0, 1].
- **Per stratum:** Wilson 95%.
- **Sensitivity:** stratified bootstrap percentile interval, B = 10,000, numpy
  `default_rng(20261112)`, resampling pairs within strata.

## 3. Agreement (initial ratings, before adjudication)
Reported overall and per stratum:
- **Grouping decision**, on pairs where neither rater gave U: percent agreement, Cohen's κ and PABAK.
- **Category** (C1–C5 and U, all pairs): percent agreement and Cohen's κ.
- **U rate** for each rater.

Overall κ values also get stratified-bootstrap 95% CIs (same B and seed). Agreement statistics
are unweighted, i.e. computed at sample level.

## 4. Sensitivity subsets (descriptive only)
- Excluding the pairs in the Round-3 default target frame (107 of the sampled pairs), because their
  raters could have seen AI-assisted labels on those pairs.
- Excluding the pairs with a round-1 binary label (9 of the sampled pairs).

## 5. Preconditions (the estimator refuses otherwise)
- Packet status COMPLETE, with the sealed key equal to the packet manifest.
- Independence check PASS or PASS_WITH_WARNINGS. FAIL or INVESTIGATION_REQUIRED blocks estimation.
- Every review-folder file is unchanged since scorer v3 ran. Both sheets equal the sheets that
  passed the independence check.
- The re-derived decisions equal scorer v3's decisions.
- Every pair still outstanding is residual U.

## 6. Not produced, and not to be reported
- Detector recall.
- A list of confirmed duplicates. The per-pair outcome file is internal and holds pair IDs only.
- Any estimate pooled with the round-1 binary labels or the AI-assisted Round-3 records.

## 7. How to report it
- Report p_L and p_U, with their intervals and the envelope, as the primary result.
- State the number of residual U pairs, by stratum.
- If p_R is reported, label it "conditional on U being missing at random".
- Example wording: "Among the 1,474 soft candidates, the design-weighted duplicate fraction lies
  between p_L [CI] and p_U [CI]; u pairs (x%) remained unresolved after adjudication and are
  reported separately."

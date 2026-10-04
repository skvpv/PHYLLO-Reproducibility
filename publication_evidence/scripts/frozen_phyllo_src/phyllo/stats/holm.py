"""Holm-Bonferroni step-down correction (EBSAP S1.4).

Applied across evaluated baselines for H2 and H3. Because the frozen acceptance rule is
expressed via bootstrap CIs excluding 0 (not raw p-values), we provide BOTH:
  * holm_reject: given per-hypothesis p-values, return the step-down reject decisions;
  * a helper to convert a two-sided bootstrap distribution into an achieved-significance
    p-value (proportion of the distribution on the null side, doubled), so the family
    can be corrected consistently when p-values are requested.

The CI-exclusion decision remains the primary frozen rule; Holm is the family-wise
control layered on top exactly as pre-registered.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def bootstrap_two_sided_p(dist: np.ndarray) -> float:
    """Achieved significance level that the true effect is 0, from a bootstrap dist.

    p = 2 * min( P(dist <= 0), P(dist >= 0) ), clipped to [1/(R+1), 1].
    """
    dist = np.asarray(dist, dtype=float)
    r = dist.size
    p_le = (np.sum(dist <= 0) + 1) / (r + 1)
    p_ge = (np.sum(dist >= 0) + 1) / (r + 1)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def holm_reject(pvalues: Dict[str, float], alpha: float = 0.05
                ) -> Dict[str, dict]:
    """Holm-Bonferroni step-down. Returns per-key {p, adjusted_threshold, reject}.

    Keys are hypothesis/comparison labels (e.g. 'H2@SegNeXt-L', 'H2@DeepLabv3+').
    """
    items: List[Tuple[str, float]] = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, dict] = {}
    prior_reject = True
    for rank, (key, p) in enumerate(items):
        thr = alpha / (m - rank)
        reject = prior_reject and (p <= thr)
        prior_reject = reject  # step-down: once we fail, all later fail
        out[key] = {"p": p, "adjusted_threshold": thr, "reject": bool(reject)}
    return out

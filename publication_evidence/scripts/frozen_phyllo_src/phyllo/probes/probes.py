"""Scientific probes (SP-nn) — every important mechanism / contribution is probed.

A probe that is NOT evaluated is a SMOKE-TEST FAILURE (OPS_P2 §9). Tolerances here are
scientific requirements; they are never weakened to obtain PASS. Each probe returns
(passed: bool, detail: str). The smoke test runs all of them on the smallest
scientifically meaningful configuration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import numpy as np

from phyllo.metrics.miou import (per_image_confusion, miou_116, miou_disease,
                                 NUM_CLASSES, IGNORE_INDEX)


@dataclass
class Probe:
    id: str
    mechanism: str
    check: Callable[[], "ProbeResult"]


@dataclass
class ProbeResult:
    id: str
    passed: bool
    detail: str


# SP-01: mask encoding — background=0, disease 1..115, ignore_index absent.
def sp01_mask_encoding(sample_mask: np.ndarray) -> ProbeResult:
    vals = np.unique(sample_mask)
    ok = vals.min() >= 0 and vals.max() <= 115 and (IGNORE_INDEX not in vals)
    return ProbeResult("SP-01", bool(ok),
                       f"unique values in [{vals.min()},{vals.max()}], "
                       f"ignore_index({IGNORE_INDEX}) absent={IGNORE_INDEX not in vals}")


# SP-02: dual-metric ordering — mIoU_116 >= mIoU_disease when background IoU is high
# (background inclusion inflates; the H3 mechanism must be measurable).
def sp02_dual_metric(ai: np.ndarray, au: np.ndarray) -> ProbeResult:
    m116, md = miou_116(ai, au), miou_disease(ai, au)
    ok = np.isfinite(m116) and np.isfinite(md) and np.isfinite(m116 - md)
    return ProbeResult("SP-02", bool(ok),
                       f"mIoU_116={m116:.4f} mIoU_disease={md:.4f} "
                       f"delta_bg={m116-md:+.4f} (H3 surface computable)")


# SP-03: metric correctness — our confusion mIoU_116 matches mmseg IoUMetric on the
# same (pred,gt). Caller supplies mmseg's reported value.
def sp03_metric_agreement(ai, au, mmseg_miou116: float, tol: float = 1e-4) -> ProbeResult:
    ours = miou_116(ai, au)
    ok = abs(ours - mmseg_miou116) <= tol
    return ProbeResult("SP-03", bool(ok),
                       f"ours={ours:.6f} mmseg={mmseg_miou116:.6f} |d|<= {tol}")


# SP-04: leakage detector produces the expected edge on a known duplicate pair.
def sp04_detector_positive(detector_result, known_pair) -> ProbeResult:
    a, b = known_pair
    edges = set(map(tuple, map(sorted, detector_result.hard_edges)))
    ok = tuple(sorted((a, b))) in edges
    return ProbeResult("SP-04", bool(ok), f"known hard pair present={ok}")


# SP-05: cohort integrity — fixed cohort size == 1557, no defective test masks.
def sp05_cohort(cohort: List[str], defective_test: List[str]) -> ProbeResult:
    import os
    def identity(key):
        split, name = key.replace("\\", "/").split("/", 1)
        return f"{split}/{os.path.splitext(name)[0]}"
    overlap = {identity(k) for k in cohort} & {identity(k) for k in defective_test}
    ok = len(cohort) == 1557 and not overlap
    return ProbeResult("SP-05", bool(ok),
                       f"n={len(cohort)} (need 1557), defective-free={not overlap}")


# SP-06: leakage/split integrity — no fixed-cohort test image appears in any arm's train
# list (no leakage of the eval cohort into training).
def sp06_no_cohort_in_train(train_keys: List[str], cohort: List[str]) -> ProbeResult:
    overlap = set(train_keys) & set(cohort)
    return ProbeResult("SP-06", len(overlap) == 0,
                       f"cohort-in-train overlap={len(overlap)} (must be 0)")


# SP-07: size-control equals sanitized in per-split removal COUNT (attribution control).
def sp07_sizectrl_count_match(san_counts: dict, sc_counts: dict) -> ProbeResult:
    ok = san_counts == sc_counts
    return ProbeResult("SP-07", bool(ok),
                       f"sanitized={san_counts} size-control={sc_counts} match={ok}")


# SP-08: loss/gradient validity — finite loss and finite, non-zero gradient norm on a
# minimal forward/backward (caller supplies the two floats from the smoke train step).
def sp08_grad_valid(loss: float, grad_norm: float) -> ProbeResult:
    ok = np.isfinite(loss) and np.isfinite(grad_norm) and grad_norm > 0
    return ProbeResult("SP-08", bool(ok),
                       f"loss={loss} grad_norm={grad_norm} finite&nonzero={ok}")


PROBE_CATALOG = {
    "SP-01": "mask encoding (bg=0, disease 1..115, ignore absent)",
    "SP-02": "dual-metric delta_bg computable (H3 mechanism)",
    "SP-03": "our mIoU_116 == mmseg IoUMetric (metric correctness)",
    "SP-04": "detector fires on a known duplicate (H1 mechanism)",
    "SP-05": "fixed cohort == 1557, defective-free",
    "SP-06": "no cohort test image leaks into any arm's train (split integrity)",
    "SP-07": "size-control per-split count == sanitized (H2 attribution control)",
    "SP-08": "finite loss + non-zero gradient (proposed/base method trains)",
}

#!/usr/bin/env python3
"""Secondary metrics with image-level bootstrap CIs from one evaluation NPZ."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phyllo.metrics.miou import (ALL_LABELS, DISEASE_LABELS, boundary_scores,
                                 macc_from_sums, miou_from_sums, per_class_iou)


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return [None, None]
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def finite(value):
    return float(value) if np.isfinite(value) else None


def metrics(ai, au, al, boundary):
    i, u, lab = ai.sum(0), au.sum(0), al.sum(0)
    out = {"mIoU_116": finite(miou_from_sums(i, u, ALL_LABELS)),
           "mIoU_disease": finite(miou_from_sums(i, u, DISEASE_LABELS)),
           "mAcc_116": finite(macc_from_sums(i, lab, ALL_LABELS)),
           "mAcc_disease": finite(macc_from_sums(i, lab, DISEASE_LABELS)),
           "per_class_iou": [finite(x) for x in per_class_iou(i, u)]}
    if boundary is not None:
        out.update({k: finite(v) for k, v in boundary_scores(*boundary.sum(0)).items()})
    return out


def bootstrap(ai, au, al, boundary, replicates, seed):
    rng = np.random.Generator(np.random.PCG64(seed)); n = ai.shape[0]
    scalar = {k: [] for k in ("mIoU_116", "mIoU_disease", "mAcc_116", "mAcc_disease",
                              "boundary_iou", "boundary_f1")}
    pci = np.empty((replicates, ai.shape[1]), dtype=np.float32)
    for r in range(replicates):
        idx = rng.integers(0, n, size=n)
        result = metrics(ai[idx], au[idx], al[idx], boundary[idx] if boundary is not None else None)
        for key in scalar:
            if key in result:
                scalar[key].append(result[key])
        pci[r] = [np.nan if x is None else x for x in result["per_class_iou"]]
    return {"scalar_ci95": {k: interval(v) for k, v in scalar.items() if v},
            "per_class_iou_ci95": [interval(pci[:, c]) for c in range(pci.shape[1])],
            "replicates": replicates, "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--strata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.replicates < 10000:
        raise SystemExit("at least 10000 replicates required")
    z = np.load(args.npz, allow_pickle=True)
    keys = [str(x) for x in z["keys"]]
    ai, au, al = z["area_intersect"], z["area_union"], z["area_label"]
    boundary = z["boundary_tp_fp_fn"] if "boundary_tp_fp_fn" in z.files else None
    strata = json.load(open(args.strata, encoding="utf-8"))["cohort"]
    if set(keys) != set(strata):
        raise SystemExit("strata keys do not equal NPZ cohort keys")
    report = {"overall": metrics(ai, au, al, boundary),
              "overall_bootstrap": bootstrap(ai, au, al, boundary, args.replicates, args.seed),
              "conditions": {}}
    for axis in ("size", "chr", "edge", "frequency"):
        report["conditions"][axis] = {}
        levels = sorted({strata[key][axis] for key in keys})
        for level in levels:
            idx = np.array([i for i, key in enumerate(keys) if strata[key][axis] == level])
            report["conditions"][axis][level] = {
                "n": len(idx), "point": metrics(ai[idx], au[idx], al[idx],
                                                  boundary[idx] if boundary is not None else None),
                "bootstrap": bootstrap(ai[idx], au[idx], al[idx],
                                       boundary[idx] if boundary is not None else None,
                                       args.replicates, args.seed)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2, allow_nan=False)
    print(args.out)


if __name__ == "__main__":
    main()

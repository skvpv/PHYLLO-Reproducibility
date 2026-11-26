"""Shared, versioned helpers for the PHYLLO-001 Publication_Evidence_Package.

Every metric formula here was cross-validated bit-for-bit (max |dev| <= 2.2e-16) against the
frozen per-run ``*.summary.json`` written by the frozen evaluator (evaluate_arm.py) before this
script was released. Nothing here re-defines a P1 decision; the locked definitions are cited.

Run registry = Runs_Manifest.csv (pre-registered). Raw files live flat in ``raw/`` as
``<RUN-ID>__<family>__<arm>__seed<k>.npz`` (+ ``.summary.json``).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

PROJECT_ID = "PHYLLO-001"
LOCKED_HASH = "sha256:594dd656d4e0bcc999c3ecea201928b663d5426806c1ee3092b5fd18c82a201f"

# run_id -> (family, arm, seed, run_class, compute_profile_id)   [Runs_Manifest.csv]
RUNS = {
    "RUN-002": ("segnext-l", "default", 0, "evidence", "CP-02"),
    "RUN-003": ("segnext-l", "default", 1, "evidence", "CP-02"),
    "RUN-004": ("segnext-l", "default", 2, "evidence", "CP-02"),
    "RUN-005": ("segnext-l", "sanitized", 0, "evidence", "CP-02"),
    "RUN-006": ("segnext-l", "sanitized", 1, "evidence", "CP-02"),
    "RUN-007": ("segnext-l", "sanitized", 2, "evidence", "CP-02"),
    "RUN-008": ("segnext-l", "sizectrl", 0, "evidence", "CP-02"),
    "RUN-009": ("segnext-l", "sizectrl", 1, "evidence", "CP-02"),
    "RUN-010": ("segnext-l", "sizectrl", 2, "evidence", "CP-02"),
    "RUN-011": ("deeplabv3plus-r101", "default", 0, "evidence", "CP-02"),
    "RUN-012": ("deeplabv3plus-r101", "sanitized", 0, "evidence", "CP-02"),
    "RUN-013": ("deeplabv3plus-r101", "sizectrl", 0, "evidence", "CP-02"),
    "RUN-015": ("segnext-l", "corrected", 0, "evidence", "CP-02"),
    "RUN-016": ("deeplabv3plus-r101", "corrected", 0, "evidence", "CP-02"),
    "RUN-017": ("segnext-l", "sens-raw", 0, "sensitivity", "CP-02"),
    "RUN-018": ("segnext-l", "sens-relabel", 0, "sensitivity", "CP-02"),
    "RUN-019": ("deeplabv3plus-r101", "sens-raw", 0, "sensitivity", "CP-02"),
    "RUN-020": ("deeplabv3plus-r101", "sens-relabel", 0, "sensitivity", "CP-02"),
}
BASELINES = {"SegNeXt-MSCAN-L": ("segnext-l", (0, 1, 2)),     # PRIMARY-1
             "DeepLabV3+-R101": ("deeplabv3plus-r101", (0,))}   # PRIMARY-2
NUM_CLASSES = 116
DISEASE = slice(1, 116)   # labels 1..115 (ASR-005 frozen dual metric)


def raw_name(run_id, ext="npz"):
    fam, arm, seed, _, _ = RUNS[run_id]
    return f"{run_id}__{fam}__{arm}__seed{seed}.{ext}"


def run_for(family, arm, seed):
    for rid, (f, a, s, _, _) in RUNS.items():
        if (f, a, s) == (family, arm, seed):
            return rid
    raise KeyError((family, arm, seed))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def load_run(raw_dir, run_id):
    """Load one run's per-image confusion contributions; verify the summary run_id."""
    p = os.path.join(raw_dir, raw_name(run_id))
    s = json.load(open(p.replace(".npz", ".summary.json"), encoding="utf-8"))
    if s.get("run_id") != run_id:
        raise SystemExit(f"FATAL: {p}: summary run_id {s.get('run_id')} != {run_id}")
    d = np.load(p, allow_pickle=False)
    out = {k: d[k].astype(np.int64) for k in ("area_intersect", "area_union", "area_pred", "area_label")}
    out["boundary"] = d["boundary_tp_fp_fn"].astype(np.float64)
    out["keys"] = d["keys"]
    out["summary"] = s
    return out


# ---- metric formulas (verified bit-exact against frozen summaries) -------------------
def _nanmean_ratio(num, den, sl):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(r[sl]))


def metrics_from_sums(I, U, P, L, B):
    tp, fp, fn = B
    return {"mIoU_116": _nanmean_ratio(I, U, slice(0, 116)),
            "mIoU_disease": _nanmean_ratio(I, U, DISEASE),
            "mAcc_116": _nanmean_ratio(I, L, slice(0, 116)),
            "mAcc_disease": _nanmean_ratio(I, L, DISEASE),
            "boundary_iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else float("nan"),
            "boundary_f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")}


def run_metrics(r, idx=None):
    sel = (lambda a: a) if idx is None else (lambda a: a[idx])
    return metrics_from_sums(sel(r["area_intersect"]).sum(0), sel(r["area_union"]).sum(0),
                             sel(r["area_pred"]).sum(0), sel(r["area_label"]).sum(0),
                             sel(r["boundary"]).sum(0))


def per_class_iou(r):
    I = r["area_intersect"].sum(0); U = r["area_union"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(U > 0, I / U, np.nan)


def miou_disease_idx(r, idx):
    return _nanmean_ratio(r["area_intersect"][idx].sum(0), r["area_union"][idx].sum(0), DISEASE)


def miou_116_idx(r, idx):
    return _nanmean_ratio(r["area_intersect"][idx].sum(0), r["area_union"][idx].sum(0), slice(0, 116))


# ---- inference helpers ---------------------------------------------------------------
def percentile_ci(x, level=0.95):
    a = (1 - level) / 2 * 100
    return [float(np.percentile(x, a)), float(np.percentile(x, 100 - a))]


def boot_p_two_sided(x):
    """Two-sided bootstrap p for H0: statistic = 0 (add-one corrected)."""
    x = np.asarray(x); n = len(x)
    return float(min(1.0, 2 * min((np.sum(x <= 0) + 1) / (n + 1), (np.sum(x >= 0) + 1) / (n + 1))))


def holm(pvals, alpha):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, (m - i) * p)
        out[k] = {"p": p, "p_holm": min(1.0, run), "reject": min(1.0, run) <= alpha}
    return out


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return {"k": k, "n": n, "point": p, "lo": (c - m) / d, "hi": (c + m) / d}


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=True)
        fh.write("\n")

#!/usr/bin/env python3
"""PHYLLO — reference-evaluation table values for a corrected partition (v2).

For each evaluation NPZ: disease-only mIoU, image-level and group-level 95% bootstrap intervals,
background-inclusive mIoU, binary-foreground boundary F1, and the non-zero-union class count.

Bootstrap convention = the frozen statistics pipeline (make_statistics.py, PairedStream):
  B = 10,000; numpy default_rng; per replicate rng.integers(0, n, n) -> bincount weights;
  stream seed = 20260917 for the 1,557-image fixed cohort, otherwise 20260917 + n
  (e.g. 20262469 for the 1,552-image v1.0 corrected test set). Percentile interval (2.5, 97.5).
  Group-level intervals resample whole groups (manifest gid) with a stream seeded the same way;
  this is a sensitivity summary added in the revision, not part of the frozen pipeline.

Checks: the build is BUILT/BUILT_DRY_RUN with intact checksums (or a frozen v1.0 manifest with
--frozen-v10); NPZ keys equal the partition's test list exactly; arrays have the frozen shape.

Usage:
  python3 -I v11_reference_table.py --repo <repo> (--build-dir <builder out> | --frozen-v10)
      --npz <file> [--npz <file>] --out <new .json>
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np

BASE_SEED, FIXED_N, B = 20260917, 1557, 10000


class TableError(Exception):
    pass


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stream_seed(n):
    return BASE_SEED if n == FIXED_N else BASE_SEED + n


def miou(I, U, sl):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(U > 0, I / U, np.nan)
    return np.nanmean(r[..., sl], axis=-1)


def load_partition(a):
    if a.frozen_v10:
        p = Path(a.repo, "publication_evidence/raw/RUN-014__corrected_split_manifest.json")
        listed = {}
        for line in Path(a.repo, "MANIFEST.sha256").read_text().splitlines():
            if line.strip():
                h, q = line.split(None, 1); listed[os.path.normpath(q.strip())] = h
        if listed.get(os.path.normpath("publication_evidence/raw/RUN-014__corrected_split_manifest.json")) != sha256(p):
            raise TableError("frozen v1.0 manifest not verified by MANIFEST.sha256")
        return json.loads(p.read_text())["assignments"], {"partition": "v1.0 (frozen RUN-014)", "manifest_sha256": sha256(p)}
    bd = Path(a.build_dir)
    st = json.loads((bd / "BUILD_STATUS.json").read_text())
    if st.get("status") not in ("BUILT", "BUILT_DRY_RUN"):
        raise TableError(f"build status {st.get('status')!r}")
    for n, h in st["artifact_sha256"].items():
        if sha256(bd / n) != h:
            raise TableError(f"{n} does not match BUILD_STATUS.json")
    m = json.loads((bd / "corrected_split_manifest_v1.1.json").read_text())
    return m["assignments"], {"partition": m.get("version"), "build_status": st["status"],
                              "manifest_sha256": st["artifact_sha256"]["corrected_split_manifest_v1.1.json"]}


def evaluate(npz, asg):
    z = np.load(npz, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    test = sorted(k for k, v in asg.items() if v["split"] == "test")
    if sorted(keys) != test or len(keys) != len(set(keys)):
        raise TableError(f"{Path(npz).name}: keys are not exactly the partition's test list "
                         f"({len(keys)} keys vs {len(test)} test images)")
    ai, au = z["area_intersect"].astype(np.float64), z["area_union"].astype(np.float64)
    if ai.shape != (len(keys), 116) or au.shape != ai.shape or z["boundary_tp_fp_fn"].shape != (len(keys), 3):
        raise TableError(f"{Path(npz).name}: unexpected array shapes {ai.shape}, {z['boundary_tp_fp_fn'].shape}")
    n = len(keys); seed = stream_seed(n)
    rng = np.random.default_rng(seed); img = np.empty(B)
    for b in range(B):
        c = np.bincount(rng.integers(0, n, n), minlength=n)
        img[b] = miou(c @ ai, c @ au, slice(1, 116))
    gid = {}
    for i, k in enumerate(keys):
        gid.setdefault(asg[k]["gid"], []).append(i)
    groups = [np.array(v) for v in gid.values()]; G = len(groups)
    rng = np.random.default_rng(seed); grp = np.empty(B)
    for b in range(B):
        c = np.bincount(np.concatenate([groups[j] for j in rng.integers(0, G, G)]), minlength=n)
        grp[b] = miou(c @ ai, c @ au, slice(1, 116))
    bd = z["boundary_tp_fp_fn"].sum(0).astype(np.float64)
    pct = lambda d: [float(100 * x) for x in np.percentile(d, [2.5, 97.5])]
    I, U = ai.sum(0), au.sum(0)
    ic, gc = pct(img), pct(grp)
    return {"file": Path(npz).name, "npz_sha256": sha256(npz), "n_test": n, "n_groups": G, "stream_seed": seed,
            "disease_mIoU_pts": float(100 * miou(I, U, slice(1, 116))), "image_ci95_pts": ic, "group_ci95_pts": gc,
            "max_ci_change_pts": max(abs(ic[0] - gc[0]), abs(ic[1] - gc[1])),
            "inclusive_mIoU_pts": float(100 * miou(I, U, slice(0, 116))),
            "boundary_f1_pts": float(100 * 2 * bd[0] / (2 * bd[0] + bd[1] + bd[2])),
            "nonzero_union_disease_classes": int((U[1:] > 0).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-dir"); g.add_argument("--frozen-v10", action="store_true")
    ap.add_argument("--npz", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        print(f"REFUSED: {a.out} exists", file=sys.stderr); return 2
    try:
        asg, meta = load_partition(a)
        rows = [evaluate(p, asg) for p in a.npz]
    except (TableError, KeyError, FileNotFoundError) as e:
        print(f"REFUSED: {e}", file=sys.stderr); return 2
    res = {**meta, "bootstrap": {"B": B, "base_seed": BASE_SEED, "rule": "seed = base if n == 1557 else base + n"},
           "script_sha256": sha256(__file__), "rows": rows}
    tmp = Path(str(a.out) + ".incomplete"); tmp.write_text(json.dumps(res, indent=1)); os.rename(tmp, a.out)
    print(json.dumps(rows, indent=1)); return 0


if __name__ == "__main__":
    sys.exit(main())

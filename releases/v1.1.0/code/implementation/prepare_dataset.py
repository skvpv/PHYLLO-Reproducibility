#!/usr/bin/env python3
"""Validate PlantSeg-v7 and emit non-evidence QC lists required by preflight."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phyllo.dataset_prep.prepare import (
    ARCHIVE_MD5, DEFECTIVE, N_TOTAL, SPLIT_COUNTS, build_fixed_cohort,
    default_qc_train_pool, discover_pairs, validate_masks, verify_archive,
    write_list, write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--out-dir", default="data_prep")
    parser.add_argument("--img-suffix", default=".jpg")
    parser.add_argument("--mask-suffix", default=".png")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    archive = verify_archive(args.archive)
    if not archive["match"]:
        raise SystemExit(f"archive MD5 mismatch: {archive}")
    splits, rows = discover_pairs(args.img_root, args.ann_root,
                                  args.img_suffix, args.mask_suffix)
    masks = validate_masks(rows)
    if not masks["valid"] or not masks["ignore_index_absent"]:
        raise SystemExit(f"mask validation failed: {masks['invalid'][:5]}")
    lists = {
        "train_default_qc.txt": default_qc_train_pool(splits["train"]),
        "val_default_qc.txt": sorted(splits["val"]),
        "cohort_1557.txt": build_fixed_cohort(splits["test"]),
        "test_raw_1561.txt": sorted(splits["test"]),
    }
    checksums = {name: write_list(os.path.join(args.out_dir, name), keys)
                 for name, keys in lists.items()}
    manifest = {
        "schema_version": "1.0", "project_id": "PHYLLO-001",
        "dataset": "PlantSeg v7", "archive_md5": ARCHIVE_MD5,
        "archive_verification": archive, "split_counts": SPLIT_COUNTS,
        "total": N_TOTAL, "mask_scan": masks, "defective_masks": DEFECTIVE,
        "list_counts": {k: len(v) for k, v in lists.items()},
        "list_checksums": checksums,
        "policy": "primary excludes 8 defective train and 4 defective test; val unchanged",
    }
    checksum = write_manifest(os.path.join(args.out_dir, "dataset_manifest.json"), manifest)
    report = {"dataset_prep_pass": True, "dataset_manifest_checksum": checksum,
              "list_counts": manifest["list_counts"], "archive": archive}
    write_manifest(os.path.join(args.out_dir, "dataset_report.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

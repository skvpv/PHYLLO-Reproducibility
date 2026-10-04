#!/usr/bin/env python3
"""Build leakage-dependent train/validation arm lists after R-LEAK-DETECT.

The fixed cohort and DEFAULT-QC lists come from ``prepare_dataset.py``.  This script
never recreates them and never uses test images as validation data.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phyllo.dataset_prep.prepare import (DEFECTIVE_TRAIN, DEFECTIVE_TEST,
    canonical_id, file_digest, load_metadata_class, write_list, write_manifest)
from phyllo.sanitize.arm_manifests import build_arm_manifests

def read_list(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="hard_clusters.json from R-LEAK-DETECT")
    ap.add_argument("--metadata", required=True, help="plantseg/Metadata.csv")
    ap.add_argument("--train-default", required=True)
    ap.add_argument("--val-default", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out-dir", default="arms"); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    qc_train = read_list(a.train_default)
    qc_val = read_list(a.val_default)
    cohort = read_list(a.cohort)
    if len(qc_train) != 5359 or len(qc_val) != 846 or len(cohort) != 1557:
        raise SystemExit("input QC-list counts are not 5359/846/1557")
    clusters = json.load(open(a.clusters))
    image_class = load_metadata_class(a.metadata)
    # Metadata may contain .png names; resolve every image key by canonical identity.
    image_class = {k: image_class.get(k, image_class.get(canonical_id(k)))
                   for k in qc_train + qc_val}
    if any(v is None for v in image_class.values()):
        missing = [k for k, v in image_class.items() if v is None]
        raise SystemExit(f"metadata class missing for {missing[:5]}")
    pool = {"train": set(qc_train), "val": set(qc_val)}
    cohort_set = set(cohort)
    test_int = set(k for comp in clusters if any(x in cohort_set for x in comp)
                   for k in comp if k.startswith(("train/", "val/")))
    m = build_arm_manifests(clusters, set(cohort), pool, test_int, image_class)
    lists = {
        "train_default_qc.txt": qc_train,
        "val_default_qc.txt": qc_val,
        "train_sanitized.txt": sorted(set(qc_train) - set(m.sanitized_remove["train"])),
        "val_sanitized.txt": sorted(set(qc_val) - set(m.sanitized_remove["val"])),
        "train_sizectrl.txt": sorted(set(qc_train) - set(m.sizectrl_remove["train"])),
        "val_sizectrl.txt": sorted(set(qc_val) - set(m.sizectrl_remove["val"])),
    }
    checksums = {}
    for name, items in lists.items():
        path = os.path.join(a.out_dir, name)
        checksums[name] = write_list(path, items)
        print("wrote", path, len(items))
    ck = write_manifest(os.path.join(a.out_dir, "arm_manifest.json"), {
        "compute_profile_id": "CP-02",
        "cohort_n": len(cohort), "defective_train": DEFECTIVE_TRAIN,
        "defective_test": DEFECTIVE_TEST,
        "cohort_source": os.path.abspath(a.cohort),
        "cohort_checksum": "sha256:" + file_digest(a.cohort),
        "sanitized_remove": m.sanitized_remove, "sizectrl_remove": m.sizectrl_remove,
        "sanitized_counts": m.sanitized_counts, "sizectrl_counts": m.sizectrl_counts,
        "list_checksums": checksums})
    print("arm_manifest checksum", ck)
if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Create frozen raw-test and metadata-consistency relabel sensitivity assets."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phyllo.dataset_prep.prepare import DEFECTIVE, canonical_id, file_digest, write_list, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-test-list", required=True)
    parser.add_argument("--out-dir", default="sensitivity")
    args = parser.parse_args()
    keys = [x.strip() for x in open(args.raw_test_list, encoding="utf-8") if x.strip()]
    if len(keys) != 1561:
        raise SystemExit(f"raw test list must contain 1561 keys, got {len(keys)}")
    by_id = {canonical_id(k): k for k in keys}
    mapping = {}
    for mask_key, (_, expected_value, _) in DEFECTIVE.items():
        if mask_key.startswith("test/"):
            image_key = by_id.get(canonical_id(mask_key))
            if image_key is None:
                raise SystemExit(f"defective test identity absent: {mask_key}")
            mapping[image_key] = expected_value
    if len(mapping) != 4:
        raise SystemExit("expected four relabelled test masks")
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "test_raw_1561.txt")
    raw_hash = write_list(raw_path, keys)
    relabel_path = os.path.join(args.out_dir, "relabel_map.json")
    with open(relabel_path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)
    write_manifest(os.path.join(args.out_dir, "sensitivity_manifest.json"), {
        "compute_profile_id": "CP-02", "raw_cohort_n": 1561,
        "relabelled_mask_n": 4, "raw_list_checksum": raw_hash,
        "relabel_map_checksum": "sha256:" + file_digest(relabel_path),
        "restriction": "sensitivity only; cannot support H1, H2, or H3",
        "definition": "replace every nonzero pixel of each defective test mask with Metadata Index+1"})
    print(json.dumps({"raw_n": len(keys), "relabel_n": len(mapping)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the defect-free component-confined corrected PlantSeg split."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phyllo.correct_split.corrected_benchmark import Group, assign_corrected_split
from phyllo.dataset_prep.prepare import DEFECTIVE_IDS, canonical_id, discover_pairs, write_list


def mask_classes(path):
    return {int(x) for x in np.unique(np.asarray(Image.open(path))) if 1 <= int(x) <= 115}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--out-dir", default="corrected")
    args = parser.parse_args()
    _, rows = discover_pairs(args.img_root, args.ann_root)
    active = {row["key"]: row for row in rows if canonical_id(row["key"]) not in DEFECTIVE_IDS}
    clusters = json.load(open(args.clusters, encoding="utf-8"))
    used, groups = set(), []
    for component in clusters:
        members = sorted(set(component) & set(active))
        if len(members) < 2:
            continue
        classes = set().union(*(mask_classes(active[k]["mask"]) for k in members))
        groups.append(Group(len(groups), members, classes)); used.update(members)
    for key in sorted(set(active) - used):
        groups.append(Group(len(groups), [key], mask_classes(active[key]["mask"])))
    result = assign_corrected_split(groups, diagnostics=lambda row: print(json.dumps({"run014_progress": row}, sort_keys=True), file=sys.stderr, flush=True))
    result["run_id"] = "RUN-014"
    result["compute_profile_id"] = "CP-02"
    if result["total_images"] != 7762 or not result["constraints_satisfied"]:
        raise SystemExit(f"corrected split invalid: total={result['total_images']} "
                         f"infeasible={result['infeasible_classes']}")
    os.makedirs(args.out_dir, exist_ok=True)
    list_hashes = {}
    for split in ("train", "val", "test"):
        keys = sorted(k for k, row in result["assignments"].items() if row["split"] == split)
        list_hashes[split] = write_list(os.path.join(args.out_dir, f"{split}_corrected.txt"), keys)
    result["list_checksums"] = list_hashes
    blob = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    result["manifest_sha256_pre_checksum"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    with open(Path(args.out_dir, "corrected_split_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"counts": result["split_image_counts"], "n_groups": result["n_groups"],
                      "constraints_satisfied": True}, indent=2))


if __name__ == "__main__":
    main()

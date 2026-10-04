#!/usr/bin/env python3
"""R-LEAK-DETECT: exact frozen census, ablations, and blinded manual-audit assets."""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phyllo.dataset_prep.prepare import canonical_id, load_metadata_class
from phyllo.leakage.detector import detect, sensitivity_census
from phyllo.stats.wilson import wilson_interval


def enumerate_images(img_root: str, suffix: str):
    keys, paths, splits = [], [], []
    for split in ("train", "val", "test"):
        for path in sorted(Path(img_root, split).glob(f"*{suffix}")):
            keys.append(f"{split}/{path.name}"); paths.append(str(path)); splits.append(split)
    if len(keys) != 7774 or collections.Counter(splits) != {"train": 5367, "val": 846, "test": 1561}:
        raise ValueError(f"unexpected frozen-census counts: {collections.Counter(splits)}")
    return keys, paths, splits


def method_census(keys, hashes, embeddings, splits):
    methods = {
        "phash_only": detect(keys, hashes, embeddings, splits, phash_hard=3, clip_hard=2.0),
        "clip_only": detect(keys, hashes, embeddings, splits, phash_hard=-1, clip_hard=.98),
        "union": detect(keys, hashes, embeddings, splits),
    }
    return {name: {"n_leaked_test": len(result.leaked_test_keys),
                   "n_test": result.n_test, "leakage_rate": result.leakage_rate}
            for name, result in methods.items()}


def blind_audit(edges_by_kind, key_to_path, out_dir, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = []
    for kind, edges in edges_by_kind.items():
        indices = np.arange(len(edges))
        if len(indices) > 100:
            indices = rng.choice(indices, 100, replace=False)
        for index in sorted(int(i) for i in indices):
            sampled.append((kind, *edges[index]))
    rng.shuffle(sampled)
    assets = Path(out_dir, "manual_audit_assets")
    assets.mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    for number, (kind, a, b) in enumerate(sampled, 1):
        pair_id = f"PAIR-{number:04d}"
        sides = [("A", a), ("B", b)]
        if int(rng.integers(0, 2)):
            sides.reverse()
        blind_files = []
        mapping = {}
        for side_no, (_, source_key) in enumerate(sides, 1):
            extension = Path(key_to_path[source_key]).suffix.lower()
            blind_name = f"{pair_id}_{side_no}{extension}"
            shutil.copy2(key_to_path[source_key], assets / blind_name)
            blind_files.append(f"manual_audit_assets/{blind_name}")
            mapping[str(side_no)] = source_key
        sheet.append({"pair_id": pair_id, "image_1": blind_files[0],
                      "image_2": blind_files[1],
                      "reviewer_1_verdict": "", "reviewer_2_verdict": "",
                      "consensus_verdict": "", "reviewer_notes": ""})
        key.append({"pair_id": pair_id, "candidate_kind": kind, "mapping": mapping})
    with open(Path(out_dir, "manual_audit_REVIEWER.json"), "w", encoding="utf-8") as handle:
        json.dump(sheet, handle, indent=2)
    with open(Path(out_dir, "manual_audit_KEY.json"), "w", encoding="utf-8") as handle:
        json.dump(key, handle, indent=2)
    return len(sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--img-suffix", default=".jpg")
    parser.add_argument("--out-dir", default="results/R-LEAK-DETECT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--emb-cache", default="")
    parser.add_argument("--audit-seed", type=int, default=20260917)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    discovered_keys, paths, discovered_splits = enumerate_images(args.img_root, args.img_suffix)
    key_to_path = dict(zip(discovered_keys, paths))
    if args.emb_cache:
        z = np.load(args.emb_cache, allow_pickle=True)
        keys = [str(x) for x in z["keys"]]; splits = [str(x) for x in z["splits"]]
        if keys != discovered_keys or splits != discovered_splits:
            raise ValueError("embedding cache key/order does not match exact dataset census")
        hashes = z["phash_u64"].astype(np.uint64)
        embeddings = z["clip_emb"].astype(np.float32)
        provenance = json.loads(str(z["provenance"]))
    else:
        from phyllo.leakage.phash_clip import compute_phash_clip
        keys, splits = discovered_keys, discovered_splits
        hashes, embeddings, provenance = compute_phash_clip(paths, keys, device=args.device)
        np.savez_compressed(Path(args.out_dir, "embeddings.npz"), keys=np.array(keys),
                            splits=np.array(splits), phash_u64=hashes,
                            clip_emb=embeddings, provenance=json.dumps(provenance))

    result = detect(keys, hashes, embeddings, splits)
    n_leaked = len(result.leaked_test_keys)
    ci = wilson_interval(n_leaked, result.n_test)
    grid = sensitivity_census(keys, hashes, embeddings, splits)
    ablation = method_census(keys, hashes, embeddings, splits)

    # Secondary descriptive censuses.
    train_edges = [(a, b) for a, b in result.hard_edges
                   if a.startswith("train/") and b.startswith("train/")]
    train_images = {x for edge in train_edges for x in edge}
    metadata = load_metadata_class(args.metadata)
    leaked_by_class = collections.Counter()
    test_by_class = collections.Counter()
    leaked_set = set(result.leaked_test_keys)
    for key in (x for x in keys if x.startswith("test/")):
        class_id = metadata.get(key, metadata.get(canonical_id(key)))
        test_by_class[class_id] += 1
        if key in leaked_set:
            leaked_by_class[class_id] += 1
    per_class = {str(c): {"n_test": test_by_class[c], "n_leaked": leaked_by_class[c],
                          "rate": leaked_by_class[c] / test_by_class[c]}
                 for c in sorted(x for x in test_by_class if x is not None)}
    audit_n = blind_audit({"hard": result.hard_edges, "soft": result.soft_edges},
                          key_to_path, args.out_dir, args.audit_seed)

    summary = {
        "run_id": "RUN-001", "run_family": "R-LEAK-DETECT",
        "compute_profile_id": "CP-01", "n_images": len(keys), "n_test": result.n_test,
        "n_leaked_test": n_leaked, "leakage_rate": result.leakage_rate,
        "leakage_rate_wilson95": {"lo": ci.lo, "hi": ci.hi},
        "n_hard_edges": len(result.hard_edges), "n_soft_edges": len(result.soft_edges),
        "n_hard_clusters": len(result.clusters), "detector_ablation": ablation,
        "intra_train": {"n_hard_edges": len(train_edges),
                        "n_affected_images": len(train_images),
                        "affected_rate": len(train_images) / 5367},
        "per_class_test_leakage": per_class, "manual_audit_n": audit_n,
        "provenance": provenance,
    }
    outputs = {
        "leak_detect_summary.json": summary,
        "edges.json": {"hard_edges": result.hard_edges, "soft_edges": result.soft_edges},
        "hard_clusters.json": result.clusters,
        "leaked_test_keys.json": result.leaked_test_keys,
        "sensitivity_surface.json": grid,
        "detector_ablation.json": ablation,
    }
    for name, value in outputs.items():
        with open(Path(args.out_dir, name), "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

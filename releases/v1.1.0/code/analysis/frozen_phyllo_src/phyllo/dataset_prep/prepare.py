"""PlantSeg-v7 validation and frozen QC-list construction.

All public keys are image keys (``split/name.jpg``). Defect evidence names masks
(``split/name.png``), so comparisons use the canonical ``split/stem`` identity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

ARCHIVE_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
SPLIT_COUNTS = {"train": 5367, "val": 846, "test": 1561}
N_TOTAL = 7774
NUM_CLASSES = 116
IGNORE_INDEX = 255

DEFECTIVE = {
    "test/broccoli_alternaria_leaf_spot_Bing_0021.png": (23, 24, 25),
    "test/broccoli_alternaria_leaf_spot_Bing_0045.png": (23, 24, 25),
    "test/broccoli_alternaria_leaf_spot_Bing_0054.png": (23, 24, 25),
    "test/wheat_head_scab_Bing_0287.png": (104, 105, 106),
    "train/bean_mosaic_virus_23.png": (12, 13, 82),
    "train/blueberry_mummy_berry_Bing_0058.png": (20, 21, 22),
    "train/broccoli_alternaria_leaf_spot_Bing_0012.png": (23, 24, 25),
    "train/broccoli_alternaria_leaf_spot_Bing_0035.png": (23, 24, 25),
    "train/broccoli_alternaria_leaf_spot_Bing_0039.png": (23, 24, 25),
    "train/cucumber_powdery_mildew_google_0067.png": (50, 51, 49),
    "train/eggplant_phomopsis_fruit_rot_Bing_0015.png": (52, 53, 54),
    "train/ginger_leaf_spot_17.png": (56, 57, 58),
}


def canonical_id(key: str) -> str:
    split, name = key.replace("\\", "/").split("/", 1)
    return f"{split.lower()}/{Path(name).stem}"


DEFECTIVE_IDS = frozenset(canonical_id(k) for k in DEFECTIVE)
DEFECTIVE_TRAIN_IDS = frozenset(k for k in DEFECTIVE_IDS if k.startswith("train/"))
DEFECTIVE_TEST_IDS = frozenset(k for k in DEFECTIVE_IDS if k.startswith("test/"))
DEFECTIVE_TRAIN = sorted(k for k in DEFECTIVE if k.startswith("train/"))
DEFECTIVE_TEST = sorted(k for k in DEFECTIVE if k.startswith("test/"))
assert len(DEFECTIVE_TRAIN_IDS) == 8 and len(DEFECTIVE_TEST_IDS) == 4


def file_digest(path: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: str) -> str:
    return file_digest(path, "md5")


def verify_archive(archive_path: str) -> Dict[str, object]:
    got = md5_of(archive_path)
    return {"expected_md5": ARCHIVE_MD5, "actual_md5": got,
            "match": got == ARCHIVE_MD5}


def is_defective(key: str) -> bool:
    return canonical_id(key) in DEFECTIVE_IDS


def _exclude_ids(keys: Iterable[str], bad_ids: Iterable[str]) -> List[str]:
    bad = set(bad_ids)
    return sorted(k for k in keys if canonical_id(k) not in bad)


def build_fixed_cohort(test_keys: List[str]) -> List[str]:
    cohort = _exclude_ids(test_keys, DEFECTIVE_TEST_IDS)
    if len(test_keys) != SPLIT_COUNTS["test"] or len(cohort) != 1557:
        raise ValueError(
            f"expected 1561 test keys and 1557 after four exclusions; got "
            f"{len(test_keys)} and {len(cohort)}")
    return cohort


def default_qc_train_pool(train_keys: List[str]) -> List[str]:
    pool = _exclude_ids(train_keys, DEFECTIVE_TRAIN_IDS)
    if len(train_keys) != SPLIT_COUNTS["train"] or len(pool) != 5359:
        raise ValueError(
            f"expected 5367 train keys and 5359 after eight exclusions; got "
            f"{len(train_keys)} and {len(pool)}")
    return pool


def write_manifest(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(blob)
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def write_list(path: str, keys: Iterable[str]) -> str:
    items = list(keys)
    if len(items) != len(set(items)):
        raise ValueError(f"duplicate keys in {path}")
    blob = ("\n".join(items) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(blob)
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def discover_pairs(img_root: str, ann_root: str, img_suffix: str = ".jpg",
                   mask_suffix: str = ".png") -> Tuple[Dict[str, List[str]], List[dict]]:
    """Discover exact image/mask pairs and reject extras or missing counterparts."""
    split_keys: Dict[str, List[str]] = {}
    rows: List[dict] = []
    for split, expected in SPLIT_COUNTS.items():
        images = sorted(Path(img_root, split).glob(f"*{img_suffix}"))
        masks = sorted(Path(ann_root, split).glob(f"*{mask_suffix}"))
        image_by_stem = {p.stem: p for p in images}
        mask_by_stem = {p.stem: p for p in masks}
        if len(image_by_stem) != len(images) or len(mask_by_stem) != len(masks):
            raise ValueError(f"duplicate stems in {split}")
        missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
        missing_images = sorted(set(mask_by_stem) - set(image_by_stem))
        if missing_masks or missing_images:
            raise ValueError(f"unpaired files in {split}: masks_missing={missing_masks[:5]} "
                             f"images_missing={missing_images[:5]}")
        if len(images) != expected:
            raise ValueError(f"{split} count {len(images)} != frozen {expected}")
        keys = [f"{split}/{p.name}" for p in images]
        split_keys[split] = keys
        for key, image in zip(keys, images):
            mask = mask_by_stem[image.stem]
            rows.append({"key": key, "canonical_id": canonical_id(key),
                         "image": str(image.resolve()), "mask": str(mask.resolve()),
                         "defective": is_defective(key), "split": split})
    if sum(map(len, split_keys.values())) != N_TOTAL:
        raise ValueError("total pair count does not equal 7774")
    return split_keys, rows


def validate_masks(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Read every mask; enforce 2-D integer values in 0..115 and absence of 255."""
    from PIL import Image
    import numpy as np

    seen = set()
    invalid = []
    count = 0
    for row in rows:
        arr = np.asarray(Image.open(str(row["mask"])))
        count += 1
        vals = np.unique(arr)
        seen.update(int(v) for v in vals)
        ok = (arr.ndim == 2 and vals.size and vals.min() >= 0 and
              vals.max() <= 115 and IGNORE_INDEX not in vals)
        if not ok:
            invalid.append({"key": row["key"], "shape": list(arr.shape),
                            "min": int(vals.min()), "max": int(vals.max())})
    return {"n_scanned": count, "valid": not invalid, "invalid": invalid,
            "observed_values": sorted(seen), "ignore_index_absent": IGNORE_INDEX not in seen}


def load_metadata_class(metadata_csv: str) -> Dict[str, int]:
    """Map both filename keys and canonical identities to Metadata Index + 1."""
    split_norm = {"training": "train", "validation": "val", "test": "test"}
    out: Dict[str, int] = {}
    with open(metadata_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            split_raw = row.get("Split", "").strip().lower()
            split = split_norm.get(split_raw, split_raw)
            name = row.get("Name", "").strip()
            value = int(row["Index"]) + 1
            key = f"{split}/{name}"
            out[key] = value
            out[canonical_id(key)] = value
    return out

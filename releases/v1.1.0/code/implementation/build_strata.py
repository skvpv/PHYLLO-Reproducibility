#!/usr/bin/env python3
"""Build deterministic size/CHR/edge/frequency strata from masks.

Size and CHR tertile cut points and class-frequency tertiles are fitted on the frozen
DEFAULT-QC training pool only, then applied unchanged to the fixed test cohort.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read(path):
    return [x.strip() for x in open(path, encoding="utf-8") if x.strip()]


def features(key, ann_root):
    split, name = key.split("/", 1)
    mask = np.asarray(Image.open(Path(ann_root, split, Path(name).stem + ".png")))
    foreground = (mask > 0).astype(np.uint8)
    area = int(foreground.sum()); ratio = area / foreground.size
    points = cv2.findNonZero(foreground)
    hull_area = float(cv2.contourArea(cv2.convexHull(points))) if points is not None and len(points) >= 3 else float(area)
    chr_value = float(area / hull_area) if hull_area > 0 else 0.0
    edge = bool(foreground[0].any() or foreground[-1].any() or
                foreground[:, 0].any() or foreground[:, -1].any())
    classes, counts = np.unique(mask[(mask >= 1) & (mask <= 115)], return_counts=True)
    return {"size_ratio": ratio, "chr": chr_value, "edge": edge,
            "classes": [int(x) for x in classes],
            "class_pixels": {str(int(c)): int(n) for c, n in zip(classes, counts)}}


def bin3(value, cuts):
    return ("low", "mid", "high")[int(value > cuts[0]) + int(value > cuts[1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--cohort-list", required=True)
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--out", default="data_prep/strata.json")
    args = parser.parse_args()
    train = {key: features(key, args.ann_root) for key in read(args.train_list)}
    cohort = {key: features(key, args.ann_root) for key in read(args.cohort_list)}
    size_cuts = np.quantile([v["size_ratio"] for v in train.values()], [1/3, 2/3]).tolist()
    chr_cuts = np.quantile([v["chr"] for v in train.values()], [1/3, 2/3]).tolist()
    frequency = {c: 0 for c in range(1, 116)}
    for row in train.values():
        for c, count in row["class_pixels"].items():
            frequency[int(c)] += count
    positive = [x for x in frequency.values() if x > 0]
    freq_cuts = np.quantile(positive, [1/3, 2/3]).tolist()
    rows = {}
    for key, row in cohort.items():
        class_freqs = [frequency[c] for c in row["classes"]] or [0]
        representative_frequency = min(class_freqs)
        rows[key] = {"size": bin3(row["size_ratio"], size_cuts),
                     "chr": bin3(row["chr"], chr_cuts),
                     "edge": "edge" if row["edge"] else "interior",
                     "frequency": bin3(representative_frequency, freq_cuts),
                     **row}
    payload = {"definitions": {"fit_partition": "DEFAULT-QC train only",
                 "size": "foreground-pixel ratio tertiles", "size_cuts": size_cuts,
                 "chr": "foreground area / convex-hull area tertiles", "chr_cuts": chr_cuts,
                 "edge": "foreground touches image border",
                 "frequency": "rarest present disease class by train pixel frequency tertiles",
                 "frequency_cuts": freq_cuts}, "cohort": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2)
    print(args.out, len(rows))


if __name__ == "__main__":
    main()

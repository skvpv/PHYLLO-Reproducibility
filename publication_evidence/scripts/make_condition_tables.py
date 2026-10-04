#!/usr/bin/env python3
"""PHYLLO-001 — condition-stratified secondary table (versioned, deterministic, descriptive).

Re-run from the package root:  python scripts/make_condition_tables.py --config configs/condition_tables.json
Reads only statistics/condition_secondary__*.json, the outputs of the frozen driver
scripts/frozen_drivers/run_secondary_metrics.py on strata from build_strata (CODE_DEFECT CD-001
correction). Reports each run x axis x level: n, mIoU_disease and mIoU_116 point values with
95% image-level bootstrap CIs, and boundary F1. No new inference; secondary and descriptive only.
"""
import argparse
import csv
import json
import os

P = 100.0


def fmt(x):
    return "" if x is None else "%.6f" % (x * P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pep-root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.pep_root)
    cfg = json.load(open(os.path.join(root, a.config), encoding="utf-8"))
    sd = os.path.join(root, cfg["stats_dir"])
    files = sorted(f for f in os.listdir(sd) if f.startswith("condition_secondary__") and f.endswith(".json"))
    if len(files) != cfg["expected_runs"]:
        raise SystemExit("expected %d condition files, found %d" % (cfg["expected_runs"], len(files)))
    rows = []
    for f in files:
        rid, fam, arm, seed = f[len("condition_secondary__"):-5].split("__")
        d = json.load(open(os.path.join(sd, f), encoding="utf-8"))
        levels = [("overall", "all", sum(v["n"] for v in d["conditions"]["size"].values()), d["overall"], d["overall_bootstrap"])]
        for ax in ("size", "chr", "edge", "frequency"):
            for lv in sorted(d["conditions"][ax]):
                c = d["conditions"][ax][lv]
                levels.append((ax, lv, c["n"], c["point"], c["bootstrap"]))
        for ax, lv, n, pt, bs in levels:
            ci = bs["scalar_ci95"]
            rows.append([rid, fam, arm, seed.replace("seed", ""), ax, lv, n,
                         fmt(pt["mIoU_disease"]), fmt(ci["mIoU_disease"][0]), fmt(ci["mIoU_disease"][1]),
                         fmt(pt["mIoU_116"]), fmt(ci["mIoU_116"][0]), fmt(ci["mIoU_116"][1]),
                         fmt(pt.get("boundary_f1"))])
    out = os.path.join(root, cfg["out_dir"]); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "T14_condition_stratified_secondary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["run_id", "family", "arm", "seed", "axis", "level", "n_images",
                    "mIoU_disease_pts", "mIoU_disease_ci95_lo", "mIoU_disease_ci95_hi",
                    "mIoU_116_pts", "mIoU_116_ci95_lo", "mIoU_116_ci95_hi", "boundary_f1_pts"])
        w.writerows(rows)
    print("T14 rows:", len(rows))


if __name__ == "__main__":
    main()

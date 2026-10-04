#!/usr/bin/env python3
"""PHYLLO-001 — result tables (versioned, deterministic; no hand-typed values).

Re-run from the package root:  python scripts/make_tables.py --config configs/tables.json
Reads only statistics/*.json and raw/ files. mIoU-type quantities are reported in points (x100).
Class names come from the P1-verified Category_Map (mask value = Index + 1; 0 = background).
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pep_common as pc  # noqa: E402

P = 100.0


def w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(header)
        for r in rows:
            wr.writerow([("%.6f" % v) if isinstance(v, float) else v for v in r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pep-root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.pep_root)
    cfg = json.load(open(os.path.join(root, a.config), encoding="utf-8"))
    S = lambda n: json.load(open(os.path.join(root, cfg["stats_dir"], n), encoding="utf-8"))
    T = os.path.join(root, cfg["out_dir"]); os.makedirs(T, exist_ok=True)
    names = {0: "background"}
    with open(os.path.join(root, cfg["raw_dir"], cfg["category_map"]), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            names[int(r["Index"]) + 1] = r["Disease"]

    h1, audit, grid = S("H1_leakage_census.json"), S("H1_manual_precision_audit.json"), S("H1_threshold_grid_ablation.json")
    w(os.path.join(T, "T01_H1_leakage_census.csv"),
      ["quantity", "value_percent", "wilson95_lo_percent", "wilson95_hi_percent", "count", "n", "source_run_ids"],
      [["hard cross-split leakage rate (primary)", h1["rate"] * P, h1["wilson95"]["lo"] * P, h1["wilson95"]["hi"] * P,
        h1["n_leaked_test"], h1["n_test"], "RUN-001"],
       ["soft-threshold leakage rate (secondary)", h1["secondary"]["soft_threshold_rate"]["rate"] * P, "", "", "", h1["n_test"], "RUN-001"],
       ["intra-train duplicate-affected rate (secondary)", h1["secondary"]["intra_train"]["affected_rate"] * P, "", "",
        h1["secondary"]["intra_train"]["n_affected_images"], "", "RUN-001"]])
    w(os.path.join(T, "T02_H1_detector_ablation.csv"), ["detector", "n_leaked_test", "n_test", "rate_percent"],
      [[k, v["n_leaked_test"], v["n_test"], v["leakage_rate"] * P] for k, v in h1["secondary"]["detector_ablation"].items()])
    rows = []
    for k, v in sorted(grid["grid"].items()):
        ph, cl = k.split("|")
        rows.append([ph.replace("pHash<=", ""), cl.replace("CLIP>=", ""), v * P, "yes" if k == grid["frozen_point"] else ""])
    w(os.path.join(T, "T03_H1_threshold_grid.csv"), ["phash_hamming_max", "clip_cosine_min", "leakage_rate_percent", "frozen_operating_point"], rows)
    w(os.path.join(T, "T04_H1_per_class_test_leakage.csv"), ["mask_value", "class_name", "n_test", "n_leaked", "rate_percent"],
      [[int(c), names.get(int(c), ""), v["n_test"], v["n_leaked"], v["rate"] * P]
       for c, v in sorted(h1["secondary"]["per_class_test_leakage"].items(), key=lambda kv: int(kv[0]))])
    w(os.path.join(T, "T05_H1_manual_precision_audit.csv"),
      ["band", "duplicates", "n_pairs", "precision", "wilson95_lo", "wilson95_hi", "raw_agreement", "cohen_kappa", "n_adjudicated"],
      [[b, audit[b]["duplicates"], audit[b]["n"], audit[b]["precision"], audit[b]["wilson95"]["lo"], audit[b]["wilson95"]["hi"],
        audit["raw_agreement"], audit["cohen_kappa"], audit["n_disagreements"]] for b in ("hard", "soft")])

    h2 = S("H2_causal_paired_bootstrap.json")
    holm = h2.get("holm_frozen", h2["holm_independent"])
    rows, rows_seed, rows_pc = [], [], []
    for bl, r in h2["per_baseline"].items():
        auth = r.get("frozen", r["independent"])
        for q in ("delta_leak", "delta_size", "delta_attr"):
            v = auth.get(q, r["independent"][q])
            p = v.get("p") if v.get("p") is not None else r["independent"][q]["p"]
            rows.append([bl, q, v["point"] * P, v["ci_lo"] * P, v["ci_hi"] * P, p,
                         r["p_conjunctive"] if q == "delta_attr" else "", h2["holm_independent"][bl]["p_holm"] if q == "delta_attr" else "",
                         {"delta_leak": r["gate_delta_leak"], "delta_attr": r["gate_delta_attr"]}.get(q, ""),
                         r["authoritative_source"], ";".join(r["source_run_ids"])])
        for s, v in r["per_seed"].items():
            rows_seed.append([bl, s, v["delta_leak"] * P, v["delta_size"] * P, v["delta_attr"] * P])
        for c, v in sorted(r["secondary"]["per_class_delta_iou_leaked_classes"].items(), key=lambda kv: int(kv[0])):
            rows_pc.append([bl, int(c), names.get(int(c), ""), "" if v is None else v * P])
        m116 = r["secondary"]["delta_leak_mIoU_116"]
        rows.append([bl, "delta_leak_mIoU_116 (secondary)", m116["point"] * P, m116["ci_lo"] * P, m116["ci_hi"] * P, m116["p"],
                     "", "", "", "independent", ";".join(r["source_run_ids"])])
    w(os.path.join(T, "T06_H2_causal_deltas.csv"),
      ["baseline", "quantity", "point_mIoU_pts", "ci95_lo_pts", "ci95_hi_pts", "bootstrap_p", "p_conjunctive", "p_holm", "gate_met",
       "authoritative_source", "source_run_ids"], rows)
    w(os.path.join(T, "T07_H2_per_seed_deltas.csv"), ["baseline", "seed", "delta_leak_pts", "delta_size_pts", "delta_attr_pts"], rows_seed)
    w(os.path.join(T, "T08_H2_per_class_delta_iou_leaked_classes.csv"), ["baseline", "mask_value", "class_name", "delta_iou_default_minus_sanitized_pts"], rows_pc)

    h3 = S("H3_background_paired_bootstrap.json")
    h3h = h3.get("holm_frozen", h3["holm_independent"])
    w(os.path.join(T, "T09_H3_background_inflation.csv"),
      ["baseline", "delta_bg_pts", "ci95_lo_pts", "ci95_hi_pts", "bootstrap_p", "p_holm", "holm_reject", "background_iou_pts", "source_run_ids"],
      [[bl, r["authoritative"]["point"] * P, r["authoritative"]["ci_lo"] * P, r["authoritative"]["ci_hi"] * P, r["authoritative"]["p"],
        h3["holm_independent"][bl]["p_holm"], h3h[bl]["reject"], r["secondary"]["background_class_iou_seed_mean"] * P,
        ";".join(r["source_run_ids"])] for bl, r in h3["per_baseline"].items()])

    sec = S("secondary_metrics_by_run.json")
    METRICS = ("mIoU_disease", "mIoU_116", "mAcc_disease", "mAcc_116", "boundary_iou", "boundary_f1")
    hdr = ["run_id", "family", "arm", "seed", "run_class", "compute_profile_id", "n_images"]
    for m in METRICS:
        hdr += [m + "_pts", m + "_ci95_lo", m + "_ci95_hi"]
    rows = []
    for rid in sorted(sec):          # evidence runs only (sensitivity runs reported in T13)
        v = sec[rid]; row = [rid, v["family"], v["arm"], v["seed"], v["run_class"], v["compute_profile_id"], v["n_images"]]
        for m in METRICS:
            row += [v["metrics"][m]["point"] * P, v["metrics"][m]["ci95"][0] * P, v["metrics"][m]["ci95"][1] * P]
        rows.append(row)
    w(os.path.join(T, "T10_run_metrics.csv"), hdr, rows)

    rows = []
    for rid in sorted(r for r, v in pc.RUNS.items() if v[3] == "evidence"):
        iou = pc.per_class_iou(pc.load_run(os.path.join(root, cfg["raw_dir"]), rid))
        for c in range(pc.NUM_CLASSES):
            rows.append([rid, c, names.get(c, ""), "" if iou[c] != iou[c] else float(iou[c]) * P])
    w(os.path.join(T, "T11_per_class_iou_by_run.csv"), ["run_id", "mask_value", "class_name", "iou_pts"], rows)

    corr = S("corrected_benchmark_descriptive.json")
    rows = []
    for bl, v in corr.items():
        for split_name, key, rid in (("corrected 70/10/20 (n=1552)", "corrected", v["corrected_run"]),
                                     ("released fixed cohort (n=1557), seed 0", "default_fixed_cohort_seed0", v["default_run_seed0"])):
            m = v[key]
            rows.append([bl, split_name, rid, m["mIoU_disease"]["point"] * P, m["mIoU_disease"]["ci95"][0] * P, m["mIoU_disease"]["ci95"][1] * P,
                         m["mIoU_116"]["point"] * P, "descriptive; non-paired (different test sets)"])
    w(os.path.join(T, "T12_corrected_benchmark_descriptive.csv"),
      ["baseline", "evaluation_split", "run_id", "mIoU_disease_pts", "ci95_lo", "ci95_hi", "mIoU_116_pts", "note"], rows)

    sens = S("sensitivity_ASR-006.json")
    rows = []
    for bl, v in sens["per_baseline"].items():
        for lab, k in (("A retain released masks (1561)", "A"), ("B metadata-consistency relabel (1561)", "B"), ("primary exclude-12 (1557)", "primary")):
            rows.append([bl, lab, v["runs"][{"A": "A_retain_raw_1561", "B": "B_relabel_1561", "primary": "primary_exclude_1557"}[k]],
                         v["mIoU_disease"][k]["point"] * P, v["mIoU_disease"][k]["ci95"][0] * P, v["mIoU_disease"][k]["ci95"][1] * P,
                         v["mIoU_116"][k]["point"] * P, "NONE"])
    w(os.path.join(T, "T13_sensitivity_ASR-006_defective_masks.csv"),
      ["baseline", "policy", "run_id", "mIoU_disease_pts", "ci95_lo", "ci95_hi", "mIoU_116_pts", "claim_support"], rows)
    print("tables written:", len(os.listdir(T)))


if __name__ == "__main__":
    main()

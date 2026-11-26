#!/usr/bin/env python3
"""PHYLLO-001 — data-dependent figures (versioned, deterministic).

Re-run from the package root:  python scripts/make_figures.py --config configs/figures.json
Stage 1 writes figures/source/<stem>.json (editable data + plot spec, derived only from
statistics/*.json). Stage 2 renders figures/{svg,pdf,png}/<stem>.* from that source alone.
P3 may restyle these figures but must never recalculate them (P2_SPEC §10A).
Conceptual figures (architecture/pipeline) are P3's and are not produced here.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "PHYLLO-001", "font.size": 9, "figure.dpi": 100})
P = 100.0


def S(root, cfg, n):
    return json.load(open(os.path.join(root, cfg["stats_dir"], n), encoding="utf-8"))


def build_sources(root, cfg):
    h1 = S(root, cfg, "H1_leakage_census.json"); grid = S(root, cfg, "H1_threshold_grid_ablation.json")
    h2 = S(root, cfg, "H2_causal_paired_bootstrap.json"); h3 = S(root, cfg, "H3_background_paired_bootstrap.json")
    sec = S(root, cfg, "secondary_metrics_by_run.json"); corr = S(root, cfg, "corrected_benchmark_descriptive.json")
    sens = S(root, cfg, "sensitivity_ASR-006.json")
    src = {}
    ab = h1["secondary"]["detector_ablation"]
    src["F01_H1_leakage_detector_ablation"] = {
        "kind": "bar", "title": "Cross-split near-duplicate leakage of the released test split (n=%d)" % h1["n_test"],
        "ylabel": "Test images with a hard cross-split near-duplicate (%)",
        "labels": list(ab.keys()), "values": [ab[k]["leakage_rate"] * P for k in ab],
        "err_for": "union", "err": [h1["wilson95"]["lo"] * P, h1["wilson95"]["hi"] * P],
        "hline": {"y": h1["threshold"] * P, "label": "pre-registered 1% threshold"},
        "source_run_ids": ["RUN-001"], "compute_profile_id": "CP-01", "run_class": "evidence"}
    ph = sorted({k.split("|")[0].replace("pHash<=", "") for k in grid["grid"]}, key=int)
    cl = sorted({k.split("|")[1].replace("CLIP>=", "") for k in grid["grid"]}, key=float)
    src["F02_H1_threshold_grid"] = {
        "kind": "heatmap", "title": "Leakage rate across the frozen detector threshold grid",
        "xlabel": "CLIP cosine threshold (>=)", "ylabel": "pHash Hamming threshold (<=)", "x": cl, "y": ph,
        "z": [[grid["grid"]["pHash<=%s|CLIP>=%s" % (p, c)] * P for c in cl] for p in ph],
        "mark": grid["frozen_point"], "source_run_ids": ["RUN-001"], "compute_profile_id": "CP-01", "run_class": "evidence"}
    rows = []
    for bl, r in h2["per_baseline"].items():
        auth = r.get("frozen", r["independent"])
        for q in ("delta_leak", "delta_size", "delta_attr"):
            v = auth.get(q, r["independent"][q])
            rows.append({"label": "%s: %s" % (bl, q), "point": v["point"] * P, "lo": v["ci_lo"] * P, "hi": v["ci_hi"] * P})
    src["F03_H2_causal_paired_deltas"] = {
        "kind": "forest", "title": "Paired effect of leakage sanitization on disease-only mIoU (fixed n=1557)",
        "xlabel": "Delta disease-only mIoU (points), 95% paired-bootstrap CI", "rows": rows,
        "vlines": [{"x": 0, "label": "no effect"}, {"x": h2["margin"] * P, "label": "pre-registered 0.5-pt margin"}],
        "source_run_ids": sorted({x for r in h2["per_baseline"].values() for x in r["source_run_ids"]}),
        "compute_profile_id": "CP-02", "run_class": "evidence"}
    src["F04_H3_background_inflation"] = {
        "kind": "forest", "title": "Background inclusion in the released 116-label metric (DEFAULT arm)",
        "xlabel": "mIoU_116 - mIoU_disease (points), 95% paired-bootstrap CI",
        "rows": [{"label": bl, "point": r["authoritative"]["point"] * P, "lo": r["authoritative"]["ci_lo"] * P,
                  "hi": r["authoritative"]["ci_hi"] * P} for bl, r in h3["per_baseline"].items()],
        "vlines": [{"x": 0, "label": "no inflation"}],
        "source_run_ids": sorted({x for r in h3["per_baseline"].values() for x in r["source_run_ids"]}),
        "compute_profile_id": "CP-02", "run_class": "evidence"}
    pts = []
    for rid, v in sorted(sec.items()):
        if v["arm"] in ("default", "sanitized", "sizectrl"):
            m = v["metrics"]["mIoU_disease"]
            pts.append({"run_id": rid, "family": v["family"], "arm": v["arm"], "seed": v["seed"],
                        "point": m["point"] * P, "lo": m["ci95"][0] * P, "hi": m["ci95"][1] * P})
    src["F05_mIoU_disease_by_arm_and_seed"] = {
        "kind": "dots", "title": "Disease-only mIoU by training arm and seed (fixed n=1557)",
        "ylabel": "Disease-only mIoU (points), 95% bootstrap CI", "points": pts,
        "source_run_ids": [p["run_id"] for p in pts], "compute_profile_id": "CP-02", "run_class": "evidence"}
    pcl = sorted(((int(c), v["rate"] * P) for c, v in h1["secondary"]["per_class_test_leakage"].items()), key=lambda t: -t[1])
    src["F06_H1_per_class_leakage"] = {
        "kind": "sortedbar", "title": "Per-class share of test images with hard cross-split near-duplicates",
        "xlabel": "Disease class (sorted by leakage rate)", "ylabel": "Leaked test images (%)",
        "labels": [str(c) for c, _ in pcl], "values": [r for _, r in pcl],
        "source_run_ids": ["RUN-001"], "compute_profile_id": "CP-01", "run_class": "evidence"}
    bars = []
    for bl, v in corr.items():
        for lab, key in (("corrected split (n=1552)", "corrected"), ("released fixed cohort (n=1557)", "default_fixed_cohort_seed0")):
            m = v[key]["mIoU_disease"]
            bars.append({"label": "%s\n%s" % (bl, lab), "point": m["point"] * P, "lo": m["ci95"][0] * P, "hi": m["ci95"][1] * P})
    src["F07_corrected_benchmark_descriptive"] = {
        "kind": "cibar", "title": "Corrected benchmark vs released split (descriptive, non-paired)",
        "ylabel": "Disease-only mIoU (points), 95% bootstrap CI", "bars": bars,
        "source_run_ids": sorted({x for v in corr.values() for x in (v["corrected_run"], v["default_run_seed0"])}),
        "compute_profile_id": "CP-02", "run_class": "evidence"}
    bars = []
    for bl, v in sens["per_baseline"].items():
        for lab, k in (("primary: exclude 12", "primary"), ("A: retain released", "A"), ("B: relabel (metadata)", "B")):
            m = v["mIoU_disease"][k]
            bars.append({"label": "%s\n%s" % (bl, lab), "point": m["point"] * P, "lo": m["ci95"][0] * P, "hi": m["ci95"][1] * P})
    src["F08_sensitivity_ASR-006_defective_masks"] = {
        "kind": "cibar", "title": "Sensitivity to the defective-mask policy (ASR-006; not claim-supporting)",
        "ylabel": "Disease-only mIoU (points), 95% bootstrap CI", "bars": bars,
        "source_run_ids": sorted({x for v in sens["per_baseline"].values() for x in v["runs"].values()}),
        "compute_profile_id": "CP-02", "run_class": "sensitivity"}
    d = os.path.join(root, cfg["out_dir"], "source"); os.makedirs(d, exist_ok=True)
    for stem, spec in src.items():
        spec["stem"] = stem
        with open(os.path.join(d, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2, sort_keys=True); fh.write("\n")
    return src


def render(spec):
    k = spec["kind"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2 if k != "forest" else 0.45 * len(spec["rows"]) + 1.6))
    if k == "bar":
        ax.bar(spec["labels"], spec["values"], color="#4C72B0")
        i = spec["labels"].index(spec["err_for"]); v = spec["values"][i]
        ax.errorbar([i], [v], yerr=[[v - spec["err"][0]], [spec["err"][1] - v]], fmt="none", ecolor="black", capsize=4)
        ax.axhline(spec["hline"]["y"], ls="--", c="grey", label=spec["hline"]["label"]); ax.legend(frameon=False)
        ax.set_ylabel(spec["ylabel"])
    elif k == "heatmap":
        im = ax.imshow(spec["z"], cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(spec["x"]))); ax.set_xticklabels(spec["x"]); ax.set_yticks(range(len(spec["y"]))); ax.set_yticklabels(spec["y"])
        for yi, row in enumerate(spec["z"]):
            for xi, val in enumerate(row):
                ax.text(xi, yi, "%.1f" % val, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Leakage rate (%)"); ax.set_xlabel(spec["xlabel"]); ax.set_ylabel(spec["ylabel"])
    elif k == "forest":
        rows = spec["rows"]; ys = list(range(len(rows)))[::-1]
        for y, r in zip(ys, rows):
            ax.errorbar([r["point"]], [y], xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]], fmt="o", c="#4C72B0", capsize=3)
        ax.set_yticks(ys); ax.set_yticklabels([r["label"] for r in rows])
        for vl, ls in zip(spec["vlines"], ("-", "--")):
            ax.axvline(vl["x"], ls=ls, c="grey", lw=1, label=vl["label"])
        ax.legend(frameon=False, fontsize=8); ax.set_xlabel(spec["xlabel"])
    elif k == "dots":
        groups = sorted({(p["family"], p["arm"]) for p in spec["points"]})
        for gi, g in enumerate(groups):
            ps = [p for p in spec["points"] if (p["family"], p["arm"]) == g]
            for j, p in enumerate(ps):
                x = gi + (j - (len(ps) - 1) / 2) * 0.12
                ax.errorbar([x], [p["point"]], yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]], fmt="o", c="#4C72B0", capsize=2)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(["%s\n%s" % g for g in groups], fontsize=7); ax.set_ylabel(spec["ylabel"])
    elif k == "sortedbar":
        ax.bar(range(len(spec["values"])), spec["values"], color="#4C72B0", width=1.0)
        ax.set_xlabel(spec["xlabel"]); ax.set_ylabel(spec["ylabel"]); ax.set_xticks([])
    elif k == "cibar":
        b = spec["bars"]
        ax.bar(range(len(b)), [x["point"] for x in b], color="#4C72B0",
               yerr=[[x["point"] - x["lo"] for x in b], [x["hi"] - x["point"] for x in b]], capsize=3)
        ax.set_xticks(range(len(b))); ax.set_xticklabels([x["label"] for x in b], fontsize=7); ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"], fontsize=9)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pep-root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.pep_root)
    cfg = json.load(open(os.path.join(root, a.config), encoding="utf-8"))
    build_sources(root, cfg)
    base = os.path.join(root, cfg["out_dir"])
    for fmt in ("svg", "pdf", "png"):
        os.makedirs(os.path.join(base, fmt), exist_ok=True)
    for fn in sorted(os.listdir(os.path.join(base, "source"))):
        spec = json.load(open(os.path.join(base, "source", fn), encoding="utf-8"))   # render FROM source only
        fig = render(spec); stem = fn[:-5]
        fig.savefig(os.path.join(base, "svg", stem + ".svg"), metadata={"Date": None})
        fig.savefig(os.path.join(base, "pdf", stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(os.path.join(base, "png", stem + ".png"), dpi=300, metadata={"Software": None})
        plt.close(fig)
    print("figures rendered:", len(os.listdir(os.path.join(base, "source"))))


if __name__ == "__main__":
    main()

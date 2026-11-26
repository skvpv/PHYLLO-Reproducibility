#!/usr/bin/env python3
"""make_tables.py / make_figures.py combined — regenerate every table and figure from the
raw statistics.json + leak summary ONLY. No manually typed numbers (P2_SPEC §11).

Tables (CSV, machine-checkable):
  T1 leakage census + Wilson CI + sensitivity surface (H1)
  T2 H2 per-baseline delta_leak / delta_size / delta_attr with CIs + verdict + Holm
  T3 H3 per-baseline delta_bg with CI + verdict + Holm
  T4 claim -> result mapping (C-01..C-05)
Figures (PNG+SVG+PDF):
  F1 sensitivity heatmap of leakage rate over the pHash x CLIP grid
  F2 H2 forest plot (delta_leak, delta_attr per baseline with 95% CI)
  F3 H3 bar (delta_bg per baseline with 95% CI)
Every figure has an editable source (the CSV it reads + this script).
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np


def _w(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", path)


def tables(stats, leak, outdir, leak_meta=None):
    os.makedirs(outdir, exist_ok=True)
    # T1 leakage
    if stats.get("H1"):
        h1 = stats["H1"]
        _w(os.path.join(outdir, "T1_leakage.csv"),
           ["metric", "value"],
           [["leakage_rate", h1["leakage_rate"]],
            ["wilson95_lo", h1["wilson95"]["lo"]], ["wilson95_hi", h1["wilson95"]["hi"]],
            ["verdict", stats["verdicts"].get("H1", "")]])
    if leak:
        grid = leak
        _w(os.path.join(outdir, "T1b_sensitivity.csv"),
           ["setting", "leakage_rate"], sorted(grid.items()))
    if leak_meta:
        _w(os.path.join(outdir, "T1c_detector_ablation.csv"),
           ["method", "n_leaked_test", "n_test", "leakage_rate"],
           [[method, row["n_leaked_test"], row["n_test"], row["leakage_rate"]]
            for method, row in sorted(leak_meta.get("detector_ablation", {}).items())])
        _w(os.path.join(outdir, "T1d_per_class_leakage.csv"),
           ["class_id", "n_leaked", "n_test", "leakage_rate"],
           [[class_id, row["n_leaked"], row["n_test"], row["rate"]]
            for class_id, row in sorted(leak_meta.get("per_class_test_leakage", {}).items(),
                                        key=lambda item: int(item[0]))])
    # T2 H2
    rows = []
    for b, d in stats.get("H2", {}).get("per_baseline", {}).items():
        dl, ds, da = d["delta_leak"], d["delta_size"], d["delta_attr"]
        rows.append([b, dl["point"], dl["ci_lo"], dl["ci_hi"],
                     ds["point"], ds["ci_lo"], ds["ci_hi"],
                     da["point"], da["ci_lo"], da["ci_hi"],
                     dl.get("p_value_two_sided"), da.get("p_value_two_sided"),
                     d.get("holm_reject"), d["verdict"]])
    _w(os.path.join(outdir, "T2_H2.csv"),
       ["baseline", "delta_leak", "dl_lo", "dl_hi", "delta_size", "dsz_lo", "dsz_hi",
        "delta_attr", "da_lo", "da_hi", "p_delta_leak", "p_delta_attr",
        "holm_reject", "verdict"], rows)
    # T3 H3
    rows = []
    for b, d in stats.get("H3", {}).get("per_baseline", {}).items():
        bg = d["delta_bg"]
        rows.append([b, bg["point"], bg["ci_lo"], bg["ci_hi"],
                     bg.get("p_value_two_sided"), d.get("holm_reject"),
                     d["positive_excl0"]])
    _w(os.path.join(outdir, "T3_H3.csv"),
       ["baseline", "delta_bg", "lo", "hi", "p", "holm_reject", "positive_excl0"], rows)
    # T4 claim map
    _w(os.path.join(outdir, "T4_claim_map.csv"),
       ["claim", "hypothesis", "run", "verdict"],
       [["C-01", "H1", "R-LEAK-DETECT", stats["verdicts"].get("H1", "")],
        ["C-02", "H2", "R-BENCH-{DEFAULT,SANITIZED,SIZECTRL}-FIXED", stats["verdicts"].get("H2", "")],
        ["C-03", "H3", "R-BENCH-DEFAULT-FIXED", stats["verdicts"].get("H3", "")],
        ["C-04", "-", "R-NONE-CODEPATH (config audit)", "verified-from-configs"],
        ["C-05", "-", "R-CORRECTED-BENCHMARK (non-paired)", "descriptive"]])


def figures(stats, leak, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    def save(fig, stem):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), bbox_inches="tight", dpi=150)
        plt.close(fig); print("wrote", stem, "(png/svg/pdf)")

    # F1 sensitivity heatmap
    if leak:
        phs = sorted({int(k.split("pHash<=")[1].split("|")[0]) for k in leak})
        cls = sorted({float(k.split("CLIP>=")[1]) for k in leak})
        M = np.full((len(phs), len(cls)), np.nan)
        for i, p in enumerate(phs):
            for j, c in enumerate(cls):
                M[i, j] = leak.get(f"pHash<={p}|CLIP>={c:.2f}", np.nan)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(cls))); ax.set_xticklabels([f"{c:.2f}" for c in cls])
        ax.set_yticks(range(len(phs))); ax.set_yticklabels(phs)
        ax.set_xlabel("CLIP cosine hard threshold"); ax.set_ylabel("pHash Hamming <=")
        ax.set_title("Leakage-rate sensitivity surface (H1)")
        fig.colorbar(im, ax=ax, label="leakage rate")
        save(fig, "F1_sensitivity_heatmap")

    # F2 H2 forest
    h2 = stats.get("H2", {}).get("per_baseline", {})
    if h2:
        fig, ax = plt.subplots(figsize=(6, 0.8 + 0.6 * len(h2)))
        y = 0; yticks, ylabels = [], []
        for b, d in h2.items():
            for name, off, col in (("delta_leak", 0.15, "C0"), ("delta_attr", -0.15, "C1")):
                v = d[name]
                ax.errorbar(v["point"], y + off,
                            xerr=[[v["point"] - v["ci_lo"]], [v["ci_hi"] - v["point"]]],
                            fmt="o", color=col, capsize=3,
                            label=name if y == 0 else None)
            yticks.append(y); ylabels.append(b); y += 1
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvline(0.005, color="grey", lw=0.8, ls=":")  # 0.5 mIoU points on 0..1 scale
        ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
        ax.set_xlabel("mIoU (0–1 scale; 0.005 = 0.5 points)")
        ax.set_title("H2 effects (95% bootstrap CI)")
        ax.legend()
        save(fig, "F2_H2_forest")

    # F3 H3 bars
    h3 = stats.get("H3", {}).get("per_baseline", {})
    if h3:
        fig, ax = plt.subplots(figsize=(6, 4))
        bs = list(h3); pts = [h3[b]["delta_bg"]["point"] for b in bs]
        los = [h3[b]["delta_bg"]["point"] - h3[b]["delta_bg"]["ci_lo"] for b in bs]
        his = [h3[b]["delta_bg"]["ci_hi"] - h3[b]["delta_bg"]["point"] for b in bs]
        ax.bar(range(len(bs)), pts, yerr=[los, his], capsize=4, color="C2")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(bs))); ax.set_xticklabels(bs, rotation=20, ha="right")
        ax.set_ylabel("delta_bg (mIoU_116 - mIoU_disease)")
        ax.set_title("H3 background inflation (95% bootstrap CI)")
        save(fig, "F3_H3_bars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--sensitivity", default="", help="sensitivity_surface.json (H1 grid)")
    ap.add_argument("--leak-summary", default="", help="full leak_detect_summary.json")
    ap.add_argument("--tables-dir", default="results/tables")
    ap.add_argument("--figures-dir", default="results/figures")
    a = ap.parse_args()
    stats = json.load(open(a.stats))
    leak = json.load(open(a.sensitivity)) if a.sensitivity and os.path.exists(a.sensitivity) else {}
    leak_meta = json.load(open(a.leak_summary)) if a.leak_summary and os.path.exists(a.leak_summary) else None
    tables(stats, leak, a.tables_dir, leak_meta)
    try:
        figures(stats, leak, a.figures_dir)
    except Exception as e:
        print("figure generation skipped (matplotlib unavailable?):", e)


if __name__ == "__main__":
    main()

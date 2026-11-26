#!/usr/bin/env python3
"""PHYLLO-001 — statistics for the Publication_Evidence_Package (versioned, deterministic).

Re-run from the package root:  python scripts/make_statistics.py --config configs/statistics.json

Authority: Hypothesis_Register.md locked block (sha256:594dd656…c82a201f) and the P2.G1
Implementation_Contract. PRIMARY inference (H1 Wilson, H2/H3 paired bootstrap, Holm) is
computed by the FROZEN P2 package modules (snapshot under scripts/frozen_phyllo_src/, SHA-256
recorded), and cross-checked by an independent implementation. Any disagreement in a point
estimate, or in any verdict-gate decision, aborts with exit code 3 (fail closed).

Verdict vocabulary (P2_SPEC §6 C7A): CONFIRMED | PARTIALLY_SUPPORTED | REJECTED. The locked
register defines no partial-support rule; a mixed per-baseline outcome is therefore NOT
auto-resolved: the script aborts (exit 4) for governed review instead of choosing a verdict.
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pep_common as pc  # noqa: E402


def fail(code, msg):
    """Explicit exit codes: 3 = frozen/independent disagreement, 4 = mixed verdict (no locked
    partial rule), 5 = input/integrity failure. (SystemExit(<str>) would always exit 1.)"""
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


# --------------------------------------------------------------------------------------
# Exact counts-weighted bootstrap engine (integer areas -> float64 matmul is exact < 2**53)
# --------------------------------------------------------------------------------------
class PairedStream:
    """One shared stream of B resampling index vectors over a fixed cohort of size N.
    The same stream is reused for every arm, seed and metric on that cohort (paired)."""

    def __init__(self, n, B, seed):
        rng = np.random.default_rng(seed)
        self.counts = np.empty((B, n), dtype=np.float64)
        for b in range(B):                      # sequential draws: identical to per-replicate
            idx = rng.integers(0, n, n)         # rng.integers(0, N, N) used by the SAP check
            self.counts[b] = np.bincount(idx, minlength=n)
        self.n, self.B, self.seed = n, B, seed

    def sums(self, a):
        """a: (N, K) per-image contributions -> (B, K) resampled dataset-level sums."""
        return self.counts @ a.astype(np.float64)


def _nanmean_rows(num, den, sl):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return np.nanmean(r[:, sl], axis=1)


def boot_metric(stream, r, metric):
    I = stream.sums(r["area_intersect"]); U = stream.sums(r["area_union"])
    if metric == "mIoU_disease":
        return _nanmean_rows(I, U, pc.DISEASE)
    if metric == "mIoU_116":
        return _nanmean_rows(I, U, slice(0, 116))
    L = stream.sums(r["area_label"])
    if metric == "mAcc_disease":
        return _nanmean_rows(I, L, pc.DISEASE)
    if metric == "mAcc_116":
        return _nanmean_rows(I, L, slice(0, 116))
    Bd = stream.sums(r["boundary"])
    tp, fp, fn = Bd[:, 0], Bd[:, 1], Bd[:, 2]
    if metric == "boundary_iou":
        return tp / (tp + fp + fn)
    if metric == "boundary_f1":
        return 2 * tp / (2 * tp + fp + fn)
    raise KeyError(metric)


# --------------------------------------------------------------------------------------
# Frozen-API adapter (introspective; never guesses semantics, fails closed)
# --------------------------------------------------------------------------------------
SEED_NAMES = ("seed", "rng_seed", "random_state", "random_seed")
P_NAMES = ("p_value_two_sided", "p", "p_value", "pvalue", "p_two_sided")
SAMPLE_NAMES = ("samples", "dist", "distribution", "draws", "boot_samples")   # never "replicates" (frozen: an int count)


def _kw(fn, B, seed):
    params = inspect.signature(fn).parameters
    kw = {}
    if "n_replicates" in params:
        kw["n_replicates"] = B
    for s in SEED_NAMES:
        if s in params:
            kw[s] = seed
            break
    return kw


def _get(o, names):
    for n in names:
        if isinstance(o, dict) and n in o:
            return o[n]
        if hasattr(o, n):
            return getattr(o, n)
    return None


def _norm(o):
    """Normalise a frozen result object to {point, ci_lo, ci_hi, excludes_zero, p, samples}."""
    out = {k: _get(o, (k,)) for k in ("point", "ci_lo", "ci_hi", "excludes_zero")}
    if out["point"] is None or out["ci_lo"] is None or out["ci_hi"] is None:
        fail(5, "FATAL(frozen API): result lacks point/ci_lo/ci_hi: %r" % (o,))
    out = {k: (float(v) if k != "excludes_zero" and v is not None else v) for k, v in out.items()}
    if out["excludes_zero"] is None:
        out["excludes_zero"] = bool(out["ci_lo"] > 0 or out["ci_hi"] < 0)
    p = _get(o, P_NAMES)
    smp = _get(o, SAMPLE_NAMES)
    out["p"] = None if p is None else float(p)
    smp = None if smp is None else np.asarray(smp, dtype=np.float64)
    out["samples"] = smp if (smp is not None and smp.ndim == 1 and smp.size > 1) else None
    return out


class Frozen:
    def __init__(self, src):
        self.src = src
        sys.path.insert(0, src)
        from phyllo.stats.wilson import wilson_interval
        from phyllo.stats.paired_bootstrap import ConfusionSet, h2_paired_bootstrap, h3_paired_bootstrap
        from phyllo.stats.holm import holm_reject, bootstrap_two_sided_p
        self.wilson_interval, self.ConfusionSet = wilson_interval, ConfusionSet
        self.h2, self.h3 = h2_paired_bootstrap, h3_paired_bootstrap
        self.holm_reject, self.boot_p = holm_reject, bootstrap_two_sided_p
        self.kwargs_used = {}

    def p_from(self, norm):
        if norm["p"] is not None:
            return norm["p"], "frozen:result.p"
        if norm["samples"] is not None:
            return float(self.boot_p(norm["samples"])), "frozen:bootstrap_two_sided_p(samples)"
        return None, "frozen API exposes neither p nor samples"


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pep-root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.pep_root)
    cfg = json.load(open(os.path.join(root, a.config), encoding="utf-8"))
    raw = os.path.join(root, cfg["raw_dir"]); out = os.path.join(root, cfg["out_dir"])
    os.makedirs(out, exist_ok=True)
    B, seed, alpha, lvl = cfg["B"], cfg["seed"], cfg["alpha"], cfg["ci_level"]

    runs = {rid: pc.load_run(raw, rid) for rid in pc.RUNS}
    cohort = [l.strip() for l in open(os.path.join(raw, cfg["cohort_file"]), encoding="utf-8") if l.strip()]

    # ---- C2 pairing integrity: every fixed-cohort run has identical key order ----------
    fixed = [r for r, v in pc.RUNS.items() if v[1] in ("default", "sanitized", "sizectrl")]
    ref_keys = runs[fixed[0]]["keys"]
    pairing = {rid: bool(np.array_equal(runs[rid]["keys"], ref_keys)) for rid in fixed}
    cohort_match = [str(k) for k in ref_keys] == cohort or \
        sorted(str(k).rsplit(".", 1)[0] for k in ref_keys) == sorted(c.rsplit(".", 1)[0] for c in cohort)
    if not all(pairing.values()):
        fail(5, "FATAL(C2): fixed-cohort key order differs across runs: %s" % pairing)
    N = len(ref_keys)
    if N != cfg["cohort_n"]:
        fail(5, f"FATAL(C2): cohort size {N} != {cfg['cohort_n']}")

    # ---- metric cross-validation against frozen per-run summaries ---------------------
    xval = {}
    for rid, r in runs.items():
        m = pc.run_metrics(r)
        xval[rid] = max(abs(m[k] - r["summary"][k]) for k in m)
    if max(xval.values()) > 1e-12:
        fail(5, "FATAL: metric recomputation deviates from frozen summaries: %s" % xval)

    fz = None
    if cfg.get("require_frozen", True):
        fz = Frozen(os.path.join(root, cfg["frozen_src"]))
    streams = {}

    def stream(n, offset=0):
        key = (n, offset)
        if key not in streams:
            streams[key] = PairedStream(n, B, seed + offset)
        return streams[key]

    S_fixed = stream(N)
    crosscheck = {"tolerances": cfg["crosscheck_tol"], "items": []}

    def check(name, frozen_v, indep_v, p_f=None, p_i=None):
        tol = cfg["crosscheck_tol"]
        item = {"name": name, "frozen": {k: frozen_v[k] for k in ("point", "ci_lo", "ci_hi", "excludes_zero")},
                "independent": indep_v, "p_frozen": p_f, "p_independent": p_i}
        item["point_ok"] = abs(frozen_v["point"] - indep_v["point"]) <= tol["point"]
        item["ci_ok"] = (abs(frozen_v["ci_lo"] - indep_v["ci_lo"]) <= tol["ci"]
                         and abs(frozen_v["ci_hi"] - indep_v["ci_hi"]) <= tol["ci"])
        item["gate_ok"] = bool(frozen_v["excludes_zero"]) == bool(indep_v["excludes_zero"])
        crosscheck["items"].append(item)
        # identical PCG64(seed) streams => the frozen and independent bootstrap p must agree exactly
        item["p_ok"] = (p_f is None) or (p_i is None) or abs(p_f - p_i) <= 1e-12
        if not (item["point_ok"] and item["gate_ok"] and item["p_ok"]):
            pc.dump(crosscheck, os.path.join(out, "frozen_vs_independent_crosscheck.json"))
            fail(3, f"FATAL(exit 3): frozen vs independent disagreement on {name}")

    def indep_summary(x, point):
        lo, hi = pc.percentile_ci(x, lvl)
        return {"point": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
                "p": pc.boot_p_two_sided(x)}

    # ================================ H1 ==============================================
    leak = json.load(open(os.path.join(raw, "RUN-001__leak_detect_summary.json"), encoding="utf-8"))
    surface = json.load(open(os.path.join(raw, "RUN-001__sensitivity_surface.json"), encoding="utf-8"))
    k, n = leak["n_leaked_test"], leak["n_test"]
    w_i = pc.wilson(k, n)
    if fz:
        wf = fz.wilson_interval(k, n)
        w_f = {"point": float(wf.point), "lo": float(wf.lo), "hi": float(wf.hi)}
        if max(abs(w_f[x] - w_i[x]) for x in ("point", "lo", "hi")) > 1e-12:
            fail(3, "FATAL(exit 3): Wilson frozen vs independent disagree")
    else:
        w_f = None
    hard_key = cfg["h1_frozen_grid_point"]
    if abs(surface[hard_key] - leak["leakage_rate"]) > 1e-15:
        fail(5, "FATAL: sensitivity-surface frozen point != primary leakage rate")
    H1 = {"hypothesis": "H1", "primary_outcome": "hard cross-split near-duplicate leakage rate (1561 test images)",
          "n_leaked_test": k, "n_test": n, "rate": w_i["point"],
          "wilson95": {"lo": w_i["lo"], "hi": w_i["hi"]}, "wilson_source": "frozen phyllo.stats.wilson" if w_f else "independent",
          "wilson_crosscheck": {"frozen": w_f, "independent": w_i, "max_abs_dev": (max(abs(w_f[x] - w_i[x]) for x in ("point", "lo", "hi")) if w_f else None)},
          "threshold": cfg["h1_threshold"], "threshold_rule": "point rate >= 1% (Wilson descriptive; contract adds no CI threshold)",
          "secondary": {
              "soft_threshold_rate": {"grid_point": cfg["h1_soft_grid_point"], "rate": surface[cfg["h1_soft_grid_point"]],
                                      "source": "frozen detector sensitivity surface (hard-union-soft thresholds)"},
              "per_class_test_leakage": leak["per_class_test_leakage"],
              "intra_train": leak["intra_train"],
              "detector_ablation": leak["detector_ablation"],
              "n_hard_edges": leak["n_hard_edges"], "n_soft_edges": leak["n_soft_edges"],
              "n_hard_clusters": leak["n_hard_clusters"]},
          "source_run_ids": ["RUN-001"], "compute_profile_id": leak.get("compute_profile_id")}
    H1["verdict"] = "CONFIRMED" if H1["rate"] >= cfg["h1_threshold"] else "REJECTED"
    pc.dump(H1, os.path.join(out, "H1_leakage_census.json"))
    pc.dump({"grid": surface, "frozen_point": hard_key, "source_run_ids": ["RUN-001"],
             "claim_support": "NONE (detector-threshold ablation; descriptive)"},
            os.path.join(out, "H1_threshold_grid_ablation.json"))

    # manual precision audit (recomputed from raw reviewer files + sealed key)
    import csv
    key = {r["pair_id"]: r["candidate_kind"].strip().lower()
           for r in json.load(open(os.path.join(raw, "RUN-001__manual_audit_KEY.json"), encoding="utf-8"))}
    r1 = {r["pair_id"]: r["verdict"].strip().lower() for r in csv.DictReader(open(os.path.join(raw, "RUN-001__manual_audit_reviewer_1.csv"), encoding="utf-8"))}
    r2 = {r["pair_id"]: r["verdict"].strip().lower() for r in csv.DictReader(open(os.path.join(raw, "RUN-001__manual_audit_reviewer_2.csv"), encoding="utf-8"))}
    adj = {r["pair_id"]: r["consensus_verdict"].strip().lower() for r in csv.DictReader(open(os.path.join(raw, "RUN-001__manual_audit_adjudication.csv"), encoding="utf-8"))}
    if set(r1) != set(r2) or set(r1) != set(key):
        fail(5, "FATAL: manual-audit pair sets differ")
    disagree = sorted(p for p in r1 if r1[p] != r2[p])
    if set(disagree) != set(adj):
        fail(5, "FATAL: adjudication set != reviewer disagreement set")
    cons = {p: (r1[p] if r1[p] == r2[p] else adj[p]) for p in r1}
    po = 1 - len(disagree) / len(r1)
    labs = sorted(set(r1.values()) | set(r2.values()))
    pe = sum((sum(v == l for v in r1.values()) / len(r1)) * (sum(v == l for v in r2.values()) / len(r2)) for l in labs)
    audit = {"n_pairs": len(r1), "raw_agreement": po, "n_disagreements": len(disagree),
             "cohen_kappa": (po - pe) / (1 - pe), "adjudicated_before_unseal": True}
    final = json.load(open(os.path.join(raw, "RUN-001__manual_audit_results_FINAL.json"), encoding="utf-8"))
    for band in ("hard", "soft"):
        ids = [p for p in cons if key[p] == band]
        dup = sum(cons[p] == "duplicate" for p in ids)
        w = pc.wilson(dup, len(ids))
        audit[band] = {"duplicates": dup, "n": len(ids), "precision": w["point"], "wilson95": {"lo": w["lo"], "hi": w["hi"]}}
        f = final[band]
        if dup != f["duplicates"] or len(ids) != f["n"] or abs(w["lo"] - f["wilson95"]["lo"]) > 1e-6:
            fail(5, f"FATAL: recomputed {band} audit != official FINAL")
    audit["matches_official_final"] = True
    pc.dump(audit, os.path.join(out, "H1_manual_precision_audit.json"))

    # ================================ H2 ==============================================
    leaked_classes = sorted(int(c) for c, v in leak["per_class_test_leakage"].items() if v["n_leaked"] > 0)
    H2 = {"hypothesis": "H2", "units": "mIoU as fraction (x100 = mIoU points)", "B": B, "seed": seed,
          "ci_level": lvl, "margin": cfg["h2_margin"], "per_baseline": {}}
    p_conj = {}
    for bl, (fam, seeds) in pc.BASELINES.items():
        arm = {a: {s: runs[pc.run_for(fam, a, s)] for s in seeds} for a in ("default", "sanitized", "sizectrl")}
        # independent paired bootstrap: per replicate, per-seed deltas then seed mean
        bd = {a: {s: boot_metric(S_fixed, arm[a][s], "mIoU_disease") for s in seeds} for a in arm}
        bl_leak = np.mean([bd["default"][s] - bd["sanitized"][s] for s in seeds], axis=0)
        bl_size = np.mean([bd["default"][s] - bd["sizectrl"][s] for s in seeds], axis=0)
        bl_attr = bl_leak - bl_size
        full = np.arange(N)
        pt = {a: {s: pc.miou_disease_idx(arm[a][s], full) for s in seeds} for a in arm}
        per_seed = {s: {"delta_leak": pt["default"][s] - pt["sanitized"][s],
                        "delta_size": pt["default"][s] - pt["sizectrl"][s]} for s in seeds}
        for s in seeds:
            per_seed[s]["delta_attr"] = per_seed[s]["delta_leak"] - per_seed[s]["delta_size"]
        ind = {"delta_leak": indep_summary(bl_leak, float(np.mean([per_seed[s]["delta_leak"] for s in seeds]))),
               "delta_size": indep_summary(bl_size, float(np.mean([per_seed[s]["delta_size"] for s in seeds]))),
               "delta_attr": indep_summary(bl_attr, float(np.mean([per_seed[s]["delta_attr"] for s in seeds])))}
        res = {"independent": ind}
        if fz:
            cs = {a: {s: fz.ConfusionSet(arm[a][s]["area_intersect"], arm[a][s]["area_union"]) for s in seeds} for a in arm}
            kw = _kw(fz.h2, B, seed); fz.kwargs_used["h2"] = kw
            fr = fz.h2(cs["default"], cs["sanitized"], cs["sizectrl"], **kw)
            fres = {}
            for d in ("delta_leak", "delta_attr", "delta_size"):
                o = fr.get(d) if isinstance(fr, dict) else getattr(fr, d, None)
                if o is None:
                    continue
                nv = _norm(o); p, psrc = fz.p_from(nv)
                fres[d] = {k2: nv[k2] for k2 in ("point", "ci_lo", "ci_hi", "excludes_zero")}
                fres[d]["p"], fres[d]["p_source"] = p, psrc
                check(f"H2/{bl}/{d}", nv, ind[d], p, ind[d]["p"])
            extra = {k2: v for k2, v in (fr.items() if isinstance(fr, dict) else vars(fr).items())
                     if k2 not in ("delta_leak", "delta_attr", "delta_size") and np.ndim(v) == 0}
            res["frozen"] = fres; res["frozen_extra_scalars"] = {k2: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k2, v in extra.items()}
            auth = fres
        else:
            auth = ind
        res["authoritative_source"] = "frozen" if fz else "independent"
        # pre-registered gate (locked threshold, transcribed verbatim in the docstring)
        dl, da = auth["delta_leak"], auth["delta_attr"]
        gate_leak = dl["point"] >= cfg["h2_margin"] and dl["ci_lo"] > 0
        gate_attr = da["point"] > 0 and da["ci_lo"] > 0
        pl = dl.get("p") if dl.get("p") is not None else ind["delta_leak"]["p"]
        pa = da.get("p") if da.get("p") is not None else ind["delta_attr"]["p"]
        p_conj[bl] = max(pl, pa)            # conservative intersection-union p (contract)
        res.update({"per_seed": {str(s): v for s, v in per_seed.items()},
                    "between_seed_sd_delta_leak": float(np.std([per_seed[s]["delta_leak"] for s in seeds], ddof=1)) if len(seeds) > 1 else None,
                    "relative_delta_leak": dl["point"] / float(np.mean([pt["default"][s] for s in seeds])),
                    "gate_delta_leak": gate_leak, "gate_delta_attr": gate_attr, "p_conjunctive": p_conj[bl]})
        # secondary outcomes (locked): per-class delta IoU on leaked classes; delta on mIoU_116
        pcl = {}
        for c in leaked_classes:
            vals = [float(pc.per_class_iou(arm["default"][s])[c] - pc.per_class_iou(arm["sanitized"][s])[c]) for s in seeds]
            pcl[str(c)] = float(np.nanmean(vals)) if not all(np.isnan(vals)) else None
        b116 = np.mean([boot_metric(S_fixed, arm["default"][s], "mIoU_116") - boot_metric(S_fixed, arm["sanitized"][s], "mIoU_116") for s in seeds], axis=0)
        p116 = float(np.mean([pc.miou_116_idx(arm["default"][s], full) - pc.miou_116_idx(arm["sanitized"][s], full) for s in seeds]))
        res["secondary"] = {"per_class_delta_iou_leaked_classes": pcl, "n_leaked_classes": len(leaked_classes),
                            "delta_leak_mIoU_116": indep_summary(b116, p116)}
        res["source_run_ids"] = sorted(pc.run_for(fam, a, s) for a in arm for s in seeds)
        H2["per_baseline"][bl] = res
    H2["holm_independent"] = pc.holm(p_conj, alpha)
    if fz:
        hf = fz.holm_reject(dict(p_conj), alpha)
        H2["holm_frozen"] = {k2: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else (float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv))) for kk, vv in (v.items() if isinstance(v, dict) else {"reject": v}.items())} for k2, v in hf.items()}
        for k2 in p_conj:
            if bool(H2["holm_frozen"][k2]["reject"]) != H2["holm_independent"][k2]["reject"]:
                fail(3, "FATAL(exit 3): Holm frozen vs independent disagree")
    holm_used = H2.get("holm_frozen", H2["holm_independent"])
    flags = {bl: bool(v["gate_delta_leak"] and v["gate_delta_attr"] and holm_used[bl]["reject"]) for bl, v in H2["per_baseline"].items()}
    H2["baseline_supported"] = flags
    if all(flags.values()):
        H2["verdict"] = "CONFIRMED"
    elif not any(flags.values()):
        H2["verdict"] = "REJECTED"
    else:
        pc.dump(H2, os.path.join(out, "H2_causal_paired_bootstrap.json"))
        fail(4, "HALT(exit 4): mixed H2 outcome across baselines; locked register has no partial rule")
    pc.dump(H2, os.path.join(out, "H2_causal_paired_bootstrap.json"))

    # ================================ H3 ==============================================
    H3 = {"hypothesis": "H3", "units": "mIoU as fraction", "B": B, "seed": seed, "per_baseline": {}}
    p_bg = {}
    for bl, (fam, seeds) in pc.BASELINES.items():
        dflt = {s: runs[pc.run_for(fam, "default", s)] for s in seeds}
        full = np.arange(N)
        bb = np.mean([boot_metric(S_fixed, dflt[s], "mIoU_116") - boot_metric(S_fixed, dflt[s], "mIoU_disease") for s in seeds], axis=0)
        pt_bg = float(np.mean([pc.miou_116_idx(dflt[s], full) - pc.miou_disease_idx(dflt[s], full) for s in seeds]))
        ind = indep_summary(bb, pt_bg)
        res = {"independent": ind}
        if fz:
            cs = {s: fz.ConfusionSet(dflt[s]["area_intersect"], dflt[s]["area_union"]) for s in seeds}
            kw = _kw(fz.h3, B, seed); fz.kwargs_used["h3"] = kw
            nv = _norm(fz.h3(cs, **kw)); p, psrc = fz.p_from(nv)
            check(f"H3/{bl}/delta_bg", nv, ind, p, ind["p"])
            auth = {k2: nv[k2] for k2 in ("point", "ci_lo", "ci_hi", "excludes_zero")}
            auth["p"], auth["p_source"] = (p if p is not None else ind["p"]), (psrc if p is not None else "independent (frozen exposes no p)")
            res["frozen"] = auth
        else:
            auth = ind
        p_bg[bl] = auth["p"]
        bg_iou = [float(pc.per_class_iou(dflt[s])[0]) for s in seeds]
        res.update({"authoritative": auth, "gate": bool(auth["point"] > 0 and auth["ci_lo"] > 0),
                    "secondary": {"background_class_iou_seed_mean": float(np.mean(bg_iou)), "per_seed": bg_iou},
                    "source_run_ids": sorted(pc.run_for(fam, "default", s) for s in seeds)})
        H3["per_baseline"][bl] = res
    H3["holm_independent"] = pc.holm(p_bg, alpha)
    if fz:
        hf = fz.holm_reject(dict(p_bg), alpha)
        H3["holm_frozen"] = {k2: {"reject": bool(v["reject"] if isinstance(v, dict) else v)} for k2, v in hf.items()}
    holm3 = H3.get("holm_frozen", H3["holm_independent"])
    flags3 = {bl: bool(v["gate"] and holm3[bl]["reject"]) for bl, v in H3["per_baseline"].items()}
    H3["baseline_supported"] = flags3
    if all(flags3.values()):
        H3["verdict"] = "CONFIRMED"
    elif not any(flags3.values()):
        H3["verdict"] = "REJECTED"
    else:
        pc.dump(H3, os.path.join(out, "H3_background_paired_bootstrap.json"))
        fail(4, "HALT(exit 4): mixed H3 outcome; locked threshold requires all baselines")
    pc.dump(H3, os.path.join(out, "H3_background_paired_bootstrap.json"))

    # ===================== secondary metrics for every run (bootstrap CIs) ============
    METRICS = ("mIoU_116", "mIoU_disease", "mAcc_116", "mAcc_disease", "boundary_iou", "boundary_f1")
    sec = {}
    for rid, r in runs.items():
        n_r = len(r["keys"])
        st = S_fixed if n_r == N and pairing.get(rid) else stream(n_r, offset=n_r)   # separate stream per cohort
        m = pc.run_metrics(r)
        sec[rid] = {"family": pc.RUNS[rid][0], "arm": pc.RUNS[rid][1], "seed": pc.RUNS[rid][2],
                    "run_class": pc.RUNS[rid][3], "compute_profile_id": pc.RUNS[rid][4], "n_images": n_r,
                    "bootstrap_stream_seed": st.seed,
                    "metrics": {k2: {"point": m[k2], "ci95": pc.percentile_ci(boot_metric(st, r, k2), lvl)} for k2 in METRICS}}
    pc.dump({k2: v for k2, v in sec.items() if v["run_class"] == "evidence"},
            os.path.join(out, "secondary_metrics_by_run.json"))

    # ===================== corrected benchmark (C-05; descriptive, non-paired) ========
    corr = {}
    for bl, (fam, _) in pc.BASELINES.items():
        rc = pc.run_for(fam, "corrected", 0); rd = pc.run_for(fam, "default", 0)
        corr[bl] = {"corrected_run": rc, "default_run_seed0": rd,
                    "corrected": sec[rc]["metrics"], "default_fixed_cohort_seed0": sec[rd]["metrics"],
                    "note": "different test sets (1552 corrected vs 1557 fixed); descriptive only, no paired test (C-05, locked)"}
    pc.dump(corr, os.path.join(out, "corrected_benchmark_descriptive.json"))

    # ===================== sensitivity A/B (ASR-006; claim_support NONE) ==============
    sens = {"asr_id": "ASR-006", "claim_support": "NONE", "per_baseline": {}}
    for bl, (fam, _) in pc.BASELINES.items():
        ra, rb, rp = pc.run_for(fam, "sens-raw", 0), pc.run_for(fam, "sens-relabel", 0), pc.run_for(fam, "default", 0)
        A, Bm, P = runs[ra], runs[rb], runs[rp]
        ka = [str(x) for x in A["keys"]]; kp = {str(x): i for i, x in enumerate(P["keys"])}
        sub = np.array([i for i, x in enumerate(ka) if x in kp])
        subP = np.array([kp[ka[i]] for i in sub])
        def _same(x, y):
            return bool(np.array_equal(x["area_intersect"][sub], y["area_intersect"][subP]) and
                        np.array_equal(x["area_union"][sub], y["area_union"][subP]))
        sens["per_baseline"][bl] = {
            "runs": {"A_retain_raw_1561": ra, "B_relabel_1561": rb, "primary_exclude_1557": rp},
            "mIoU_disease": {"A": sec[ra]["metrics"]["mIoU_disease"], "B": sec[rb]["metrics"]["mIoU_disease"], "primary": sec[rp]["metrics"]["mIoU_disease"]},
            "mIoU_116": {"A": sec[ra]["metrics"]["mIoU_116"], "B": sec[rb]["metrics"]["mIoU_116"], "primary": sec[rp]["metrics"]["mIoU_116"]},
            "delta_A_minus_primary_disease": sec[ra]["metrics"]["mIoU_disease"]["point"] - sec[rp]["metrics"]["mIoU_disease"]["point"],
            "delta_B_minus_primary_disease": sec[rb]["metrics"]["mIoU_disease"]["point"] - sec[rp]["metrics"]["mIoU_disease"]["point"],
            "metrics_full": {ra: sec[ra]["metrics"], rb: sec[rb]["metrics"]},
            "n_shared_images": int(len(sub)),
            "shared_1557_identical_to_primary": {"A": _same(A, P), "B": _same(Bm, P)},
            # Same checkpoint re-evaluated: quantifies the evaluation (inference) reproducibility floor
            "inference_reproducibility_same_checkpoint": {
                "checkpoints_identical": A["summary"]["checkpoint"] == P["summary"]["checkpoint"],
                "n_shared_images": int(len(sub)),
                "n_images_with_any_confusion_difference": int(((A["area_intersect"][sub] != P["area_intersect"][subP]).any(1)
                                                               | (A["area_union"][sub] != P["area_union"][subP]).any(1)).sum()),
                "mIoU_disease_shared_primary": pc.miou_disease_idx(P, subP),
                "mIoU_disease_shared_sensA": pc.miou_disease_idx(A, sub),
                "delta_sensA_minus_primary_on_identical_images": pc.miou_disease_idx(A, sub) - pc.miou_disease_idx(P, subP)}}
    pc.dump(sens, os.path.join(out, "sensitivity_ASR-006.json"))

    # ===================== verdicts + provenance ======================================
    verdicts = {"locked_content_hash": pc.LOCKED_HASH, "vocabulary": "CONFIRMED | PARTIALLY_SUPPORTED | REJECTED",
                "H1": H1["verdict"], "H2": H2["verdict"], "H3": H3["verdict"]}
    crosscheck["frozen_kwargs_used"] = fz.kwargs_used if fz else None
    crosscheck["metric_xval_max_abs_dev"] = max(xval.values())
    crosscheck["pairing_integrity"] = pairing
    crosscheck["cohort_file_matches_npz_keys"] = bool(cohort_match)
    pc.dump(crosscheck, os.path.join(out, "frozen_vs_independent_crosscheck.json"))
    print(json.dumps(verdicts))


if __name__ == "__main__":
    main()

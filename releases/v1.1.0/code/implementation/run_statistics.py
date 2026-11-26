#!/usr/bin/env python3
"""run_statistics.py — frozen statistical analysis for H1/H2/H3 (EBSAP).

MANDATORY FIRST STEP: re-verify the Hypothesis_Register locked_content_hash against the
manifest value. On mismatch the script HALTS — a verdict against a mutated hypothesis is
invalid (OPS_P2 I-12).

Consumes per-arm/seed confusion npz (from phyllo.eval.evaluate_arm) and the leak-detect
summary (from run_leak_detect). Produces, from raw results only (no hand-typed numbers):
  * H1: leakage rate + Wilson CI -> verdict.
  * H2: paired image-level bootstrap (>=10,000, shared indices, seed-averaged deltas) of
    delta_leak / delta_size / delta_attr per baseline; Holm-Bonferroni across baselines
    -> verdict.
  * H3: paired bootstrap of delta_bg (mIoU_116 - mIoU_disease) on the DEFAULT arm per
    baseline; Holm across baselines -> verdict.
Verdicts are CONFIRMED / PARTIALLY_SUPPORTED / REJECTED — recorded honestly, never tuned.

npz layout expected under --results-dir:
  <baseline>/<arm>/seed<k>.npz   with arrays area_intersect,area_union,keys  (arm in
  {default,sanitized,sizectrl}; H3 uses default only). Baselines e.g. segnext-l, deeplabv3plus-r101.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phyllo.stats.paired_bootstrap import (ConfusionSet, h2_paired_bootstrap,
                                           h3_paired_bootstrap, N_REPLICATES)
from phyllo.stats.holm import holm_reject
from phyllo.stats.wilson import wilson_interval
from phyllo.stats.hypothesis_verdict import (verify_locked_hash, verdict_h1,
                                             verdict_h2, verdict_h3)


def _load_arm(base_dir):
    """Return ({seed: ConfusionSet}, {seed: exact ordered cohort keys})."""
    out = {}
    keys = {}
    for p in sorted(glob.glob(os.path.join(base_dir, "seed*.npz"))):
        m = re.search(r"seed(\d+)\.npz$", p)
        s = int(m.group(1))
        z = np.load(p, allow_pickle=True)
        required = {"area_intersect", "area_union", "keys"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{p}: missing arrays {sorted(missing)}")
        out[s] = ConfusionSet(z["area_intersect"], z["area_union"])
        keys[s] = tuple(str(x) for x in z["keys"].tolist())
        if len(keys[s]) != out[s].n or len(set(keys[s])) != len(keys[s]):
            raise ValueError(f"{p}: invalid/duplicate cohort keys")
    return out, keys


def _assert_same_keys(named):
    """Enforce byte-for-byte image ordering across every arm and seed."""
    reference_name, reference = next(iter(named.items()))
    for name, keys in named.items():
        if keys != reference:
            raise ValueError(f"cohort key/order mismatch: {name} != {reference_name}")


def analyse(results_dir, register, expected_hash, leak_summary, out, replicates):
    # 0. HARD GATE — locked hash
    verify_locked_hash(register, expected_hash)

    report = {"locked_hash_verified": True, "replicates": replicates,
              "H1": None, "H2": {}, "H3": {}, "verdicts": {}}

    # H1
    if leak_summary and os.path.exists(leak_summary):
        ls = json.load(open(leak_summary))
        if ls["n_test"] != 1561 or ls["n_images"] != 7774:
            raise ValueError("H1 census must cover exactly 7774 images and 1561 test images")
        ci = wilson_interval(ls["n_leaked_test"], ls["n_test"])
        v1 = verdict_h1(ls["leakage_rate"], ci.lo, ci.hi)
        report["H1"] = {"leakage_rate": ls["leakage_rate"],
                        "wilson95": {"lo": ci.lo, "hi": ci.hi},
                        "verdict": v1.verdict, "evidence": v1.verdict_evidence}
        report["verdicts"]["H1"] = v1.verdict
    else:
        raise FileNotFoundError("H1 leak summary is mandatory")

    baselines = sorted(d for d in os.listdir(results_dir)
                       if os.path.isdir(os.path.join(results_dir, d)))

    # H2 per baseline
    h2_pvals, h2_detail = {}, {}
    for b in baselines:
        base = os.path.join(results_dir, b)
        loaded = {a: _load_arm(os.path.join(base, a))
                  for a in ("default", "sanitized", "sizectrl")}
        arms = {a: loaded[a][0] for a in loaded}
        if not all(arms.values()):
            continue
        expected_seeds = {"segnext-l": [0, 1, 2], "deeplabv3plus-r101": [0]}.get(b)
        if expected_seeds is None:
            continue
        for arm, values in arms.items():
            if sorted(values) != expected_seeds:
                raise ValueError(f"{b}/{arm}: seeds {sorted(values)} != {expected_seeds}")
            if any(cs.n != 1557 for cs in values.values()):
                raise ValueError(f"{b}/{arm}: primary cohort N must equal 1557")
        key_maps = {a: loaded[a][1] for a in loaded}
        named_keys = {f"{a}/seed{s}": keys for a, km in key_maps.items()
                      for s, keys in km.items()}
        _assert_same_keys(named_keys)
        res = h2_paired_bootstrap(arms["default"], arms["sanitized"], arms["sizectrl"],
                                  n_replicates=replicates)
        dl, ds, da = res["delta_leak"], res["delta_size"], res["delta_attr"]
        v2 = verdict_h2(dl.point, dl.ci_lo, dl.ci_hi, da.point, da.ci_lo, da.ci_hi)
        # H2 is conjunctive; max(p_leak,p_attr) is the conservative intersection-union
        # p-value. Holm then controls the family across evaluated baselines.
        composite_p = max(dl.p_value_two_sided, da.p_value_two_sided)
        h2_pvals[f"H2@{b}"] = composite_p
        h2_detail[b] = {"delta_leak": vars(dl), "delta_size": vars(ds),
                        "delta_attr": vars(da), "verdict": v2.verdict,
                        "evidence": v2.verdict_evidence}
    holm2 = holm_reject(h2_pvals) if h2_pvals else {}
    required_baselines = {"segnext-l", "deeplabv3plus-r101"}
    if set(h2_detail) != required_baselines:
        raise ValueError(f"H2 requires both frozen baselines; found {sorted(h2_detail)}")
    for b, detail in h2_detail.items():
        decision = holm2[f"H2@{b}"]
        detail["holm_reject"] = decision["reject"]
        if not decision["reject"] and detail["verdict"] == "CONFIRMED":
            detail["verdict"] = "PARTIALLY_SUPPORTED"
            detail["evidence"] += "; downgraded because Holm-adjusted test did not reject"
    report["H2"] = {"per_baseline": h2_detail, "holm": holm2}

    # H3 per baseline (DEFAULT arm)
    h3_pvals, h3_detail = {}, {}
    for b in baselines:
        default, default_keys = _load_arm(os.path.join(results_dir, b, "default"))
        if not default:
            continue
        _assert_same_keys({f"{b}/default/seed{s}": k for s, k in default_keys.items()})
        r3 = h3_paired_bootstrap(default, n_replicates=replicates)
        h3_pvals[f"H3@{b}"] = r3.p_value_two_sided
        h3_detail[b] = {"delta_bg": vars(r3),
                        "positive_excl0": bool(r3.point > 0 and r3.excludes_zero)}
    holm3 = holm_reject(h3_pvals) if h3_pvals else {}
    if set(h3_detail) != required_baselines:
        raise ValueError(f"H3 requires both frozen baselines; found {sorted(h3_detail)}")
    for b, detail in h3_detail.items():
        detail["holm_reject"] = holm3[f"H3@{b}"]["reject"]
    all_true = bool(h3_detail) and all(v["positive_excl0"] and v["holm_reject"]
                                        for v in h3_detail.values())
    any_true = any(v["positive_excl0"] and v["holm_reject"]
                   for v in h3_detail.values())
    v3 = verdict_h3(all_true, f"per_baseline={ {b: v['positive_excl0'] for b,v in h3_detail.items()} } any_true={any_true}")
    report["H3"] = {"per_baseline": h3_detail, "holm": holm3, "verdict": v3.verdict}

    # aggregate H2 verdict across baselines: CONFIRMED iff every evaluated baseline CONFIRMED
    if h2_detail:
        verds = {b: d["verdict"] for b, d in h2_detail.items()}
        if all(x == "CONFIRMED" for x in verds.values()):
            report["verdicts"]["H2"] = "CONFIRMED"
        elif any(x in ("CONFIRMED", "PARTIALLY_SUPPORTED") for x in verds.values()):
            report["verdicts"]["H2"] = "PARTIALLY_SUPPORTED"
        else:
            report["verdicts"]["H2"] = "REJECTED"
    report["verdicts"]["H3"] = v3.verdict

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=str, allow_nan=False)
    print(json.dumps(report["verdicts"], indent=2))
    print("full statistics ->", out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="dir of <baseline>/<arm>/seed*.npz")
    ap.add_argument("--register", required=True, help="Hypothesis_Register.md")
    ap.add_argument("--expected-hash", required=True,
                    help="manifest locked_content_hash (sha256:...)")
    ap.add_argument("--leak-summary", required=True, help="R-LEAK-DETECT summary json")
    ap.add_argument("--out", default="results/statistics.json")
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    a = ap.parse_args()
    analyse(a.results_dir, a.register, a.expected_hash, a.leak_summary, a.out, a.replicates)


if __name__ == "__main__":
    main()

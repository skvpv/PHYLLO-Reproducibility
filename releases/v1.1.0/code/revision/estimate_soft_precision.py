#!/usr/bin/env python3
"""PHYLLO CR-PHYLLO-R3-01 — prespecified soft-candidate precision and agreement estimator (v1).
Resolves decision D-VAL-1. Specification: review_protocol/PRESPECIFIED_ESTIMATOR_D-VAL-1.md.

It ESTIMATES; it never decides on a rater's behalf and never relabels U. Per-pair outcomes are
derived with the unchanged scorer v3 decision function (imported, SHA-256 asserted):
  D  final grouping decision "grouped"     (final category C1-C3)
  N  final grouping decision "not_grouped" (final category C4/C5)
  U  residual unresolved: a rater gave U (or the raters disagreed on grouping) and adjudication
     returned U. U is reported separately and is NEVER counted as a verified positive or negative.

Preconditions (otherwise REFUSED, exit 2):
  * packet: PACKET_MANIFEST status COMPLETE; the review folder's sealed key equals it;
  * independence: INDEPENDENCE_STATUS PASS or PASS_WITH_WARNINGS for this packet;
  * scorer v3 output: STATUS COMPLETE, or ADJUDICATION_REQUIRED where EVERY outstanding pair is
    residual U after adjudication (reason "...; adjudication returned U"); the scorer's recorded
    input SHA-256s equal the current review-folder files (no edits after scoring);
  * re-derived decisions agree with scorer v3 (review_decisions.csv when COMPLETE; the
    adjudication_required.csv pair set otherwise).

Estimates (stratum h = 1..4, N_h population, n_h sample, W_h = N_h / N, d_h, u_h, m_h = n_h - u_h):
  lower bound   p_L = sum_h W_h d_h / n_h            (U counted as not duplicate: bound only)
  upper bound   p_U = sum_h W_h (d_h + u_h) / n_h    (U counted as duplicate:     bound only)
  resolved-only p_R = sum_h W_h d_h / m_h            (CONDITIONAL on U missing at random within
                                                      stratum; secondary, labelled as such)
  95% CI: stratified normal with finite-population correction,
          Var = sum_h W_h^2 (1 - n_h/N_h) p_h (1 - p_h) / (n_h - 1), truncated to [0, 1];
          per stratum: Wilson 95%. Sensitivity: stratified bootstrap percentile CI
          (B = 10,000, numpy default_rng(20261112), resampling pairs within strata).
  uncertainty envelope = [lower CI limit of p_L, upper CI limit of p_U].
  estimated genuine duplicates among the N soft candidates = N x p (same three versions).
Agreement (initial ratings, before adjudication), overall and per stratum:
  grouping decision on pairs where neither rater gave U: percent agreement, Cohen's kappa, PABAK;
  category (C1-C5, U): percent agreement, Cohen's kappa; U rate per rater.
  Bootstrap 95% CIs for overall kappas (same B and seed). Agreement is UNWEIGHTED (sample-level).
Sensitivity subsets (descriptive): excluding pairs in the Round-3 default target frame; excluding
pairs with a round-1 binary label.
NOT produced: detector recall; a list of confirmed duplicates; pooled estimates with round-1 or
AI-assisted records.
Outputs (new --out): PRECISION_ESTIMATE.json, per_stratum_estimates.csv,
per_pair_outcomes_INTERNAL.csv (pair_id, stratum, outcome; no image keys), ESTIMATE_STATUS.json.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_soft_review as A        # approved scorer v3 — decision logic reused, not modified

SCORER_V3_SHA256 = "e1e20a48ad0494a826c1e1b56055285adc418263ad2b7db866a34c4ef0d5af31"
STRATA = ("CLIP 0.95-0.96", "CLIP 0.96-0.97", "CLIP >=0.97", "pHash band")
B_BOOT, SEED = 10_000, 20261112
Z = 1.959963984540054


class EstError(Exception):
    pass


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wilson(k, n):
    return A.wilson(k, n)


def strat_est(W, x, n, Nh):
    """Design-weighted proportion and stratified normal 95% CI with FPC. x, n, Nh per stratum."""
    p = sum(W[h] * x[h] / n[h] for h in W if n[h])
    var = sum(W[h] ** 2 * (1 - n[h] / Nh[h]) * (x[h] / n[h]) * (1 - x[h] / n[h]) / (n[h] - 1)
              for h in W if n[h] > 1)
    se = math.sqrt(var)
    return {"estimate": p, "se": se, "ci95": [max(0.0, p - Z * se), min(1.0, p + Z * se)]}


def kappa(x, y):
    r = A.cohen(list(x), list(y))
    out = {"n": r["n"], "pct_agree": r["pct_agree"], "kappa": r["kappa"]}
    if r["n"] and set(x) | set(y) <= {"grouped", "not_grouped"}:
        out["pabak"] = 2 * r["pct_agree"] - 1
    return out


def derive(review_dir):
    errors = []
    key = A.load_key(review_dir, errors)
    r1 = A.load_sheet(review_dir, A.SHEETS[0], key, errors)
    r2 = A.load_sheet(review_dir, A.SHEETS[1], key, errors)
    adj = A.load_adjudication(review_dir, key, errors)
    if errors:
        raise EstError(f"scorer v3 validation errors: {errors[:5]}")
    out, pending = {}, []
    for pid in sorted(key):
        rec, need, err = A.decide(pid, r1[pid], r2[pid], adj, False)
        if err:
            raise EstError(f"scorer v3 decision error: {err}")
        if need:
            if not need.endswith("adjudication returned U"):
                pending.append((pid, need))
            out[pid] = ("U", r1[pid], r2[pid], rec)
        else:
            dec = rec["final_group_decision"]
            out[pid] = ({"grouped": "D", "not_grouped": "N"}[dec], r1[pid], r2[pid], rec)
    if pending:
        raise EstError(f"{len(pending)} pair(s) still need adjudication (not residual U), e.g. {pending[:3]}")
    return key, out


def estimate(a):
    if sha(A.__file__) != SCORER_V3_SHA256:
        raise EstError("imported scorer is not the approved scorer v3")
    pkt, rd, sd = Path(a.packet), Path(a.review_dir), Path(a.scored_dir)
    pm = lj(pkt / "PACKET_MANIFEST.json")
    if pm.get("status") != "COMPLETE":
        raise EstError(f"packet status {pm.get('status')!r} (a released, complete packet is required)")
    key_sha = pm["sealed_key_sha256"]
    if sha(pkt / "sealed" / A.KEY_NAME) != key_sha or sha(rd / A.KEY_NAME) != key_sha:
        raise EstError("sealed key in the packet or review folder does not match PACKET_MANIFEST")
    ind = lj(Path(a.independence_dir, "INDEPENDENCE_STATUS.json"))
    if ind.get("packet_sealed_key_sha256") != key_sha or ind.get("status") not in ("PASS", "PASS_WITH_WARNINGS"):
        raise EstError(f"independence status {ind.get('status')!r} for this packet: estimation not permitted")
    for rv in (1, 2):
        if sha(rd / f"sheet_reviewer_{rv}.csv") != ind["returned_sha256"][f"sheet_reviewer_{rv}.csv"]:
            raise EstError(f"review-folder sheet {rv} differs from the sheet that passed the independence check")
    st = lj(sd / "STATUS.json")
    if st.get("script_sha256") != SCORER_V3_SHA256:
        raise EstError("scorer output was not produced by scorer v3")
    if st.get("status") not in ("COMPLETE", "ADJUDICATION_REQUIRED"):
        raise EstError(f"scorer status {st.get('status')!r}")
    rec_in = set((st.get("input_sha256") or {}).values())
    present = [rd / A.KEY_NAME] + [rd / s for s in A.SHEETS] + ([rd / A.ADJ_NAME] if (rd / A.ADJ_NAME).is_file() else [])
    if any(sha(p) not in rec_in for p in present):
        raise EstError("review-folder files differ from the scorer v3 inputs (edited or added after scoring)")
    for n_, h in (st.get("artifact_sha256") or {}).items():
        if sha(sd / n_) != h:
            raise EstError(f"scorer artifact {n_} does not match STATUS.json")
    key, out = derive(rd)
    residual = sorted(p for p, v in out.items() if v[0] == "U")
    if st["status"] == "COMPLETE":
        if residual:
            raise EstError("scorer COMPLETE but residual U re-derived: inconsistent inputs")
        with open(sd / "review_decisions.csv", newline="") as f:
            sc = {r["pair_id"]: r["final_group_decision"] for r in csv.DictReader(f)}
        if any({"D": "grouped", "N": "not_grouped"}[v[0]] != sc.get(p) for p, v in out.items()):
            raise EstError("re-derived decisions differ from scorer v3 review_decisions.csv")
    else:
        with open(sd / "adjudication_required.csv", newline="") as f:
            req = list(csv.DictReader(f))
        if sorted(r["pair_id"] for r in req) != residual or \
                not all(r["reason"].endswith("adjudication returned U") for r in req):
            raise EstError("scorer v3 outstanding items are not exactly the residual-U pairs")

    items = {it["pair_id"]: it for it in lj(rd / A.KEY_NAME)["items"]}
    if set(items) != set(out):
        raise EstError("key items and decisions differ")
    Nh = {h: None for h in STRATA}
    for it in items.values():
        if it["stratum"] not in Nh:
            raise EstError(f"unknown stratum {it['stratum']!r}")
        if Nh[it["stratum"]] not in (None, it["N_h"]):
            raise EstError("inconsistent N_h in sealed key")
        Nh[it["stratum"]] = it["N_h"]
    if None in Nh.values():
        raise EstError("a stratum has no sampled pairs")
    N = sum(Nh.values()); W = {h: Nh[h] / N for h in STRATA}
    by = defaultdict(list)
    for pid, v in out.items():
        by[items[pid]["stratum"]].append((pid, v))

    def block(sel):
        """Estimates for a subset of pairs (sel: pid -> bool)."""
        n = {h: sum(1 for p, _ in by[h] if sel(p)) for h in STRATA}
        d = {h: sum(1 for p, v in by[h] if sel(p) and v[0] == "D") for h in STRATA}
        u = {h: sum(1 for p, v in by[h] if sel(p) and v[0] == "U") for h in STRATA}
        m = {h: n[h] - u[h] for h in STRATA}
        if min(n.values()) == 0:
            return {"n": n, "note": "a stratum is empty in this subset; design-weighted estimate not computed"}
        res = {"n": n, "D": d, "U": u, "N": {h: n[h] - d[h] - u[h] for h in STRATA},
               "lower_bound_U_as_not_duplicate": strat_est(W, d, n, Nh),
               "upper_bound_U_as_duplicate": strat_est(W, {h: d[h] + u[h] for h in STRATA}, n, Nh)}
        res["resolved_only_CONDITIONAL_MAR"] = (strat_est(W, d, m, Nh) if min(m.values()) > 0
                                                else {"note": "a stratum has no resolved pairs"})
        res["uncertainty_envelope_ci95"] = [res["lower_bound_U_as_not_duplicate"]["ci95"][0],
                                            res["upper_bound_U_as_duplicate"]["ci95"][1]]
        res["estimated_genuine_duplicates_among_candidates"] = {
            "population_N": N,
            "lower_bound": [N * res["lower_bound_U_as_not_duplicate"]["estimate"]] + [N * c for c in res["lower_bound_U_as_not_duplicate"]["ci95"]],
            "upper_bound": [N * res["upper_bound_U_as_duplicate"]["estimate"]] + [N * c for c in res["upper_bound_U_as_duplicate"]["ci95"]],
            "format": "[point, ci95_low, ci95_high]"}
        return res

    main = block(lambda p: True)
    # stratified bootstrap (sensitivity)
    rng = np.random.default_rng(SEED)
    codes = {h: np.array([{"D": 0, "N": 1, "U": 2}[v[0]] for _, v in by[h]]) for h in STRATA}
    g = {h: [("grouped" if A.group_of(v[1]) else "not_grouped") if A.group_of(v[1]) is not None else None for _, v in by[h]]
         for h in STRATA}
    g2 = {h: [("grouped" if A.group_of(v[2]) else "not_grouped") if A.group_of(v[2]) is not None else None for _, v in by[h]]
          for h in STRATA}
    c1 = {h: [v[1] for _, v in by[h]] for h in STRATA}; c2 = {h: [v[2] for _, v in by[h]] for h in STRATA}
    bl, bu, kg, kc = [], [], [], []
    for _ in range(B_BOOT):
        pl = pu = 0.0; X1, X2, Y1, Y2 = [], [], [], []
        for h in STRATA:
            ix = rng.integers(0, len(codes[h]), len(codes[h])); s_ = codes[h][ix]
            pl += W[h] * np.mean(s_ == 0); pu += W[h] * np.mean(s_ != 1)
            for i in ix:
                if g[h][i] is not None and g2[h][i] is not None:
                    X1.append(g[h][i]); X2.append(g2[h][i])
                Y1.append(c1[h][i]); Y2.append(c2[h][i])
        bl.append(pl); bu.append(pu)
        kg.append(A.cohen(X1, X2)["kappa"]); kc.append(A.cohen(Y1, Y2)["kappa"])
    pct = lambda v: [float(np.percentile([x for x in v if x is not None], 2.5)),
                     float(np.percentile([x for x in v if x is not None], 97.5))] if any(x is not None for x in v) else None
    main["lower_bound_U_as_not_duplicate"]["bootstrap_ci95"] = pct(bl)
    main["upper_bound_U_as_duplicate"]["bootstrap_ci95"] = pct(bu)

    def agreement(hs):
        x1 = [a_ for h in hs for a_, b_ in zip(g[h], g2[h]) if a_ is not None and b_ is not None]
        x2 = [b_ for h in hs for a_, b_ in zip(g[h], g2[h]) if a_ is not None and b_ is not None]
        y1 = [x for h in hs for x in c1[h]]; y2 = [x for h in hs for x in c2[h]]
        return {"grouping_excluding_any_U": kappa(x1, x2), "category_including_U": kappa(y1, y2),
                "U_rate_rater_1": y1.count("U") / len(y1), "U_rate_rater_2": y2.count("U") / len(y2)}

    agr = {"overall": agreement(STRATA) | {"kappa_bootstrap_ci95": {"grouping": pct(kg), "category": pct(kc)}}}
    per_stratum = []
    for h in STRATA:
        n, d, u = main["n"][h], main["D"][h], main["U"][h]
        agr[h] = agreement([h])
        per_stratum.append({"stratum": h, "N_h": Nh[h], "n_h": n, "W_h": round(W[h], 6), "D": d, "N_not_dup": n - d - u, "U": u,
                            "p_lower": d / n, "p_lower_wilson95": wilson(d, n), "p_upper": (d + u) / n,
                            "p_upper_wilson95": wilson(d + u, n),
                            "p_resolved_only": d / (n - u) if n - u else None,
                            "kappa_grouping": agr[h]["grouping_excluding_any_U"]["kappa"],
                            "kappa_category": agr[h]["category_including_U"]["kappa"]})
    sens = {"excluding_round3_default_target_frame": block(lambda p: not items[p].get("in_round3_default_target_frame")),
            "excluding_round1_labelled": block(lambda p: items[p].get("round1_binary_consensus") is None)}
    return {"status": "ESTIMATED", "decision": "D-VAL-1", "residual_U_only_outstanding": st["status"] == "ADJUDICATION_REQUIRED",
            "n_pairs": len(out), "n_residual_U": len(residual), "strata_population": Nh, "weights": W,
            "precision": main, "agreement": agr, "sensitivity_descriptive": sens,
            "bootstrap": {"B": B_BOOT, "seed": SEED, "generator": "numpy default_rng", "resampling": "pairs within strata"},
            "packet_sealed_key_sha256": key_sha, "scorer_status": st["status"], "scorer_status_sha256": sha(sd / "STATUS.json"),
            "independence_status_sha256": sha(Path(a.independence_dir, "INDEPENDENCE_STATUS.json")),
            "estimator_sha256": sha(__file__), "scorer_v3_sha256": SCORER_V3_SHA256,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reporting_restrictions": ["U is reported separately; p_lower and p_upper are bounds, not estimates of U's status",
                                       "p_resolved_only assumes U missing at random within stratum (secondary)",
                                       "no detector recall; no list of confirmed duplicates; no pooling with round-1 or AI-assisted records"]}, \
        per_stratum, [(p, items[p]["stratum"], v[0]) for p, v in sorted(out.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--packet", "--review-dir", "--scored-dir", "--independence-dir", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out); stg = out.parent / f".{out.name}.incomplete"
    try:
        if out.exists() or stg.exists():
            raise EstError(f"{out} or its staging directory exists")
        rep, rows, pairs = estimate(a)
    except (EstError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"REFUSED: {e}", file=sys.stderr); return 2
    stg.mkdir()
    (stg / "PRECISION_ESTIMATE.json").write_text(json.dumps(rep, indent=1))
    with open(stg / "per_stratum_estimates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(stg / "per_pair_outcomes_INTERNAL.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_id", "stratum", "outcome_D_N_U"]); w.writerows(pairs)
    arts = {p.name: sha(p) for p in sorted(stg.iterdir())}
    (stg / "ESTIMATE_STATUS.json").write_text(json.dumps({"status": "ESTIMATED", "packet_sealed_key_sha256": rep["packet_sealed_key_sha256"],
        "scorer_status_sha256": rep["scorer_status_sha256"], "residual_U_only_outstanding": rep["residual_U_only_outstanding"],
        "estimator_sha256": rep["estimator_sha256"], "artifact_sha256": arts}, indent=1))
    os.rename(stg, out)
    pr = rep["precision"]
    print(json.dumps({"n_pairs": rep["n_pairs"], "n_residual_U": rep["n_residual_U"],
                      "p_lower": pr["lower_bound_U_as_not_duplicate"]["estimate"],
                      "p_upper": pr["upper_bound_U_as_duplicate"]["estimate"],
                      "envelope_ci95": pr["uncertainty_envelope_ci95"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

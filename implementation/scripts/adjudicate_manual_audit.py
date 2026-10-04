#!/usr/bin/env python3
"""Join blinded reviewer decisions to the sealed key; report hard/soft Wilson precision."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phyllo.stats.wilson import wilson_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--out", default="results/R-LEAK-DETECT/manual_audit_results.json")
    args = parser.parse_args()
    review = {x["pair_id"]: x for x in json.load(open(args.review, encoding="utf-8"))}
    key = {x["pair_id"]: x for x in json.load(open(args.key, encoding="utf-8"))}
    if set(review) != set(key):
        raise SystemExit("review/key pair-id mismatch")
    counts = {"hard": [0, 0], "soft": [0, 0]}
    allowed = {"duplicate": True, "non-duplicate": False}
    for pair_id, row in review.items():
        first = str(row.get("reviewer_1_verdict", "")).strip().lower()
        second = str(row.get("reviewer_2_verdict", "")).strip().lower()
        consensus = str(row.get("consensus_verdict", "")).strip().lower()
        if first not in allowed or second not in allowed:
            raise SystemExit(f"{pair_id}: both reviewer verdicts are required")
        if first == second:
            verdict = first
            if consensus and consensus != first:
                raise SystemExit(f"{pair_id}: consensus contradicts agreeing reviewers")
        else:
            if consensus not in allowed:
                raise SystemExit(f"{pair_id}: disagreement requires consensus_verdict")
            verdict = consensus
        kind = key[pair_id]["candidate_kind"]
        counts[kind][1] += 1
        counts[kind][0] += int(allowed[verdict])
    report = {}
    for kind, (successes, n) in counts.items():
        interval = wilson_interval(successes, n)
        report[kind] = {"duplicates": successes, "n": n, "precision": interval.point,
                        "wilson95": {"lo": interval.lo, "hi": interval.hi}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

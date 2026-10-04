#!/usr/bin/env python3
"""Machine gate used by Slurm entry points; exits nonzero on missing/false evidence.

Corrections vs prior version (engineering only; no experiment redesign):
  1. The training runtime-budget gate now reads the AUTHORITATIVE official split
     estimate (budget 1200 GPU-h, budget_pass=True), not the invalid conservative
     ceiling in results/runtime/runtime_estimate.json (budget 168, budget_pass=False),
     which the CP-02 discovery explicitly marks invalid for sizing.
  2. The training stage now verifies that CP-02 is frozen and integral before any
     model-training run launches (schema rule: an evidence run may not launch under an
     unknown/unfrozen compute profile). RUN-001 (detection) ran under CP-01 and is
     unaffected.
"""
import argparse
import hashlib
import json
import os


def load(path):
    if not os.path.isfile(path):
        raise SystemExit(f"required gate artifact missing: {path}")
    return json.load(open(path, encoding="utf-8"))


def assert_cp02_frozen(path):
    """Fail closed unless Compute_Profile.json is CP-02, FROZEN, and self-consistent."""
    cp = load(path)
    if cp.get("compute_profile_id") != "CP-02":
        raise SystemExit(f"compute profile gate: expected CP-02, got {cp.get('compute_profile_id')!r}")
    frozen = (cp.get("freeze_status") == "FROZEN") or ("P2.G0 PASS" in str(cp.get("frozen_at", "")))
    if not frozen:
        raise SystemExit("compute profile gate: CP-02 is not FROZEN")
    stored = cp.pop("checksum", None)
    if not stored:
        raise SystemExit("compute profile gate: CP-02 checksum missing")
    canonical = json.dumps(cp, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    recomputed = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    if recomputed != stored:
        raise SystemExit("compute profile gate: CP-02 checksum mismatch (profile edited without re-freeze)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["detection", "training"], required=True)
    parser.add_argument("--preflight", default="preflight_out/preflight_report.json")
    parser.add_argument("--smoke", default="smoke_out/smoke_report.json")
    # CORRECTION 1: authoritative official split estimate is the default runtime gate source.
    parser.add_argument("--runtime",
                        default="results/runtime/cp02_pilot/job_870/estimate_split_official.json")
    parser.add_argument("--compute-profile", default="Compute_Profile.json")
    args = parser.parse_args()

    preflight, smoke = load(args.preflight), load(args.smoke)
    if preflight.get("preflight_pass") is not True:
        raise SystemExit("preflight gate is not PASS")
    if not (smoke.get("mode") == "real" and smoke.get("smoke_pass") is True and
            smoke.get("authorizes_campaign") is True):
        raise SystemExit("real scientific smoke gate is not PASS")

    if args.stage == "training":
        if load(args.runtime).get("budget_pass") is not True:
            raise SystemExit("runtime budget gate is not PASS")
        # CORRECTION 2: CP-02 must be frozen and integral before model-training runs.
        assert_cp02_frozen(args.compute_profile)

    print(f"campaign gates PASS for {args.stage}")


if __name__ == "__main__":
    main()

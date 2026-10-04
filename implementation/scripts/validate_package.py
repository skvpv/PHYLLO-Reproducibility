#!/usr/bin/env python3
"""Offline authoritative package-integrity/static-contract validator."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXPECTED_HASH = "sha256:594dd656d4e0bcc999c3ecea201928b663d5426806c1ee3092b5fd18c82a201f"


def result(name, ok, detail=""):
    print(f"{name:34} {'PASS' if ok else 'FAIL'} {detail}")
    return bool(ok)


def main():
    checks = []
    manifest = json.load(open(ROOT / "manifests" / "PACKAGE_MANIFEST.json"))
    failures = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(entry["path"])
    checks.append(result("package_hashes", not failures, failures[:5]))
    checks.append(result("manifest_count", manifest["n_files"] == len(manifest["files"])))

    cp = json.load(open(ROOT / "Compute_Profile.json")); expected = cp.pop("checksum")
    actual = "sha256:" + hashlib.sha256(json.dumps(
        cp, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    checks.append(result("compute_profile_checksum", actual == expected, actual))
    from phyllo.stats.hypothesis_verdict import recompute_locked_hash
    got = recompute_locked_hash(str(ROOT / "inputs" / "Hypothesis_Register.md"))
    checks.append(result("hypothesis_locked_hash", got == EXPECTED_HASH, got))

    required_columns = ["run_id", "project_id", "run_class", "method", "config_path",
        "seed", "dataset", "split", "compute_profile_id", "pre_registered",
        "linked_claim_ids", "linked_asr_id", "sweep_value", "status",
        "quarantine_cr", "returned_at"]
    with open(ROOT / "manifests" / "Runs_Manifest.csv", newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader)
    checks.append(result("runs_schema", reader.fieldnames == required_columns))
    checks.append(result("runs_unique", len(rows) == len({r['run_id'] for r in rows}) == 22))

    smoke = (ROOT / "scripts" / "smoke_test.py").read_text()
    checks.append(result("real_smoke_fail_closed",
                         "synthetic_smoke() if args.mode" in smoke and
                         "else real_smoke(args)" in smoke and
                         '"authorizes_campaign": bool(mode == "real" and passed)' in smoke))
    env = (ROOT / "env" / "build_mmcv_blackwell.sh").read_text()
    checks.append(result("mmcv_compatible_pin", "v2.1.0" in env and "v2.2.0" not in env))
    stats = (ROOT / "src" / "phyllo" / "stats" / "hypothesis_verdict.py").read_text()
    checks.append(result("h2_scale", "margin: float = 0.005" in stats))
    config = (ROOT / "configs" / "phyllo_segnext_mscan-l_512x512_40k.py").read_text()
    checks.append(result("test_not_validation", 'PHYLLO_VAL_LIST' in config and
                         'PHYLLO_COHORT_LIST' in config and "test_dataloader = val_dataloader" not in config))
    proc = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT)])
    checks.append(result("python_compile", proc.returncode == 0))
    for script in ROOT.rglob("*.sh"):
        proc = subprocess.run(["bash", "-n", str(script)])
        if proc.returncode:
            checks.append(result("shell_syntax", False, script.relative_to(ROOT)))
            break
    else:
        checks.append(result("shell_syntax", True))
    print("OVERALL:", "PASS" if all(checks) else "FAIL")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed artifact barrier for the PHYLLO evidence campaign."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
RESULTS = ROOT / "results"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_npz(path, expected_n):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing/empty result: {path}")
    with np.load(path, allow_pickle=True) as z:
        required = {"area_intersect", "area_union", "keys"}
        missing = required - set(z.files)
        if missing:
            raise SystemExit(f"{path}: missing arrays {sorted(missing)}")
        keys = tuple(str(x) for x in z["keys"].tolist())
        if len(keys) != expected_n:
            raise SystemExit(f"{path}: n={len(keys)} expected={expected_n}")
        if len(set(keys)) != expected_n:
            raise SystemExit(f"{path}: duplicate cohort keys")
        if z["area_intersect"].shape[0] != expected_n:
            raise SystemExit(f"{path}: area_intersect first dimension mismatch")
        if z["area_union"].shape[0] != expected_n:
            raise SystemExit(f"{path}: area_union first dimension mismatch")
    print(f"NPZ PASS n={expected_n} sha256={digest(path)} {path}")


def verify_record(path, run_id, model, arm, seed):
    if not path.is_file():
        raise SystemExit(f"missing run record: {path}")
    d = json.loads(path.read_text())
    expected = {
        "run_id": run_id,
        "compute_profile_id": "CP-02",
        "model": model,
        "arm": arm,
        "seed": seed,
        "status": "COMPLETED",
    }
    for key, value in expected.items():
        if d.get(key) != value:
            raise SystemExit(
                f"{path}: {key}={d.get(key)!r}, expected={value!r}"
            )
    print(f"RECORD PASS {run_id} {model}/{arm}/seed{seed}")


def verify_checkpoint(workdir):
    checkpoints = sorted(workdir.glob("iter_*.pth"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint: {workdir}")
    latest = checkpoints[-1]
    if latest.stat().st_size == 0:
        raise SystemExit(f"empty checkpoint: {latest}")
    print(f"CHECKPOINT PASS bytes={latest.stat().st_size} {latest}")


def verify_model_run(run_id, model, baseline, arm, seed, expected_n):
    workdir = WORK / model / arm / f"seed{seed}"
    record = workdir / "run_record.json"
    result = RESULTS / baseline / arm / f"seed{seed}.npz"
    verify_record(record, run_id, model, arm, seed)
    verify_checkpoint(workdir)
    verify_npz(result, expected_n)


def verify_segnext():
    run_number = 2
    for arm in ("default", "sanitized", "sizectrl"):
        for seed in (0, 1, 2):
            verify_model_run(
                f"RUN-{run_number:03d}",
                "segnext",
                "segnext-l",
                arm,
                seed,
                1557,
            )
            run_number += 1


def verify_deeplab():
    for offset, arm in enumerate(("default", "sanitized", "sizectrl")):
        verify_model_run(
            f"RUN-{11 + offset:03d}",
            "deeplab",
            "deeplabv3plus-r101",
            arm,
            0,
            1557,
        )


def verify_corrected():
    specs = (
        ("RUN-015", "segnext", "segnext-l"),
        ("RUN-016", "deeplab", "deeplabv3plus-r101"),
    )
    for run_id, model, baseline in specs:
        verify_model_run(run_id, model, baseline, "corrected", 0, 1552)


def verify_sensitivity():
    specs = (
        ("segnext-l", "raw"),
        ("segnext-l", "relabel"),
        ("deeplabv3plus-r101", "raw"),
        ("deeplabv3plus-r101", "relabel"),
    )
    for baseline, mode in specs:
        verify_npz(
            RESULTS / "sensitivity" / baseline / mode / "seed0.npz",
            1561,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("segnext", "deeplab", "corrected", "sensitivity"),
    )
    args = parser.parse_args()

    {
        "segnext": verify_segnext,
        "deeplab": verify_deeplab,
        "corrected": verify_corrected,
        "sensitivity": verify_sensitivity,
    }[args.stage]()

    print(f"ARTIFACT_BARRIER_PASS stage={args.stage}")


if __name__ == "__main__":
    main()

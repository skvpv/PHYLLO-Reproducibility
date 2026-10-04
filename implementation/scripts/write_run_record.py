#!/usr/bin/env python3
"""Write a checksummed, non-scientific execution record for one planned RUN ID."""
import argparse
import datetime
import hashlib
import json
import os


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {"run_id": args.run_id, "project_id": "PHYLLO-001",
               "compute_profile_id": "CP-02", "model": args.model, "arm": args.arm,
               "seed": int(args.seed), "status": args.status,
               "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "inputs": {name: {"path": path, "sha256": digest(path)} for name, path in {
                   "config": args.config, "train_list": args.train_list,
                   "val_list": args.val_list, "test_list": args.test_list}.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

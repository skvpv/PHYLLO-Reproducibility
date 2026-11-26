#!/usr/bin/env python3
"""Append-only failure/quarantine record; never deletes or overwrites run outputs."""
import argparse
import datetime
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--out-dir", default="quarantine")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(args.out_dir, f"{args.run_id}_{stamp}.json")
    payload = {"run_id": args.run_id, "status": "QUARANTINED", "reason": args.reason,
               "logs": args.log, "recorded_at": stamp,
               "policy": "outputs preserved; do not use for claim support"}
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(path)


if __name__ == "__main__":
    main()

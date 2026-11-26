#!/usr/bin/env python3
"""Capture exact runtime/package/checkpoint provenance without sensitive host identity."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", default=[], help="checkpoint/weight file")
    parser.add_argument("--out", default="results/runtime_provenance.json")
    args = parser.parse_args()
    freeze = subprocess.check_output(["python", "-m", "pip", "freeze", "--all"], text=True)
    data = {"compute_profile_id": "CP-02", "python": platform.python_version(),
            "platform": platform.system() + " " + platform.release(),
            "pip_freeze": freeze.splitlines(),
            "files": [{"basename": Path(p).name, "bytes": os.path.getsize(p), "sha256": sha(p)}
                      for p in args.file]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(data, open(args.out, "w", encoding="utf-8"), indent=2)
    print(args.out)


if __name__ == "__main__":
    main()

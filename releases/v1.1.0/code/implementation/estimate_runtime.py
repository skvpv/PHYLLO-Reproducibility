#!/usr/bin/env python3
"""Project the frozen campaign from both timed pilots; fail if 7 GPU-days is exceeded."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segnext", required=True)
    parser.add_argument("--deeplab", required=True)
    parser.add_argument("--out", default="results/runtime/runtime_estimate.json")
    args = parser.parse_args()
    seg = json.load(open(args.segnext)); dee = json.load(open(args.deeplab))
    components = {
        "segnext_9x40000": 9 * 40000 * seg["seconds_per_iteration_conservative"] / 3600,
        "deeplab_3x160000": 3 * 160000 * dee["seconds_per_iteration_conservative"] / 3600,
    }
    raw = sum(components.values()); buffered = raw * 1.20
    report = {"components_gpu_hours": components, "raw_gpu_hours": raw,
              "buffered_gpu_hours": buffered, "budget_gpu_hours": 168,
              "budget_pass": buffered <= 168,
              "policy": "If false, stop before evidence training and open a CR; never silently drop a frozen run."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["budget_pass"] else 2)


if __name__ == "__main__":
    main()

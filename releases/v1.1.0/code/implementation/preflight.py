#!/usr/bin/env python3
"""Fail-closed implementation preflight for CP-02.

Every check is blocking. There are no SKIP-OK branches. This resolves the CP-02
UNKNOWN fields on the allocated compute node and must pass before the real smoke gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXPECTED_COMMIT = "1a3dd4d9224bcc97a5850af7dd1c423abc24eae0"
REQUIRED_IMPORTS = ["torch", "torchvision", "mmengine", "mmcv", "mmseg", "clip",
                    "imagehash", "numpy", "scipy", "pandas", "cv2", "yaml"]


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail=""):
        self.rows.append({"check": name, "pass": bool(passed), "detail": str(detail)})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def read_list(path):
    with open(path, encoding="utf-8") as handle:
        return [x.strip() for x in handle if x.strip()]


def verify_package_manifest(checks: Checks):
    path = ROOT / "manifests" / "PACKAGE_MANIFEST.json"
    data = json.load(open(path, encoding="utf-8"))
    failures = []
    for entry in data["files"]:
        target = ROOT / entry["path"]
        if not target.is_file():
            failures.append(f"missing:{entry['path']}")
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            failures.append(f"hash:{entry['path']}")
    checks.add("package.manifest", not failures,
               "all declared hashes match" if not failures else failures[:8])


def run(args) -> int:
    c = Checks()
    for module in REQUIRED_IMPORTS:
        try:
            imported = importlib.import_module(module)
            c.add(f"import:{module}", True, getattr(imported, "__version__", "present"))
        except Exception as exc:
            c.add(f"import:{module}", False, repr(exc))

    try:
        from packaging.version import Version
        import mmcv, mmseg, torch
        versions_ok = (torch.__version__.startswith("2.7.1") and
                       Version("2.0.0rc4") <= Version(mmcv.__version__) < Version("2.2.0") and
                       mmseg.__version__ == "1.2.2")
        c.add("environment.version_matrix", versions_ok,
              f"torch={torch.__version__} mmcv={mmcv.__version__} mmseg={mmseg.__version__}")
        available = torch.cuda.is_available()
        c.add("cuda.available", available, available)
        count = torch.cuda.device_count()
        c.add("cuda.device_count", count == 1, f"allocated visible devices={count}")
        capability = torch.cuda.get_device_capability(0) if available and count else None
        c.add("cuda.sm120", capability == (12, 0), capability)
        if not available or count != 1:
            raise RuntimeError("one CUDA device is required")
        free, total = torch.cuda.mem_get_info(0)
        c.add("cuda.allocatable_vram", total > 0,
              f"total={total / 2**30:.2f}GiB free={free / 2**30:.2f}GiB")
        x = torch.randn(256, 256, device="cuda")
        y = (x @ x).sum()
        y.backward() if y.requires_grad else None
        c.add("cuda.kernel", bool(torch.isfinite(y)), "CUDA matmul executed")
    except Exception as exc:
        c.add("cuda.environment_block", False, f"{exc}\n{traceback.format_exc()}")

    try:
        import torch
        from mmcv.ops import nms
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]], device="cuda")
        scores = torch.tensor([.9, .8], device="cuda")
        dets, keep = nms(boxes, scores, .5)
        c.add("mmcv.compiled_ops", keep.numel() == 1,
              f"NMS indices={keep.tolist()} detections={tuple(dets.shape)}")
    except Exception as exc:
        c.add("mmcv.compiled_ops", False, repr(exc))

    repo = Path(args.repo).resolve()
    c.add("repo.train_py", (repo / "tools" / "train.py").is_file(), repo)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        c.add("repo.commit", commit == EXPECTED_COMMIT, commit)
    except Exception as exc:
        c.add("repo.commit", False, repr(exc))

    # Dataset/archive evidence is mandatory.
    from phyllo.dataset_prep.prepare import verify_archive
    try:
        archive = verify_archive(args.archive)
        c.add("dataset.archive_md5", archive["match"], archive)
    except Exception as exc:
        c.add("dataset.archive_md5", False, repr(exc))
    try:
        report = json.load(open(args.dataset_report, encoding="utf-8"))
        c.add("dataset.preparation", report.get("dataset_prep_pass") is True, report)
    except Exception as exc:
        c.add("dataset.preparation", False, repr(exc))

    lists = {"train": args.train_list, "val": args.val_list, "cohort": args.cohort_list}
    expected = {"train": 5359, "val": 846, "cohort": 1557}
    loaded = {}
    for name, path in lists.items():
        try:
            loaded[name] = read_list(path)
            ok = len(loaded[name]) == expected[name] and len(loaded[name]) == len(set(loaded[name]))
            c.add(f"dataset.{name}_list", ok, f"n={len(loaded[name])}")
        except Exception as exc:
            c.add(f"dataset.{name}_list", False, repr(exc))
    if len(loaded) == 3:
        ids = {name: {x.rsplit(".", 1)[0] for x in values} for name, values in loaded.items()}
        overlaps = {"train_val": len(ids["train"] & ids["val"]),
                    "train_test": len(ids["train"] & ids["cohort"]),
                    "val_test": len(ids["val"] & ids["cohort"])}
        c.add("dataset.partition_disjoint", not any(overlaps.values()), overlaps)

    os.environ.update({
        "PHYLLO_PLANTSEG_REPO": str(repo), "PHYLLO_TRAIN_LIST": args.train_list,
        "PHYLLO_VAL_LIST": args.val_list, "PHYLLO_COHORT_LIST": args.cohort_list,
        "PHYLLO_IMG_ROOT": args.img_root, "PHYLLO_ANN_ROOT": args.ann_root,
        "PHYLLO_SEED": "0",
    })
    for config in sorted(Path(args.config_dir).glob("phyllo_*.py")):
        try:
            from mmengine.config import Config
            from mmseg.registry import DATASETS
            from mmseg.utils import register_all_modules
            register_all_modules(init_default_scope=True)
            cfg = Config.fromfile(str(config))
            no_test_validation = (cfg.val_dataloader.dataset.ann_file == args.val_list and
                                  cfg.test_dataloader.dataset.ann_file == args.cohort_list)
            c.add(f"config:{config.name}", no_test_validation,
                  "parsed; validation and test lists are distinct")
            expected_iters = 40000 if "segnext" in config.name else 160000
            expected_batch = 16 if "segnext" in config.name else 4
            frozen_ok = (cfg.train_cfg.max_iters == expected_iters and
                         cfg.train_dataloader.batch_size == expected_batch and
                         cfg.model.decode_head.num_classes == 116)
            c.add(f"config.frozen_fields:{config.name}", frozen_ok,
                  f"iters={cfg.train_cfg.max_iters} batch={cfg.train_dataloader.batch_size} "
                  f"classes={cfg.model.decode_head.num_classes}")
            dataset = DATASETS.build(cfg.train_dataloader.dataset)
            sample = dataset[0]
            c.add(f"dataset.build:{config.name}", len(dataset) == 5359 and bool(sample),
                  f"n={len(dataset)} first sample loaded through real pipeline")
        except Exception as exc:
            c.add(f"config:{config.name}", False, repr(exc))

    try:
        os.makedirs(args.out_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.out_dir) as handle:
            handle.write(b"ok"); handle.flush()
        c.add("io.output_writable", True, args.out_dir)
    except Exception as exc:
        c.add("io.output_writable", False, repr(exc))

    # CP-02 requires Slurm; its absence is not a skip.
    sbatch = subprocess.run(["bash", "-lc", "command -v sbatch"], capture_output=True,
                            text=True).stdout.strip()
    c.add("scheduler.sbatch_present", bool(sbatch), sbatch or "not found")
    if sbatch:
        os.makedirs(ROOT / "logs", exist_ok=True)
        failures = []
        for script in sorted((ROOT / "slurm").glob("*.sh")):
            proc = subprocess.run(["sbatch", "--test-only", str(script)],
                                  capture_output=True, text=True)
            if proc.returncode:
                failures.append({"script": script.name, "stderr": proc.stderr[-300:]})
        c.add("scheduler.scripts", not failures, failures or "all sbatch --test-only PASS")

    try:
        verify_package_manifest(c)
    except Exception as exc:
        c.add("package.manifest", False, repr(exc))

    passed = all(row["pass"] for row in c.rows)
    report = {"run_id": "RUN-021", "preflight_pass": passed,
              "compute_profile_id": "CP-02", "checks": c.rows}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("PREFLIGHT:", "PASS" if passed else "FAIL — SMOKE/CAMPAIGN BLOCKED")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--ann-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--dataset-report", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--cohort-list", required=True)
    parser.add_argument("--out-dir", default="preflight_out")
    parser.add_argument("--out", default="preflight_out/preflight_report.json")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scientific smoke gate.

``--mode synthetic`` validates only the harness and can never authorize evidence runs.
``--mode real`` is fail-closed: it reads real PlantSeg masks/lists, invokes the actual
MMSeg metric, the actual pHash+CLIP detector path, and builds/runs/backpropagates through
both frozen baseline models on CUDA. No fallback or synthetic substitution is allowed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phyllo.dataset_prep.prepare import DEFECTIVE_TEST_IDS, canonical_id
from phyllo.leakage.detector import detect
from phyllo.metrics.miou import per_image_confusion, miou_116
from phyllo.probes.probes import (
    PROBE_CATALOG, sp01_mask_encoding, sp02_dual_metric, sp03_metric_agreement,
    sp04_detector_positive, sp05_cohort, sp06_no_cohort_in_train,
    sp07_sizectrl_count_match, sp08_grad_valid,
)
from phyllo.sanitize.arm_manifests import build_arm_manifests


def read_list(path: str):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def finalize(results, mode: str) -> dict:
    evaluated = {r.id for r in results}
    missing = sorted(set(PROBE_CATALOG) - evaluated)
    passed = not missing and all(r.passed for r in results)
    report = {
        "run_id": "RUN-022",
        "mode": mode,
        "authorizes_campaign": bool(mode == "real" and passed),
        "smoke_pass": bool(mode == "real" and passed),
        "harness_pass": bool(passed),
        "all_probes_evaluated": not missing,
        "unevaluated_probes": missing,
        "probes": [{"id": r.id, "pass": r.passed,
                    "mechanism": PROBE_CATALOG[r.id], "detail": r.detail}
                   for r in results],
    }
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.id}: {result.detail}")
    print("SMOKE GATE:", "PASS" if report["smoke_pass"] else "BLOCKED")
    return report


def synthetic_smoke() -> dict:
    """Unit-like harness diagnostic; deliberately not a campaign gate."""
    rng = np.random.default_rng(0)
    results = [sp01_mask_encoding(rng.integers(0, 116, size=(32, 32)))]
    gt = np.zeros((32, 32), dtype=np.int64); gt[:8] = 5
    pred = gt.copy(); pred[0, 0] = 0
    ai, au, _, _ = per_image_confusion(pred, gt)
    results += [sp02_dual_metric(ai, au), sp03_metric_agreement(ai, au, miou_116(ai, au))]
    keys = ["train/a.jpg", "test/b.jpg", "train/c.jpg"]
    det = detect(keys, np.array([15, 15, 42], dtype=np.uint64),
                 np.eye(3, 512, dtype=np.float32), ["train", "test", "train"])
    results.append(sp04_detector_positive(det, (keys[0], keys[1])))
    cohort = [f"test/img_{i:04d}.jpg" for i in range(1557)]
    results += [sp05_cohort(cohort, []), sp06_no_cohort_in_train(["train/a.jpg"], cohort),
                sp07_sizectrl_count_match({"train": 2, "val": 1},
                                          {"train": 2, "val": 1}),
                sp08_grad_valid(1.0, 0.5)]
    return finalize(results, "synthetic")


def mmseg_reference(pred: np.ndarray, gt: np.ndarray) -> float:
    import torch
    from mmseg.evaluation.metrics.iou_metric import IoUMetric

    ai, au, _, _ = IoUMetric.intersect_and_union(
        torch.from_numpy(pred), torch.from_numpy(gt), 116, 255)
    ai = ai.cpu().numpy(); au = au.cpu().numpy()
    valid = au > 0
    return float(np.mean(ai[valid] / au[valid]))


def baseline_step(config_path: str) -> tuple[float, float]:
    """Build the actual configured segmentor; run tensor forward and backward on CUDA."""
    import torch
    import torch.nn.functional as F
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.copy()

    def strip_pretrained(node):
        if isinstance(node, dict):
            node.pop("pretrained", None)
            if "init_cfg" in node:
                node["init_cfg"] = None
            for value in node.values():
                strip_pretrained(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                strip_pretrained(value)

    strip_pretrained(model_cfg)
    model = MODELS.build(model_cfg).cuda().train()
    x = torch.randn(2, 3, 128, 128, device="cuda")
    logits = model(x, data_samples=None, mode="tensor")
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)
    target = torch.randint(0, 116, (2, 128, 128), device="cuda")
    loss = F.cross_entropy(logits, target)
    loss.backward()
    grad = sum(float(p.grad.norm().item()) for p in model.parameters()
               if p.grad is not None and torch.isfinite(p.grad).all())
    return float(loss.item()), grad


def real_smoke(args) -> dict:
    required = {
        "ann_root": args.ann_root, "img_root": args.img_root, "repo": args.repo,
        "cohort": args.cohort_list, "train_list": args.train_list,
        "val_list": args.val_list,
        "config_dir": args.config_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"real mode requires: {', '.join(missing)}")
    os.environ.update({"PHYLLO_PLANTSEG_REPO": args.repo,
                       "PHYLLO_IMG_ROOT": args.img_root,
                       "PHYLLO_ANN_ROOT": args.ann_root,
                       "PHYLLO_TRAIN_LIST": args.train_list,
                       "PHYLLO_VAL_LIST": args.val_list,
                       "PHYLLO_COHORT_LIST": args.cohort_list,
                       "PHYLLO_SEED": "0"})
    for path in [args.cohort_list, args.train_list, args.val_list]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    cohort = read_list(args.cohort_list)
    sample_key = cohort[0]
    split, name = sample_key.split("/", 1)
    from PIL import Image
    mask = np.asarray(Image.open(Path(args.ann_root, split, Path(name).stem + ".png")))
    results = [sp01_mask_encoding(mask)]

    # Actual MMSeg IoUMetric call on a deterministic perturbation of an actual mask.
    pred = mask.astype(np.int64).copy()
    pred.flat[0] = 0 if pred.flat[0] else 1
    ai, au, _, _ = per_image_confusion(pred, mask)
    results += [sp02_dual_metric(ai, au),
                sp03_metric_agreement(ai, au, mmseg_reference(pred, mask))]

    # Actual frozen preprocessing/model on one real image duplicated intentionally.
    from phyllo.leakage.phash_clip import compute_phash_clip
    image_path = str(Path(args.img_root, split, name))
    p, e, _ = compute_phash_clip([image_path, image_path],
                                 ["train/smoke_dup.jpg", "test/smoke_dup.jpg"],
                                 device="cuda", batch_size=2)
    detector = detect(["train/smoke_dup.jpg", "test/smoke_dup.jpg"], p, e,
                      ["train", "test"])
    results.append(sp04_detector_positive(
        detector, ("train/smoke_dup.jpg", "test/smoke_dup.jpg")))

    defective_ids = list(DEFECTIVE_TEST_IDS)
    results.append(sp05_cohort(cohort, defective_ids))
    train = read_list(args.train_list); val = read_list(args.val_list)
    results.append(sp06_no_cohort_in_train(train, cohort))
    # Exercise the actual arm-builder on a small deterministic fixture drawn from real
    # dataset keys. This tests the mechanism before, and independently of, H1 evidence.
    clusters = [[cohort[0], train[0], val[0]]]
    pool = {"train": set(train[:20]), "val": set(val[:20])}
    classes = {key: 1 + i % 3 for i, key in enumerate(train[:20] + val[:20])}
    fixture = build_arm_manifests(clusters, set(cohort), pool,
                                  {train[0], val[0]}, classes)
    results.append(sp07_sizectrl_count_match(fixture.sanitized_counts,
                                             fixture.sizectrl_counts))

    cfgs = [Path(args.config_dir, "phyllo_segnext_mscan-l_512x512_40k.py"),
            Path(args.config_dir, "phyllo_deeplabv3plus_r101_512x512_160k.py")]
    losses, grads = zip(*(baseline_step(str(cfg)) for cfg in cfgs))
    results.append(sp08_grad_valid(float(sum(losses)), float(min(grads))))
    report = finalize(results, "real")
    report["baseline_steps"] = [{"config": str(c), "loss": l, "grad_norm": g}
                                for c, l, g in zip(cfgs, losses, grads)]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    parser.add_argument("--img-root", default="")
    parser.add_argument("--ann-root", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--cohort-list", default="")
    parser.add_argument("--train-list", default="")
    parser.add_argument("--val-list", default="")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--out", default="smoke_out/smoke_report.json")
    args = parser.parse_args()
    report = synthetic_smoke() if args.mode == "synthetic" else real_smoke(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    raise SystemExit(0 if (report["harness_pass"] if args.mode == "synthetic"
                           else report["smoke_pass"]) else 1)


if __name__ == "__main__":
    main()

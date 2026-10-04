"""Evaluate a trained arm on the FIXED 1557 cohort -> per-image confusion npz.

Produces exactly the arrays the paired bootstrap consumes (area_intersect / area_union /
area_pred / area_label, each [N,116]) in the SAME cohort image order for every arm and
seed, so the bootstrap indices are paired. Cohort order is read from a single frozen
cohort list, guaranteeing identical ordering across all runs.

Kept deliberately framework-thin: it uses mmseg's inference_model to obtain a prediction
per image, then computes confusion with the frozen metric definition in phyllo.metrics.
mmseg's own IoUMetric is cross-checked in the smoke test (metric-correctness probe SP-03).

Heavy imports are lazy so unit tests importing the package do not require mmseg.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List
import numpy as np

from phyllo.metrics.miou import (per_image_confusion, NUM_CLASSES, miou_116,
                                 miou_disease, macc_116, macc_disease,
                                 per_class_iou, boundary_confusion, boundary_scores)


def _load_cohort(cohort_list: str) -> List[str]:
    with open(cohort_list) as f:
        return [ln.strip() for ln in f if ln.strip()]


def evaluate(config: str, checkpoint: str, cohort_list: str,
             img_root: str, ann_root: str, out_npz: str,
             device: str = "cuda:0", img_suffix: str = ".jpg",
             seg_suffix: str = ".png", relabel_map: str = "", run_id: str = "") -> dict:
    import mmcv
    from mmseg.apis import init_model, inference_model

    keys = _load_cohort(cohort_list)
    n = len(keys)
    ai = np.zeros((n, NUM_CLASSES), dtype=np.int64)
    au = np.zeros((n, NUM_CLASSES), dtype=np.int64)
    ap = np.zeros((n, NUM_CLASSES), dtype=np.int64)
    al = np.zeros((n, NUM_CLASSES), dtype=np.int64)
    boundary = np.zeros((n, 3), dtype=np.float64)
    relabel = json.load(open(relabel_map)) if relabel_map else {}

    model = init_model(config, checkpoint, device=device)
    for i, key in enumerate(keys):
        split, name = key.split("/", 1)
        stem = name[:-len(img_suffix)] if name.endswith(img_suffix) else os.path.splitext(name)[0]
        img_path = os.path.join(img_root, split, stem + img_suffix)
        gt_path = os.path.join(ann_root, split, stem + seg_suffix)
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
        gt = np.array(mmcv.imread(gt_path, flag="grayscale")).astype(np.int64)
        if key in relabel:
            target = int(relabel[key])
            gt = np.where(gt > 0, target, 0).astype(np.int64)
        a_i, a_u, a_p, a_l = per_image_confusion(pred, gt)
        ai[i], au[i], ap[i], al[i] = a_i, a_u, a_p, a_l
        boundary[i] = boundary_confusion(pred, gt)

    np.savez_compressed(out_npz, keys=np.array(keys), area_intersect=ai,
                        area_union=au, area_pred=ap, area_label=al,
                        boundary_tp_fp_fn=boundary)
    tot_i, tot_u, tot_l = ai.sum(0), au.sum(0), al.sum(0)
    b = boundary.sum(0)
    class_iou = per_class_iou(tot_i, tot_u)
    summary = {"run_id": run_id or None, "n": n, "cohort_list": cohort_list,
               "checkpoint": checkpoint,
               "mIoU_116": miou_116(tot_i, tot_u),
               "mIoU_disease": miou_disease(tot_i, tot_u),
               "mAcc_116": macc_116(tot_i, tot_l),
               "mAcc_disease": macc_disease(tot_i, tot_l),
               "per_class_iou": [float(x) if np.isfinite(x) else None for x in class_iou],
               **boundary_scores(*b),
               "relabel_map": relabel_map or None,
               "out_npz": out_npz}
    with open(out_npz + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Per-image cohort evaluation -> confusion npz")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cohort-list", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--ann-root", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--img-suffix", default=".jpg")
    ap.add_argument("--relabel-map", default="")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    s = evaluate(a.config, a.checkpoint, a.cohort_list, a.img_root, a.ann_root,
                 a.out_npz, a.device, a.img_suffix, relabel_map=a.relabel_map,
                 run_id=a.run_id)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
